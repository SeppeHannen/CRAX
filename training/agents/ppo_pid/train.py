"""PPO-PID Lagrange training.

Thin wrapper around the base PPO trainer with PID-based Lagrangian constraint handling.
"""

from typing import Any, Callable, Dict, Mapping, Optional, Tuple

import flax
import jax
import jax.numpy as jnp

from crax import base
from crax import envs
from training import rounds
from training import types
from training.agents.ppo import train as ppo_train
from training.agents.ppo import networks as ppo_networks
from training.agents.ppo_lag import losses as ppo_lag_losses

Metrics = types.Metrics
TrainingState = ppo_train.TrainingState


@flax.struct.dataclass
class PIDLagrangeState:
    """The controller's state, kept in ``TrainingState.aux_state``: the multiplier
    the loss uses and the three terms the controller updates it from. Every
    array has shape ``[1]``."""

    lagrange_multiplier: jnp.ndarray  # λ, the multiplier on the cost advantage
    integral: jnp.ndarray  # clipped running sum of the violation
    previous_violation: jnp.ndarray  # last round's violation, for the derivative
    violation_derivative_ema: jnp.ndarray  # smoothed change of the violation


def train(
        environment: envs.Env,
        num_timesteps: int,
        episode_length: int,
        max_devices_per_host: Optional[int] = None,
        wrap_env: bool = True,
        augment_pixels: bool = False,
        vision_kwargs: Optional[Dict[str, Any]] = None,
        num_envs: int = 1,
        action_repeat: int = 1,
        wrap_env_fn: Optional[Callable[[Any], Any]] = None,
        randomization_fn: Optional[
            Callable[[base.System, jnp.ndarray], Tuple[base.System, base.System]]
        ] = None,
        learning_rate: float = 1e-4,
        entropy_cost: float = 1e-4,
        discounting: float = 0.9,
        unroll_length: int = 10,
        batch_size: int = 32,
        num_minibatches: int = 16,
        num_updates_per_batch: int = 2,
        num_resets_per_eval: int = 0,
        normalize_observations: bool = False,
        reward_scaling: float = 1.0,
        clipping_epsilon: float = 0.3,
        gae_lambda: float = 0.95,
        max_grad_norm: Optional[float] = None,
        normalize_advantage: bool = True,
        network_factory: types.NetworkFactory[ppo_networks.PPONetworks] = None,
        seed: int = 0,
        safety_bound: float = 0.0,
        initial_lambda_lagr: float = 0.0,
        pid_kp: float = 10.0,
        pid_ki: float = 0.01,
        pid_kd: float = 0.01,
        pid_integral_clip: float = 1.0,
        pid_lambda_clip: float = 1e6,
        pid_deriv_ema_beta: float = 0.95,
        num_evals: int = 0,
        eval_env: Optional[envs.Env] = None,
        num_eval_envs: int = 128,
        deterministic_eval: bool = False,
        buffer_size: int = 1000,
        log_training_metrics: bool = True,
        training_metrics_steps: Optional[int] = None,
        progress_fn: Callable[[int, Metrics], None] = lambda *args: None,
        policy_params_fn: Callable[..., None] = lambda *args: None,
        save_checkpoint_path: Optional[str] = None,
        restore_checkpoint_path: Optional[str] = None,
        restore_params: Optional[Any] = None,
        restore_value_fn: bool = True,
        pretrained_params: Optional[Any] = None,
        init_cost_value_from: str = 'value',
        # per-round host hook and named evaluation distributions (see ppo.train)
        round_hook: Optional[rounds.RoundHook] = None,
        evaluation_wrap_env_fns: Optional[Mapping[str, Callable[[Any], Any]]] = None,
):
    """PPO-PID Lagrange training."""
    per_step_safety_bound = safety_bound / episode_length if episode_length else safety_bound

    if network_factory is None:
        def network_factory(obs_size, action_size, **kwargs):
            return ppo_networks.make_ppo_networks(
                obs_size, action_size, cost_value_hidden_layer_sizes=(256,) * 5, **kwargs)

    def post_step_fn(training_state: TrainingState, metrics: Metrics, data: types.Transition) -> Tuple[TrainingState, Metrics]:
        del data  # the controller reads the batch's mean cost from the loss metrics
        controller: PIDLagrangeState = training_state.aux_state
        avg_cost = jnp.mean(metrics["mean_cost"][-1])
        violation = (avg_cost - per_step_safety_bound).reshape((1,))
        integral = jnp.clip(controller.integral + violation, -pid_integral_clip, pid_integral_clip)
        violation_derivative_ema = (
            pid_deriv_ema_beta * controller.violation_derivative_ema
            + (1.0 - pid_deriv_ema_beta) * (violation - controller.previous_violation)
        )
        pid_output = pid_kp * violation + pid_ki * integral + pid_kd * violation_derivative_ema
        lagrange_multiplier = jnp.clip(jax.nn.relu(controller.lagrange_multiplier + pid_output), 0.0, pid_lambda_clip)
        new_state = training_state.replace(aux_state=PIDLagrangeState(
            lagrange_multiplier=lagrange_multiplier, integral=integral, previous_violation=violation,
            violation_derivative_ema=violation_derivative_ema,
        ))
        # `lambda_lagr` is the upstream metric key every constrained agent logs λ under.
        return new_state, {"lambda_lagr": lagrange_multiplier, "pid/violation": violation}

    def lagrange_loss_fn(params, normalizer_params, data, rng, aux_state: PIDLagrangeState,
                         ppo_network=None, entropy_cost=1e-4, discounting=0.9,
                         reward_scaling=1.0, gae_lambda=0.95, clipping_epsilon=0.3,
                         normalize_advantage=True):
        return ppo_lag_losses.compute_ppo_lagrange_loss(
            params=params, normalizer_params=normalizer_params, data=data, rng=rng,
            ppo_network=ppo_network, lambda_lagr=aux_state.lagrange_multiplier, safety_bound=per_step_safety_bound,
            entropy_cost=entropy_cost, discounting=discounting, reward_scaling=reward_scaling,
            gae_lambda=gae_lambda, clipping_epsilon=clipping_epsilon, normalize_advantage=normalize_advantage)

    # The round hook's view of the rollout: the advantages the loss is about to use, under the λ in force.
    def compute_learning_signals(params, normalizer_params, data, aux_state: PIDLagrangeState,
                                 ppo_network, discounting, reward_scaling, gae_lambda, normalize_advantage):
        return ppo_lag_losses.compute_learning_signals(
            params=params, normalizer_params=normalizer_params, data=data, lagrange_multiplier=aux_state.lagrange_multiplier,
            ppo_network=ppo_network, discounting=discounting, reward_scaling=reward_scaling,
            gae_lambda=gae_lambda, normalize_advantage=normalize_advantage)

    learning_signals_fn = rounds.LearningSignalsFunction(
        produces=ppo_lag_losses.LEARNING_SIGNALS, compute=compute_learning_signals
    )

    def init_aux_state_fn() -> PIDLagrangeState:
        return PIDLagrangeState(
            lagrange_multiplier=jnp.array([initial_lambda_lagr], dtype=jnp.float32),
            integral=jnp.zeros((1,), dtype=jnp.float32),
            previous_violation=jnp.zeros((1,), dtype=jnp.float32),
            violation_derivative_ema=jnp.zeros((1,), dtype=jnp.float32),
        )

    effective_restore_params = pretrained_params if pretrained_params is not None else restore_params

    return ppo_train.train(
        environment=environment, num_timesteps=num_timesteps, max_devices_per_host=max_devices_per_host,
        wrap_env=wrap_env, augment_pixels=augment_pixels, vision_kwargs=vision_kwargs,
        num_envs=num_envs, episode_length=episode_length, action_repeat=action_repeat,
        wrap_env_fn=wrap_env_fn, randomization_fn=randomization_fn, learning_rate=learning_rate,
        entropy_cost=entropy_cost, discounting=discounting, unroll_length=unroll_length,
        batch_size=batch_size, num_minibatches=num_minibatches, num_updates_per_batch=num_updates_per_batch,
        num_resets_per_eval=num_resets_per_eval, normalize_observations=normalize_observations,
        reward_scaling=reward_scaling, clipping_epsilon=clipping_epsilon, gae_lambda=gae_lambda,
        max_grad_norm=max_grad_norm, normalize_advantage=normalize_advantage, network_factory=network_factory,
        seed=seed, num_evals=num_evals, eval_env=eval_env, num_eval_envs=num_eval_envs,
        deterministic_eval=deterministic_eval, buffer_size=buffer_size, log_training_metrics=log_training_metrics,
        training_metrics_steps=training_metrics_steps, progress_fn=progress_fn, policy_params_fn=policy_params_fn,
        save_checkpoint_path=save_checkpoint_path, restore_checkpoint_path=restore_checkpoint_path,
        restore_params=effective_restore_params, restore_value_fn=restore_value_fn,
        loss_fn=lagrange_loss_fn, post_step_fn=post_step_fn,
        extra_fields=("truncation", "episode_metrics", "episode_done", "cost"),
        init_aux_state_fn=init_aux_state_fn,
        round_hook=round_hook,
        learning_signals_fn=learning_signals_fn,
        evaluation_wrap_env_fns=evaluation_wrap_env_fns)
