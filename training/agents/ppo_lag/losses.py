"""PPO-Lagrange loss functions for constrained RL."""

from typing import Any, FrozenSet, Tuple

import jax
import jax.numpy as jnp

from training import networks
from training import rounds
from training import types
from training.rounds import Signal
from training.agents.ppo import networks as ppo_networks
from training.agents.ppo.losses import compute_gae
from training.types import Params

# Re-export for backward compatibility
from training.agents.ppo.losses import PPONetworkParams, compute_ppo_loss, with_shared_latent

__all__ = ["PPONetworkParams", "compute_gae", "compute_ppo_loss", "compute_ppo_lagrange_loss", "compute_learning_signals",
           "lagrangian_advantage", "LEARNING_SIGNALS"]

# What a Lagrangian agent (two critics and a multiplier) can tell a round hook about its rollout.
LEARNING_SIGNALS: FrozenSet[Signal] = frozenset({Signal.REWARD_ADVANTAGE, Signal.COST_ADVANTAGE, Signal.LAGRANGIAN_ADVANTAGE})


def lagrangian_advantage(
    advantages: jnp.ndarray, cost_advantages: jnp.ndarray, lagrange_multiplier: jnp.ndarray, normalize_advantage: bool
) -> jnp.ndarray:
    """The advantage the policy loss weights each transition by: ``A_r − λ A_c``,
    each advantage first standardised over the batch it is given when
    ``normalize_advantage``. ``lagrange_multiplier`` is λ per transition in the
    advantages' layout, or one value broadcast to all. The one rule for both the
    loss and the learning signals, so a score on the Lagrangian objective is the
    loss's own quantity."""
    if normalize_advantage:
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        cost_advantages = (cost_advantages - cost_advantages.mean()) / (cost_advantages.std() + 1e-8)
    return advantages - lagrange_multiplier * cost_advantages


def _time_first(lagrange_multiplier: jnp.ndarray, data_shape: Tuple[int, ...]) -> jnp.ndarray:
    """λ as one value per transition, laid out ``[T, B]`` like the time-first data.

    ``lagrange_multiplier`` is either ``[1]`` (a scalar constraint) or already one value
    per transition in the data's ``[B, T]`` layout (a per-context constraint)."""
    return jnp.swapaxes(jnp.broadcast_to(lagrange_multiplier, data_shape), 0, 1)


def compute_learning_signals(
    params: PPONetworkParams,
    normalizer_params: Any,
    data: types.Transition,
    lagrange_multiplier: jnp.ndarray,
    ppo_network: ppo_networks.PPONetworks,
    discounting: float,
    reward_scaling: float,
    gae_lambda: float,
    normalize_advantage: bool,
) -> rounds.LearningSignals:
    """The reward, cost and Lagrangian advantage of every transition in ``data``
    (leading dimension ``[B, T]``), each as ``[B, T]``, under the critics in
    ``params`` and the multiplier ``lagrange_multiplier`` (λ: ``[1]``, or one per
    transition as ``[B, T]``).

    The same estimates :func:`compute_ppo_lagrange_loss` makes, computed once over
    the whole batch so that a round hook can read them per transition (the loss
    sees shuffled minibatches; the Lagrangian advantage is therefore standardised
    over the batch here and over a minibatch there).
    """
    lagrange_multiplier = _time_first(lagrange_multiplier, data.reward.shape)
    data = jax.tree_util.tree_map(lambda x: jnp.swapaxes(x, 0, 1), data)
    obs = with_shared_latent(ppo_network, params, normalizer_params, data.observation)
    terminal_obs = jax.tree_util.tree_map(lambda x: x[-1], data.next_observation)
    terminal_obs = with_shared_latent(ppo_network, params, normalizer_params, terminal_obs)
    truncation = data.extras['state_extras']['truncation']
    termination = (1 - data.discount) * (1 - truncation)

    def advantages(network: networks.FeedForwardNetwork, network_params: Params, per_step: jnp.ndarray) -> jnp.ndarray:
        _, estimate = compute_gae(
            truncation=truncation, termination=termination, rewards=per_step,
            values=network.apply(normalizer_params, network_params, obs),
            bootstrap_value=network.apply(normalizer_params, network_params, terminal_obs),
            lambda_=gae_lambda, discount=discounting,
        )
        return estimate

    reward_advantage = advantages(ppo_network.value_network, params.value, data.reward * reward_scaling)
    cost_advantage = advantages(ppo_network.cost_value_network, params.cost_value, data.extras['state_extras']['cost'])
    batch_first = lambda per_step: jnp.swapaxes(per_step, 0, 1)
    return {
        Signal.REWARD_ADVANTAGE: batch_first(reward_advantage),
        Signal.COST_ADVANTAGE: batch_first(cost_advantage),
        Signal.LAGRANGIAN_ADVANTAGE: batch_first(
            lagrangian_advantage(reward_advantage, cost_advantage, lagrange_multiplier, normalize_advantage)
        ),
    }


def compute_ppo_lagrange_loss(
    params: PPONetworkParams,
    normalizer_params: Any,
    data: types.Transition,
    rng: jnp.ndarray,
    ppo_network: ppo_networks.PPONetworks,
    lambda_lagr: jnp.ndarray,
    safety_bound: float = 0.0,
    entropy_cost: float = 1e-4,
    discounting: float = 0.9,
    reward_scaling: float = 1.0,
    gae_lambda: float = 0.95,
    clipping_epsilon: float = 0.3,
    normalize_advantage: bool = True,
) -> Tuple[jnp.ndarray, types.Metrics]:
    """Computes PPO-Lagrange loss for constrained RL.

    Args:
      params: Network parameters (must include cost_value).
      normalizer_params: Parameters of the normalizer.
      data: Transition that with leading dimension [B, T]. extra fields required
        are ['state_extras']['truncation'] ['policy_extras']['raw_action']
        ['policy_extras']['log_prob'] and ['state_extras']['cost']
      rng: Random key
      ppo_network: PPO networks (must include cost_value_network).
      lambda_lagr: Current Lagrange multiplier: ``[1]``, or one per transition as ``[B, T]``
      safety_bound: Safety constraint bound (per-step)
      entropy_cost: entropy cost.
      discounting: discounting,
      reward_scaling: reward multiplier.
      gae_lambda: General advantage estimation lambda.
      clipping_epsilon: Policy loss clipping epsilon
      normalize_advantage: whether to normalize advantage estimate

    Returns:
      A tuple (loss, metrics)
    """
    parametric_action_distribution = ppo_network.parametric_action_distribution
    policy_apply = ppo_network.policy_network.apply
    value_apply = ppo_network.value_network.apply
    cost_value_apply = ppo_network.cost_value_network.apply

    # Put the time dimension first.
    lambda_lagr = _time_first(lambda_lagr, data.reward.shape)
    data = jax.tree_util.tree_map(lambda x: jnp.swapaxes(x, 0, 1), data)
    obs = with_shared_latent(ppo_network, params, normalizer_params, data.observation)
    policy_logits = policy_apply(normalizer_params, params.policy, obs)

    baseline = value_apply(normalizer_params, params.value, obs)
    cost_baseline = cost_value_apply(normalizer_params, params.cost_value, obs)

    terminal_obs = jax.tree_util.tree_map(lambda x: x[-1], data.next_observation)
    terminal_obs = with_shared_latent(ppo_network, params, normalizer_params, terminal_obs)
    bootstrap_value = value_apply(normalizer_params, params.value, terminal_obs)
    bootstrap_cost_value = cost_value_apply(normalizer_params, params.cost_value, terminal_obs)

    rewards = data.reward * reward_scaling
    state_extras = data.extras.get('state_extras', {})
    costs = state_extras.get('cost', jnp.zeros_like(rewards))

    truncation = data.extras['state_extras']['truncation']
    termination = (1 - data.discount) * (1 - truncation)

    target_action_log_probs = parametric_action_distribution.log_prob(
        policy_logits, data.extras['policy_extras']['raw_action']
    )
    behaviour_action_log_probs = data.extras['policy_extras']['log_prob']

    # Compute regular advantages and returns
    vs, advantages = compute_gae(
        truncation=truncation, termination=termination, rewards=rewards,
        values=baseline, bootstrap_value=bootstrap_value,
        lambda_=gae_lambda, discount=discounting,
    )

    # Compute cost advantages and returns
    cost_vs, cost_advantages = compute_gae(
        truncation=truncation, termination=termination, rewards=costs,
        values=cost_baseline, bootstrap_value=bootstrap_cost_value,
        lambda_=gae_lambda, discount=discounting,
    )

    # Modify advantages with Lagrangian term
    modified_advantages = lagrangian_advantage(advantages, cost_advantages, lambda_lagr, normalize_advantage)
    rho_s = jnp.exp(target_action_log_probs - behaviour_action_log_probs)

    surrogate_loss1 = rho_s * modified_advantages
    surrogate_loss2 = jnp.clip(rho_s, 1 - clipping_epsilon, 1 + clipping_epsilon) * modified_advantages
    policy_loss = -jnp.mean(jnp.minimum(surrogate_loss1, surrogate_loss2))

    # Value function loss
    v_error = vs - baseline
    v_loss = jnp.mean(v_error * v_error) * 0.5 * 0.5

    # Cost value function loss
    cost_v_error = cost_vs - cost_baseline
    cost_v_loss = jnp.mean(cost_v_error * cost_v_error) * 0.5 * 0.5

    # Entropy reward
    entropy = jnp.mean(parametric_action_distribution.entropy(policy_logits, rng))
    entropy_loss = entropy_cost * -entropy

    mean_cost = jnp.mean(costs)

    total_loss = policy_loss + v_loss + cost_v_loss + entropy_loss
    return total_loss, {
        'total_loss': total_loss,
        'policy_loss': policy_loss,
        'v_loss': v_loss,
        'cost_v_loss': cost_v_loss,
        'entropy_loss': entropy_loss,
        'mean_cost': mean_cost,
    }
