"""PPO networks for a context run: each network reads the parts of the
observation it is allowed to see.

The observation is ``{"sensors": o, "context": ω}`` (:mod:`observation`). The
critics always read both: the cost critic has to estimate cost per context for
a per-context constraint to be about something, and a critic exists only in
training, so what a deployed robot would know is no concern. The policy reads
the sensors, and the context too when it is *told* — the input it needs to
hold a different reward–cost trade-off per context
(``docs/acl/design/contextual_learner.md`` §2).

Each network is Brax's MLP over the concatenation of its normalised parts; the
observation normaliser already keeps statistics per part.
"""
from __future__ import annotations

from typing import Mapping, Sequence, Tuple

import jax
import jax.numpy as jnp
from flax import linen

from training import distribution, networks, types
from training.acme import running_statistics
from training.agents.ppo.networks import PPONetworks
from training.contexts.observation import CONTEXT_OBSERVATION_KEY, SENSORS_KEY

SENSORS_ONLY: Tuple[str, ...] = (SENSORS_KEY,)
SENSORS_AND_CONTEXT: Tuple[str, ...] = (SENSORS_KEY, CONTEXT_OBSERVATION_KEY)

# The sizes PPO-Lagrange's default factory uses (training/agents/ppo_lag/train.py).
POLICY_HIDDEN_LAYER_SIZES: Tuple[int, ...] = (32,) * 4
VALUE_HIDDEN_LAYER_SIZES: Tuple[int, ...] = (256,) * 5


def make_contextual_ppo_networks(
    observation_size: Mapping[str, Tuple[int, ...]],
    action_size: int,
    preprocess_observations_fn: types.PreprocessObservationFn = types.identity_observation_preprocessor,
    policy_observes_context: bool = False,
) -> PPONetworks:
    """Policy over the sensors (and the context when ``policy_observes_context``);
    reward and cost critics over sensors and context."""
    policy_reads = SENSORS_AND_CONTEXT if policy_observes_context else SENSORS_ONLY
    action_distribution = distribution.NormalTanhDistribution(event_size=action_size)
    return PPONetworks(
        policy_network=_mlp_over_parts(
            observation_size, policy_reads, preprocess_observations_fn,
            layer_sizes=(*POLICY_HIDDEN_LAYER_SIZES, action_distribution.param_size), squeeze_output=False,
        ),
        value_network=_mlp_over_parts(
            observation_size, SENSORS_AND_CONTEXT, preprocess_observations_fn,
            layer_sizes=(*VALUE_HIDDEN_LAYER_SIZES, 1), squeeze_output=True,
        ),
        cost_value_network=_mlp_over_parts(
            observation_size, SENSORS_AND_CONTEXT, preprocess_observations_fn,
            layer_sizes=(*VALUE_HIDDEN_LAYER_SIZES, 1), squeeze_output=True,
        ),
        parametric_action_distribution=action_distribution,
    )


def _mlp_over_parts(
    observation_size: Mapping[str, Tuple[int, ...]],
    parts: Sequence[str],
    preprocess_observations_fn: types.PreprocessObservationFn,
    layer_sizes: Sequence[int],
    squeeze_output: bool,
) -> networks.FeedForwardNetwork:
    """An MLP whose input is the concatenation of the named observation parts, each
    normalised with its own statistics."""
    module = networks.MLP(layer_sizes=list(layer_sizes), activation=linen.swish, kernel_init=jax.nn.initializers.lecun_uniform())
    input_size = sum(observation_size[part][-1] for part in parts)

    def apply(processor_params: running_statistics.RunningStatisticsState, params: types.Params, observation: Mapping[str, jax.Array]) -> jax.Array:
        inputs = jnp.concatenate(
            [preprocess_observations_fn(observation[part], networks.normalizer_select(processor_params, part)) for part in parts],
            axis=-1,
        )
        output = module.apply(params, inputs)
        return jnp.squeeze(output, axis=-1) if squeeze_output else output

    dummy_input = jnp.zeros((1, input_size))
    return networks.FeedForwardNetwork(init=lambda key: module.init(key, dummy_input), apply=apply)
