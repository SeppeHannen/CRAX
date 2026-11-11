"""PPO networks for a context run: each network reads the parts of the
observation it is allowed to see.

The observation is ``{"sensors": o, "context": ω}`` (:mod:`observation`). The
critics always read every part: the cost critic has to estimate cost per context
for a per-context constraint to be about something, and a critic exists only in
training, so what a deployed robot would know is no concern. The policy reads
every other part, and of the context the columns it is handed: none when it is
blind; when it is *told*, the coordinates a deployed robot could know — the
input it needs to hold a different reward–cost trade-off per context
(``docs/acl/design/contextual_learner.md`` §2). A new labelled part is read by
every network without a change here.

Each network is Brax's MLP over the concatenation of its parts, each scaled its
own way: the sensors by the trainer's running observation normaliser, whose
statistics the trainer keeps per part; the context by Ω's box
(:func:`training.contexts.space.scale_to_bounds`), because its range is declared
before training and must not depend on what the curriculum samples. The bounds
are a factory keyword of plain floats so a checkpoint's config can rebuild the
network. The trainer's normaliser still
keeps statistics for the ``context`` key (it updates every key of the
observation; ``training/agents/ppo/train.py`` is upstream code); nothing reads
them.
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
from training.contexts.space import Bounds, scale_to_bounds

# The sizes PPO-Lagrange's default factory uses (training/agents/ppo_lag/train.py).
POLICY_HIDDEN_LAYER_SIZES: Tuple[int, ...] = (32,) * 4
VALUE_HIDDEN_LAYER_SIZES: Tuple[int, ...] = (256,) * 5


def make_contextual_ppo_networks(
    observation_size: Mapping[str, Tuple[int, ...]],
    action_size: int,
    preprocess_observations_fn: types.PreprocessObservationFn = types.identity_observation_preprocessor,
    *,
    policy_context_columns: Sequence[int],
    context_bounds: Mapping[str, Sequence[float]],
) -> PPONetworks:
    """Policy over every labelled part, reading ``policy_context_columns`` of the context
    (none for a blind policy); reward and cost critics over every labelled part, whole.

    Both keywords are bound by :meth:`training.contexts.learner.Learner.train_kwargs`;
    the trainer supplies the positional arguments. ``context_bounds`` is the bounds of the
    run's contexts, ``{"low": [...], "high": [...]}`` — ``ContextSpace.bounds`` as a mapping,
    so it survives the checkpoint's JSON config, as the columns do.
    """
    bounds = Bounds(low=tuple(context_bounds["low"]), high=tuple(context_bounds["high"]))
    context_size = observation_size[CONTEXT_OBSERVATION_KEY][-1]
    if len(bounds.low) != context_size:
        raise ValueError(f"context_bounds has {len(bounds.low)} dimensions, the context observation {context_size}")
    policy_columns = tuple(int(column) for column in policy_context_columns)
    if len(set(policy_columns)) != len(policy_columns) or any(not 0 <= column < context_size for column in policy_columns):
        raise ValueError(f"policy_context_columns {policy_columns} are not distinct columns of a context of {context_size} dimensions")
    parts = labelled_parts(observation_size)
    action_distribution = distribution.NormalTanhDistribution(event_size=action_size)
    return PPONetworks(
        policy_network=_mlp_over_parts(
            observation_size, parts, preprocess_observations_fn, bounds, policy_columns,
            layer_sizes=(*POLICY_HIDDEN_LAYER_SIZES, action_distribution.param_size), squeeze_output=False,
        ),
        value_network=_mlp_over_parts(
            observation_size, parts, preprocess_observations_fn, bounds, tuple(range(context_size)),
            layer_sizes=(*VALUE_HIDDEN_LAYER_SIZES, 1), squeeze_output=True,
        ),
        cost_value_network=_mlp_over_parts(
            observation_size, parts, preprocess_observations_fn, bounds, tuple(range(context_size)),
            layer_sizes=(*VALUE_HIDDEN_LAYER_SIZES, 1), squeeze_output=True,
        ),
        parametric_action_distribution=action_distribution,
    )


def labelled_parts(observation_size: Mapping[str, Tuple[int, ...]]) -> Tuple[str, ...]:
    """The observation's parts in a fixed order: the sensors, the context, then any other part by name."""
    parts = set(observation_size)
    for required in (SENSORS_KEY, CONTEXT_OBSERVATION_KEY):
        if required not in parts:
            raise KeyError(f"a context run's observation has a {required!r} part; this one has {sorted(observation_size)}")
    return (SENSORS_KEY, CONTEXT_OBSERVATION_KEY, *sorted(parts - {SENSORS_KEY, CONTEXT_OBSERVATION_KEY}))


def _mlp_over_parts(
    observation_size: Mapping[str, Tuple[int, ...]],
    parts: Sequence[str],
    preprocess_observations_fn: types.PreprocessObservationFn,
    bounds: Bounds,
    context_columns: Tuple[int, ...],
    layer_sizes: Sequence[int],
    squeeze_output: bool,
) -> networks.FeedForwardNetwork:
    """An MLP whose input is the concatenation of the named observation parts: the sensors
    through the running normaliser, and ``context_columns`` of the context scaled to ``bounds``."""
    module = networks.MLP(layer_sizes=list(layer_sizes), activation=linen.swish, kernel_init=jax.nn.initializers.lecun_uniform())
    input_size = sum(len(context_columns) if part == CONTEXT_OBSERVATION_KEY else observation_size[part][-1] for part in parts)
    columns = jnp.asarray(context_columns, dtype=jnp.int32)

    def scaled(part: str, processor_params: running_statistics.RunningStatisticsState, values: jax.Array) -> jax.Array:
        if part == CONTEXT_OBSERVATION_KEY:
            return scale_to_bounds(values, bounds)[..., columns]
        return preprocess_observations_fn(values, networks.normalizer_select(processor_params, part))

    def apply(processor_params: running_statistics.RunningStatisticsState, params: types.Params, observation: Mapping[str, jax.Array]) -> jax.Array:
        pieces = [scaled(part, processor_params, observation[part]) for part in parts]
        output = module.apply(params, jnp.concatenate(pieces, axis=-1))
        return jnp.squeeze(output, axis=-1) if squeeze_output else output

    dummy_input = jnp.zeros((1, input_size))
    return networks.FeedForwardNetwork(init=lambda key: module.init(key, dummy_input), apply=apply)
