"""PPO networks for a context run: each network reads the parts of the
observation it is allowed to see.

The observation is ``{"sensors": o, "context": ω}`` (:mod:`observation`), plus
``"memory"`` when the learner keeps a trace (:mod:`memory`) and
``"pixels/<camera>"`` parts when the run trains from pixels (``--vision``;
``GpuPixelObservationWrapper`` adds them outermost). The critics always read every
part: the cost critic has to estimate cost per context for a per-context
constraint to be about something, and a critic exists only in training, so what
a deployed robot would know is no concern. The policy reads every part but the
context, and the context too when it is *told* — the input it needs to hold a
different reward–cost trade-off per context
(``docs/acl/design/contextual_learner.md`` §2). A new labelled part is read by
every network without a change here.

Each network is Brax's MLP over the concatenation of its parts, each scaled its
own way: the sensors by the trainer's running observation normaliser, whose
statistics the trainer keeps per part; the context by Ω's box
(:func:`training.contexts.space.scale_to_bounds`), because its range is declared
before training and must not depend on what the curriculum samples; the pixels
through one shared Nature-CNN encoder (:class:`training.networks.VisionEncoder`)
whose latent the trainer computes once per step and stashes under
:data:`training.networks.VISION_LATENT_KEY` for every head to read — the
mechanism the stock vision networks use, so the trainer's encoder slot in the
parameters, the loss's ``with_shared_latent`` and the inference function all
apply unchanged. The bounds are a factory keyword of plain floats so a
checkpoint's config can rebuild the network. The trainer's normaliser still
keeps statistics for the ``context`` key (it updates every key of the
observation; ``training/agents/ppo/train.py`` is upstream code); nothing reads
them.
"""
from __future__ import annotations

from typing import Mapping, Optional, Sequence, Tuple

import jax
import jax.numpy as jnp
from flax import linen

from training import distribution, networks, types
from training.acme import running_statistics
from training.agents.ppo.networks import PPONetworks
from training.contexts.observation import CONTEXT_OBSERVATION_KEY, SENSORS_KEY
from training.contexts.space import Bounds, scale_to_bounds
from training.pixels import is_pixel_part

# The sizes PPO-Lagrange's default factory uses (training/agents/ppo_lag/train.py).
POLICY_HIDDEN_LAYER_SIZES: Tuple[int, ...] = (32,) * 4
VALUE_HIDDEN_LAYER_SIZES: Tuple[int, ...] = (256,) * 5


def make_contextual_ppo_networks(
    observation_size: Mapping[str, Tuple[int, ...]],
    action_size: int,
    preprocess_observations_fn: types.PreprocessObservationFn = types.identity_observation_preprocessor,
    *,
    policy_observes_context: bool,
    context_bounds: Mapping[str, Sequence[float]],
) -> PPONetworks:
    """Policy over every labelled part but the context (and the context too when
    ``policy_observes_context``); reward and cost critics over every labelled part.

    Both keywords are bound by :meth:`training.contexts.learner.Learner.train_kwargs`;
    the trainer supplies the positional arguments. ``context_bounds`` is Ω's bounds,
    ``{"low": [...], "high": [...]}`` — ``ContextSpace.bounds`` as a mapping, so it survives
    the checkpoint's JSON config.
    """
    bounds = Bounds(low=tuple(context_bounds["low"]), high=tuple(context_bounds["high"]))
    if len(bounds.low) != observation_size[CONTEXT_OBSERVATION_KEY][-1]:
        raise ValueError(
            f"context_bounds has {len(bounds.low)} dimensions, the context observation {observation_size[CONTEXT_OBSERVATION_KEY][-1]}"
        )
    encoder, latent_size = _vision_encoder(observation_size)
    critics_read = labelled_parts(observation_size)
    policy_reads = critics_read if policy_observes_context else tuple(part for part in critics_read if part != CONTEXT_OBSERVATION_KEY)
    action_distribution = distribution.NormalTanhDistribution(event_size=action_size)
    return PPONetworks(
        policy_network=_mlp_over_parts(
            observation_size, policy_reads, preprocess_observations_fn, bounds, latent_size,
            layer_sizes=(*POLICY_HIDDEN_LAYER_SIZES, action_distribution.param_size), squeeze_output=False,
        ),
        value_network=_mlp_over_parts(
            observation_size, critics_read, preprocess_observations_fn, bounds, latent_size,
            layer_sizes=(*VALUE_HIDDEN_LAYER_SIZES, 1), squeeze_output=True,
        ),
        cost_value_network=_mlp_over_parts(
            observation_size, critics_read, preprocess_observations_fn, bounds, latent_size,
            layer_sizes=(*VALUE_HIDDEN_LAYER_SIZES, 1), squeeze_output=True,
        ),
        parametric_action_distribution=action_distribution,
        encoder_network=encoder,
    )


def labelled_parts(observation_size: Mapping[str, Tuple[int, ...]]) -> Tuple[str, ...]:
    """The observation's vector parts — every part that is not a camera's pixels — in a fixed
    order: the sensors, the context, then any other part by name."""
    parts = {part for part in observation_size if not is_pixel_part(part)}
    for required in (SENSORS_KEY, CONTEXT_OBSERVATION_KEY):
        if required not in parts:
            raise KeyError(f"a context run's observation has a {required!r} part; this one has {sorted(observation_size)}")
    return (SENSORS_KEY, CONTEXT_OBSERVATION_KEY, *sorted(parts - {SENSORS_KEY, CONTEXT_OBSERVATION_KEY}))


def _vision_encoder(observation_size: Mapping[str, Tuple[int, ...]]) -> Tuple[Optional[networks.FeedForwardNetwork], int]:
    """The shared CNN over the observation's ``pixels/*`` parts and the width of its latent;
    ``(None, 0)`` for an observation without pixels."""
    pixel_parts = {part: shape for part, shape in observation_size.items() if is_pixel_part(part)}
    if not pixel_parts:
        return None, 0
    encoder = networks.make_vision_encoder_network(observation_size)
    probe = {part: jnp.zeros((1, *shape)) for part, shape in pixel_parts.items()}
    latent_size = int(encoder.apply(None, encoder.init(jax.random.PRNGKey(0)), probe).shape[-1])
    if latent_size == 0:
        raise ValueError(f"the vision encoder reduces the pixel parts {pixel_parts} to nothing; the image is too small for its strides")
    return encoder, latent_size


def _mlp_over_parts(
    observation_size: Mapping[str, Tuple[int, ...]],
    parts: Sequence[str],
    preprocess_observations_fn: types.PreprocessObservationFn,
    bounds: Bounds,
    latent_size: int,
    layer_sizes: Sequence[int],
    squeeze_output: bool,
) -> networks.FeedForwardNetwork:
    """An MLP whose input is the concatenation of the named observation parts — the sensors
    through the running normaliser, the context scaled to Ω's box — and, when
    ``latent_size > 0``, the shared vision encoder's latent the trainer put under
    :data:`networks.VISION_LATENT_KEY`."""
    module = networks.MLP(layer_sizes=list(layer_sizes), activation=linen.swish, kernel_init=jax.nn.initializers.lecun_uniform())
    input_size = sum(observation_size[part][-1] for part in parts) + latent_size

    def scaled(part: str, processor_params: running_statistics.RunningStatisticsState, values: jax.Array) -> jax.Array:
        if part == CONTEXT_OBSERVATION_KEY:
            return scale_to_bounds(values, bounds)
        return preprocess_observations_fn(values, networks.normalizer_select(processor_params, part))

    def apply(processor_params: running_statistics.RunningStatisticsState, params: types.Params, observation: Mapping[str, jax.Array]) -> jax.Array:
        pieces = [scaled(part, processor_params, observation[part]) for part in parts]
        if latent_size:
            pieces.append(observation[networks.VISION_LATENT_KEY])
        output = module.apply(params, jnp.concatenate(pieces, axis=-1))
        return jnp.squeeze(output, axis=-1) if squeeze_output else output

    dummy_input = jnp.zeros((1, input_size))
    return networks.FeedForwardNetwork(init=lambda key: module.init(key, dummy_input), apply=apply)
