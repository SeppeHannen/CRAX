"""r = Uniform(Ω): the reference distribution and domain-randomisation baseline."""
from __future__ import annotations

import dataclasses
from typing import Dict

import jax
import jax.numpy as jnp
from flax import struct

from training.contexts.distribution import EpisodeFeedback
from training.contexts.space import Contexts, ContextSpace


@struct.dataclass
class UniformParams:
    """No learnable state; kept as a pytree so all distributions checkpoint alike."""

    round_index: jax.Array  # scalar int32, informational


@dataclasses.dataclass(frozen=True)
class UniformDistribution:
    space: ContextSpace

    def initialise(self, key: jax.Array) -> UniformParams:
        del key
        return UniformParams(round_index=jnp.asarray(0, jnp.int32))

    def sample(self, params: UniformParams, key: jax.Array, n: int) -> Contexts:
        del params
        return self.space.sample_uniform(key, n)

    def update(self, params: UniformParams, feedback: EpisodeFeedback) -> UniformParams:
        return params.replace(round_index=jnp.asarray(feedback.round_index + 1, jnp.int32))

    def log_probability(self, params: UniformParams, contexts: Contexts) -> jax.Array:
        del params
        return uniform_log_density(self.space, contexts)

    def summary(self, params: UniformParams) -> Dict[str, float]:
        return {}


def uniform_log_density(space: ContextSpace, contexts: Contexts) -> jax.Array:
    """log-density of Uniform(Ω) at each context: −log|Ω| inside, −inf outside.

    |Ω| is the volume of the continuous coordinates times the number of admissible
    integer tuples (``space.integer_points``, which applies the cap). Degenerate
    (fixed) continuous dimensions contribute no volume.
    """
    continuous = jnp.asarray([d.kind == "continuous" for d in space.dimensions])
    widths = jnp.where(continuous, space.high - space.low, 0.0)
    log_volume = jnp.sum(jnp.where(widths > 0, jnp.log(jnp.where(widths > 0, widths, 1.0)), 0.0))
    log_integer_points = jnp.log(float(space.integer_points.shape[0])) if space.integer_dimensions else 0.0
    inside = space.contains(contexts)
    return jnp.where(inside, -(log_volume + log_integer_points), -jnp.inf)
