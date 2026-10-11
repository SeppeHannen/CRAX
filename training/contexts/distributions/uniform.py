"""r = Uniform(Ω): the reference distribution and domain-randomisation baseline."""
from __future__ import annotations

import dataclasses
from typing import Dict, FrozenSet

import jax
import jax.numpy as jnp
from flax import struct

from training.contexts.rollout import RoundRollout
from training.contexts.space import Contexts, ContextSpace
from training.contexts.statistics import EpisodeStatistic


@struct.dataclass
class UniformParameters:
    """No learnable state; kept as a pytree so all distributions checkpoint alike."""

    round_index: jax.Array  # scalar int32, informational


@dataclasses.dataclass(frozen=True)
class UniformDistribution:
    space: ContextSpace

    @property
    def required_statistics(self) -> FrozenSet[EpisodeStatistic]:
        return frozenset()

    def initialise(self) -> UniformParameters:
        return UniformParameters(round_index=jnp.asarray(0, jnp.int32))

    def sample(self, parameters: UniformParameters, key: jax.Array, n: int) -> Contexts:
        del parameters
        return self.space.sample_uniform(key, n)

    def update(self, parameters: UniformParameters, rollout: RoundRollout) -> UniformParameters:
        return parameters.replace(round_index=jnp.asarray(rollout.round_index + 1, jnp.int32))

    def summary(self, parameters: UniformParameters) -> Dict[str, float]:
        return {}
