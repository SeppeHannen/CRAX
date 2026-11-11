"""One given context per slot: the distribution of a scoring run.

A scoring run (``training/score.py``) chooses its contexts on the host — so many
episodes in every cell of an evaluation protocol — and rolls them all out at
once. Slot ``i`` runs in ``contexts[i]``, in every episode it starts, so what
the evaluator reports for slot ``i`` is the outcome in that context.
"""
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
class AssignedParameters:
    contexts: jax.Array  # [N, D]


@dataclasses.dataclass(frozen=True)
class AssignedContexts:
    space: ContextSpace
    contexts: Contexts  # [N, D], one per slot

    def __post_init__(self) -> None:
        if self.contexts.ndim != 2 or self.contexts.shape[1] != self.space.size:
            raise ValueError(f"contexts has shape {self.contexts.shape}, expected (N, {self.space.size})")
        if not bool(jnp.all(self.space.contains(self.contexts))):
            raise ValueError(f"an assigned context lies outside the space\n{self.space.describe()}")

    @property
    def required_statistics(self) -> FrozenSet[EpisodeStatistic]:
        return frozenset()

    def initialise(self) -> AssignedParameters:
        return AssignedParameters(contexts=jnp.asarray(self.contexts, jnp.float32))

    def sample(self, parameters: AssignedParameters, key: jax.Array, n: int) -> Contexts:
        del key
        if n != parameters.contexts.shape[0]:
            raise ValueError(f"{parameters.contexts.shape[0]} contexts were assigned, one per slot; the population has {n} slots")
        return parameters.contexts

    def update(self, parameters: AssignedParameters, rollout: RoundRollout) -> AssignedParameters:
        return parameters

    def summary(self, parameters: AssignedParameters) -> Dict[str, float]:
        return {}
