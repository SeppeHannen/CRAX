"""A point mass on one context: training on a single difficulty level, or the
deployment target w when w is a single task."""
from __future__ import annotations

import dataclasses
from typing import Dict

import jax
import jax.numpy as jnp
from flax import struct

from training.contexts.rollout import RoundRollout
from training.contexts.space import Context, Contexts, ContextSpace


@struct.dataclass
class FixedParameters:
    context: jax.Array  # [D]


@dataclasses.dataclass(frozen=True)
class FixedContext:
    space: ContextSpace
    context: Context

    def __post_init__(self) -> None:
        if self.context.shape != (self.space.size,):
            raise ValueError(f"context has shape {self.context.shape}, expected ({self.space.size},)")

    def initialise(self) -> FixedParameters:
        return FixedParameters(context=jnp.asarray(self.context, jnp.float32))

    def sample(self, parameters: FixedParameters, key: jax.Array, n: int) -> Contexts:
        del key
        return jnp.broadcast_to(parameters.context, (n, self.space.size))

    def update(self, parameters: FixedParameters, rollout: RoundRollout) -> FixedParameters:
        return parameters

    def summary(self, parameters: FixedParameters) -> Dict[str, float]:
        return {}
