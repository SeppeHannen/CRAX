"""A point mass on one context: training on a single difficulty level, or the
deployment target w when w is a single task."""
from __future__ import annotations

import dataclasses
from typing import Dict

import jax
import jax.numpy as jnp
from flax import struct

from training.contexts.distribution import EpisodeFeedback
from training.contexts.space import Context, Contexts, ContextSpace


@struct.dataclass
class FixedParams:
    context: jax.Array  # [D]


@dataclasses.dataclass(frozen=True)
class FixedContext:
    space: ContextSpace
    context: Context

    def __post_init__(self) -> None:
        if self.context.shape != (self.space.size,):
            raise ValueError(f"context has shape {self.context.shape}, expected ({self.space.size},)")

    def initialise(self, key: jax.Array) -> FixedParams:
        del key
        return FixedParams(context=jnp.asarray(self.context, jnp.float32))

    def sample(self, params: FixedParams, key: jax.Array, n: int) -> Contexts:
        del key
        return jnp.broadcast_to(params.context, (n, self.space.size))

    def update(self, params: FixedParams, feedback: EpisodeFeedback) -> FixedParams:
        return params

    def log_probability(self, params: FixedParams, contexts: Contexts) -> jax.Array:
        match = jnp.all(jnp.isclose(contexts, params.context, atol=1e-6), axis=-1)
        return jnp.where(match, 0.0, -jnp.inf)

    def summary(self, params: FixedParams) -> Dict[str, float]:
        return {f"context/{name}": float(v) for name, v in self.space.decode(params.context).items()}
