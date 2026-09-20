"""A fixed sequence of contexts switched at fixed training rounds: CRAX's manual
curriculum (level 1 → 2 → 3, equal budget split) without recompiling.

The stage index is a *value* in the parameters and the stage table is a constant
array, so advancing a stage changes data, not the compiled program.
"""
from __future__ import annotations

import dataclasses
from typing import Dict, Sequence

import jax
import jax.numpy as jnp
from flax import struct

from training.contexts.distribution import EpisodeFeedback
from training.contexts.space import Contexts, ContextSpace


@struct.dataclass
class StagedParams:
    stage: jax.Array  # scalar int32, index into the stage table
    round_index: jax.Array  # scalar int32


@dataclasses.dataclass(frozen=True)
class StagedContexts:
    """``switch_rounds[i]`` is the first round at which stage ``i + 1`` is used."""

    space: ContextSpace
    stages: Contexts  # [S, D]
    switch_rounds: Sequence[int]

    def __post_init__(self) -> None:
        if self.stages.ndim != 2 or self.stages.shape[1] != self.space.size:
            raise ValueError(f"stages has shape {self.stages.shape}, expected (S, {self.space.size})")
        if len(self.switch_rounds) != self.stages.shape[0] - 1:
            raise ValueError("need exactly S-1 switch rounds for S stages")
        if list(self.switch_rounds) != sorted(self.switch_rounds):
            raise ValueError("switch_rounds must be increasing")

    @classmethod
    def equal_split(cls, space: ContextSpace, stages: Contexts, total_rounds: int) -> "StagedContexts":
        num_stages = int(stages.shape[0])
        switch_rounds = [round(total_rounds * (i + 1) / num_stages) for i in range(num_stages - 1)]
        return cls(space=space, stages=stages, switch_rounds=switch_rounds)

    def initialise(self, key: jax.Array) -> StagedParams:
        del key
        return StagedParams(stage=jnp.asarray(0, jnp.int32), round_index=jnp.asarray(0, jnp.int32))

    def sample(self, params: StagedParams, key: jax.Array, n: int) -> Contexts:
        del key
        current = jnp.asarray(self.stages, jnp.float32)[params.stage]
        return jnp.broadcast_to(current, (n, self.space.size))

    def update(self, params: StagedParams, feedback: EpisodeFeedback) -> StagedParams:
        next_round = feedback.round_index + 1
        stage = sum(1 for r in self.switch_rounds if next_round >= r)
        return StagedParams(stage=jnp.asarray(stage, jnp.int32), round_index=jnp.asarray(next_round, jnp.int32))

    def log_probability(self, params: StagedParams, contexts: Contexts) -> jax.Array:
        current = jnp.asarray(self.stages, jnp.float32)[params.stage]
        match = jnp.all(jnp.isclose(contexts, current, atol=1e-6), axis=-1)
        return jnp.where(match, 0.0, -jnp.inf)

    def summary(self, params: StagedParams) -> Dict[str, float]:
        current = self.space.decode(jnp.asarray(self.stages)[int(params.stage)])
        return {"stage": float(params.stage), **{f"context/{k}": float(v) for k, v in current.items()}}
