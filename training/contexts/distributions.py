"""Concrete context distributions.

* :class:`UniformDistribution` — the reference distribution r = Uniform(Ω); the
  domain-randomisation baseline. Stateless: ``update`` is the identity.
* :class:`FixedContext` — always the same ω. Recovers training on one
  difficulty level. Also the building block for evaluation on the target w.
* :class:`StagedContexts` — a fixed sequence of contexts switched at given
  training rounds. Recovers CRAX's manual curriculum (level 1 → 2 → 3 with an
  equal budget split) **without recompiling** between stages.

All three keep their parameters as a small pytree so they checkpoint and log
uniformly with adaptive methods added later.
"""
from __future__ import annotations

import dataclasses
from typing import Dict, Sequence

import jax
import jax.numpy as jnp
from flax import struct

from training.contexts.distribution import EpisodeFeedback, batched_log_probability_uniform
from training.contexts.space import Context, Contexts, ContextSpace


# --------------------------------------------------------------------------- #
# Uniform
# --------------------------------------------------------------------------- #


@struct.dataclass
class UniformParams:
    """No learnable state; kept as a pytree for uniformity."""

    round_index: jax.Array  # scalar int32, purely informational


@dataclasses.dataclass(frozen=True)
class UniformDistribution:
    """r = Uniform(Ω)."""

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
        return batched_log_probability_uniform(self.space, contexts)

    def summary(self, params: UniformParams) -> Dict[str, float]:
        return {}


# --------------------------------------------------------------------------- #
# Fixed
# --------------------------------------------------------------------------- #


@struct.dataclass
class FixedParams:
    context: jax.Array  # [D]


@dataclasses.dataclass(frozen=True)
class FixedContext:
    """Always the same context. ``FixedContext.at_level(space, registry, 3)`` etc."""

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


# --------------------------------------------------------------------------- #
# Staged (manual curriculum)
# --------------------------------------------------------------------------- #


@struct.dataclass
class StagedParams:
    stage: jax.Array  # scalar int32: index into the stage table
    round_index: jax.Array  # scalar int32


@dataclasses.dataclass(frozen=True)
class StagedContexts:
    """A fixed sequence of contexts, advanced at fixed training rounds.

    ``switch_rounds[i]`` is the first round at which stage ``i + 1`` is used;
    before ``switch_rounds[0]`` stage 0 is used. Equal budget split over
    ``S`` stages and ``K`` rounds is ``switch_rounds = [K/S, 2K/S, ...]``.

    Because the stage index is a *value* in ``params`` and the stage table is a
    constant array, switching stages does not change the compiled program.
    """

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
