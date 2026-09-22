"""How an environment reads its per-episode task parameters — its *context*.

A context is a vector of floats, one per parallel environment, stored in
``state.info[CONTEXT_KEY]`` from ``reset`` to the end of the episode. The
training side (``training/contexts``) decides *which* context each episode
gets; an environment only *reads* it. This module fixes how.

The invariant every contextual suite keeps, and ``tests/test_suites.py`` checks:

    Every difficulty knob is read from the context, always — in a plain
    ``reset(rng)`` as much as in a ``reset_with_context(rng, ω)``. The
    constructor's value of a knob is used for exactly one thing: building
    ``default_context()``.

So a stock CRAX run and a context run execute the same ``step`` code; the only
difference is who wrote the context. There is no "if a context is present"
branch anywhere.

A suite adopts the pattern with four members::

    class SafeSomething(PipelineEnv):
        CONTEXT_PARAMETERS = ("max_height",)          # names, in order

        def default_context(self) -> jax.Array:       # from constructor args
            return jp.asarray([self._max_height], jp.float32)

        def reset_with_context(self, rng, context):   # the one reset
            ...
            return State(..., info={CONTEXT_KEY: context, ...})

        def reset(self, rng):
            return self.reset_with_context(rng, self.default_context())

and reads a knob in ``step`` with ``parameter(self, state, "max_height")``.
"""
from __future__ import annotations

from typing import Protocol, Tuple

import jax
from jax import numpy as jp

from crax.envs.base import State

CONTEXT_KEY = "context"


class ContextualEnv(Protocol):
    """What the training side may assume of a suite that reads a context."""

    CONTEXT_PARAMETERS: Tuple[str, ...]

    def default_context(self) -> jax.Array:
        """The context a plain ``reset`` uses: the constructor's knob values, in ``CONTEXT_PARAMETERS`` order."""

    def reset_with_context(self, rng: jax.Array, context: jax.Array) -> State:
        """Reset one environment into context ``context`` (shape ``[len(CONTEXT_PARAMETERS)]``)."""


def parameter(env: ContextualEnv, state: State, name: str) -> jax.Array:
    """The current episode's value of knob ``name``, read from the state's context.

    Works for a single environment (context ``[D]``) and under ``vmap``
    (``[..., D]``) alike; the result has the batch shape of the state.
    """
    return state.info[CONTEXT_KEY][..., env.CONTEXT_PARAMETERS.index(name)]


def encode(env: ContextualEnv, **values: float) -> jax.Array:
    """A context vector from named knob values; every ``CONTEXT_PARAMETERS`` entry must be given."""
    if set(values) != set(env.CONTEXT_PARAMETERS):
        raise ValueError(
            f"{type(env).__name__}.default_context: expected exactly {env.CONTEXT_PARAMETERS}, got {tuple(values)}"
        )
    return jp.asarray([float(values[name]) for name in env.CONTEXT_PARAMETERS], jp.float32)
