"""Concrete context distributions, one module each.

Non-adaptive:
* :class:`UniformDistribution` — r = Uniform(Ω), the domain-randomisation baseline.
* :class:`FixedContext` — a single context (one difficulty level; the target w).
* :class:`StagedContexts` — a fixed sequence switched at fixed rounds (the manual curriculum).

Adaptive:
* :class:`PrioritizedLevelReplay` — PLR (Jiang et al. 2021) in its unbounded-level form:
  a buffer of contexts scored by L1 value loss, replayed by rank and staleness
  (``docs/acl/design/prioritized_level_replay.md``).
"""

from training.contexts.distributions.fixed import FixedContext, FixedParameters
from training.contexts.distributions.prioritized_level_replay import PrioritizedLevelReplay, PrioritizedLevelReplayParameters
from training.contexts.distributions.staged import StagedContexts, StagedParameters
from training.contexts.distributions.uniform import UniformDistribution, UniformParameters

__all__ = [
    "FixedContext",
    "FixedParameters",
    "PrioritizedLevelReplay",
    "PrioritizedLevelReplayParameters",
    "StagedContexts",
    "StagedParameters",
    "UniformDistribution",
    "UniformParameters",
]
