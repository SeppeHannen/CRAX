"""Concrete context distributions, one module each.

Non-adaptive:
* :class:`UniformDistribution` — r = Uniform(Ω), the domain-randomisation baseline.
* :class:`FixedContext` — a single context (one difficulty level; the target w).
* :class:`StagedContexts` — a fixed sequence switched at fixed rounds (the manual curriculum).

Adaptive methods (PLR-style replay, learnability, …) go here as further modules.
"""

from training.contexts.distributions.fixed import FixedContext, FixedParams
from training.contexts.distributions.staged import StagedContexts, StagedParams
from training.contexts.distributions.uniform import UniformDistribution, UniformParams, uniform_log_density

__all__ = [
    "FixedContext",
    "FixedParams",
    "StagedContexts",
    "StagedParams",
    "UniformDistribution",
    "UniformParams",
    "uniform_log_density",
]
