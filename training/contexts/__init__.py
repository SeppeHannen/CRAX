"""Context spaces, context distributions, and per-episode context sampling.

Vocabulary (matches the thesis):

* **context** ω — a vector of floats identifying one task of a suite.
* **context space** Ω — the box of contexts a suite can instantiate
  (:class:`ContextSpace`); per-suite definitions live in :mod:`registry`.
* **context distribution** — an object that decides which contexts the student
  trains on (:class:`ContextDistribution`). Its parameters φ_k are a pytree that
  is frozen for one training round and updated on the host between rounds.
  ``UniformDistribution`` is the reference r; ``FixedContext`` and
  ``StagedContexts`` reproduce single-level training and CRAX's manual
  curriculum without recompiling.
* **realised curriculum** q̂ — the transition-weighted empirical distribution of
  contexts the student actually trained on. Every transition carries its context
  (``state.info["context"]``), so q̂ is a histogram over the rollout data.

The :class:`ContextualAutoResetWrapper` gives every parallel slot its own
context and a *fresh* episode (new context, new layout) whenever it terminates.

See ``docs/acl/design/`` for the constraints this design follows.
"""

from training.contexts.distribution import ContextDistribution, DistributionParameters
from training.contexts.distributions import FixedContext, PrioritizedLevelReplay, StagedContexts, UniformDistribution
from training.contexts.training_curriculum import training_curriculum_metrics
from training.contexts.registry import SuiteContexts, registered_environments, suite_contexts
from training.contexts.episodes import EpisodeTracker
from training.contexts.rollout import SLOT_INDEX_KEY, TRANSITION_CONTEXT_KEY, CompletedEpisodes, RoundRollout, Transitions
from training.contexts.round_hook import ContextRoundHook
from training.contexts.setup import (
    DEPLOYMENT_EVALUATION,
    NO_DISTRIBUTION,
    UNIFORM_EVALUATION,
    ContextTrainingSetup,
    context_training_setup,
    parse_distribution,
    spec_label,
)
from training.contexts.space import Context, Contexts, ContextSpace, Dimension, box
from training.contexts.wrapper import (
    CONTEXT_KEY,
    ContextualAutoResetWrapper,
    attach_parameters,
    current_contexts,
    current_parameters,
    make_evaluation_wrap_env_fn,
    make_wrap_env_fn,
    wrap_for_context_evaluation,
    wrap_for_context_training,
)

__all__ = [
    "CONTEXT_KEY",
    "DEPLOYMENT_EVALUATION",
    "NO_DISTRIBUTION",
    "SLOT_INDEX_KEY",
    "TRANSITION_CONTEXT_KEY",
    "UNIFORM_EVALUATION",
    "Context",
    "ContextDistribution",
    "ContextRoundHook",
    "ContextSpace",
    "ContextTrainingSetup",
    "Contexts",
    "ContextualAutoResetWrapper",
    "CompletedEpisodes",
    "Dimension",
    "EpisodeTracker",
    "FixedContext",
    "DistributionParameters",
    "PrioritizedLevelReplay",
    "RoundRollout",
    "StagedContexts",
    "SuiteContexts",
    "Transitions",
    "UniformDistribution",
    "attach_parameters",
    "box",
    "context_training_setup",
    "current_contexts",
    "current_parameters",
    "make_evaluation_wrap_env_fn",
    "make_wrap_env_fn",
    "parse_distribution",
    "training_curriculum_metrics",
    "registered_environments",
    "spec_label",
    "suite_contexts",
    "wrap_for_context_evaluation",
    "wrap_for_context_training",
]
