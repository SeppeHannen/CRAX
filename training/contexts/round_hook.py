"""The host-side half of context-distribution training.

Inside the compiled training step the wrapper samples contexts from frozen
parameters φ_k, which travel in the environment state. Between calls,
:class:`ContextRoundHook` closes the loop: it takes the round's transitions (the
recorded state fields and the learner's signals on them), reads φ_k out of the
environment state, asks the distribution for φ_{k+1}, and writes φ_{k+1} back
into the state that the next call will run with. It returns the round's
``training_curriculum/*`` metrics (:mod:`training.contexts.training_curriculum`)
for the trainer to log.

φ lives in the environment state only; the hook keeps no copy. Its one piece of
host state is the :class:`EpisodeTracker`, the per-slot sums of episodes that
have not ended yet, which must survive from one round to the next.
"""
from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Tuple

import numpy as np
from crax.envs.base import State

from training.rounds import LearningSignals, RoundHook
from training.contexts.distribution import ContextDistribution
from training.contexts.episodes import EpisodeTracker
from training.contexts.rollout import ROLLOUT_FIELDS, RoundRollout, Transitions
from training.contexts.training_curriculum import training_curriculum_metrics
from training.contexts.wrapper import attach_parameters, current_parameters


class ContextRoundHook(RoundHook):
    """The round hook for a :class:`ContextDistribution`.

    ``num_slots`` is the trainer's ``num_envs``: the number of parallel
    environments whose partial episodes the tracker carries across rounds.
    """

    def __init__(self, distribution: ContextDistribution, num_slots: int):
        self.distribution = distribution
        self.episodes = EpisodeTracker(num_slots)
        self._transitions: Optional[Transitions] = None  # this round's, between observe and on_round_end

    @property
    def extra_fields(self) -> Tuple[str, ...]:
        return ROLLOUT_FIELDS

    def observe(self, rollout: Mapping[str, np.ndarray], learning_signals: LearningSignals) -> None:
        """The trainer's recorded fields become typed, per-slot :class:`Transitions` here,
        at the boundary; nothing downstream sees the recorded layout."""
        self._transitions = Transitions.from_recorded_fields(rollout, learning_signals, self.episodes.num_slots)

    def on_round_end(self, round_index: int, env_state: State) -> Tuple[State, Dict[str, Any]]:
        if self._transitions is None:
            raise RuntimeError("on_round_end called before observe: is the hook's extra_fields recorded?")
        transitions, self._transitions = self._transitions, None
        rollout = RoundRollout(round_index, transitions, self.episodes.complete(transitions))

        parameters = current_parameters(env_state)  # φ_k: the distribution this round was sampled from
        env_state = attach_parameters(env_state, self.distribution.update(parameters, rollout))
        return env_state, training_curriculum_metrics(self.distribution, parameters, rollout)
