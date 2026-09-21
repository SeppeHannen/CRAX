"""The host-side half of context-distribution training.

Inside the compiled training step the wrapper samples contexts from frozen
parameters φ_k. Between calls, :class:`ContextRoundHook` closes the loop: it
takes the round's transitions, extracts the completed episodes, asks the
distribution for φ_{k+1}, and writes φ_{k+1} into the environment state that
the next call will run with. It returns the round's ``training_curriculum/*``
metrics (:mod:`training.contexts.training_curriculum`) for the trainer to log.
"""
from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Tuple

import numpy as np
from crax.envs.base import State

from training.rounds import RoundHook
from training.contexts.distribution import ContextDistribution, Params
from training.contexts.rollout import ROLLOUT_FIELDS, RoundRollout, completed_episodes
from training.contexts.training_curriculum import training_curriculum_metrics
from training.contexts.wrapper import attach_parameters, current_parameters


class ContextRoundHook(RoundHook):
    """The round hook for a :class:`ContextDistribution`."""

    def __init__(self, distribution: ContextDistribution):
        self.distribution = distribution
        self.parameters: Optional[Params] = None  # φ for the next round; None until the first round ends
        self._recorded_rollout: Optional[Mapping[str, np.ndarray]] = None

    @property
    def extra_fields(self) -> Tuple[str, ...]:
        return ROLLOUT_FIELDS

    def observe(self, rollout: Mapping[str, np.ndarray]) -> None:
        self._recorded_rollout = rollout

    def on_round_end(self, round_index: int, env_state: State) -> Tuple[State, Dict[str, Any]]:
        if self._recorded_rollout is None:
            raise RuntimeError("on_round_end called before observe: is the hook's extra_fields recorded?")
        rollout = RoundRollout.from_recorded_fields(self._recorded_rollout, round_index)
        self._recorded_rollout = None
        feedback = completed_episodes(rollout)

        parameters = self.parameters if self.parameters is not None else current_parameters(env_state)
        self.parameters = self.distribution.update(parameters, feedback)
        env_state = attach_parameters(env_state, self.parameters)

        # `parameters` is φ_k, the distribution the round was actually sampled from.
        return env_state, training_curriculum_metrics(self.distribution, parameters, rollout)
