"""The host-side half of context-distribution training.

Inside the compiled training step the wrapper samples contexts from frozen
parameters φ_k. Between calls, :class:`ContextRoundHook` closes the loop: it
takes the round's transitions, extracts the completed episodes, asks the
distribution for φ_{k+1}, and writes φ_{k+1} into the environment state that
the next call will run with. It also reports the intended curriculum q (from
``distribution.summary(φ_k)``) and the realised one q̂ (from the transitions),
so the two are always logged side by side.
"""
from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Tuple

import numpy as np
from crax.envs.base import State

from training.rounds import RoundHook
from training.contexts.distribution import ContextDistribution, EpisodeFeedback, Params
from training.contexts.realised_curriculum import curriculum_metrics
from training.contexts.rollout import ROLLOUT_FIELDS, RoundRollout, completed_episodes
from training.contexts.wrapper import attach_parameters, current_parameters


class ContextRoundHook(RoundHook):
    """The round hook for a :class:`ContextDistribution`."""

    def __init__(self, distribution: ContextDistribution):
        self.distribution = distribution
        self.parameters: Optional[Params] = None  # φ for the next round; None until the first round ends
        self.last_feedback: Optional[EpisodeFeedback] = None
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
        self.last_feedback = feedback
        env_state = attach_parameters(env_state, self.parameters)

        metrics: Dict[str, Any] = {"curriculum/round": float(round_index)}
        for key, value in self.distribution.summary(parameters).items():
            metrics[f"curriculum/intended/{key}"] = value  # q as the distribution states it
        for key, value in curriculum_metrics(self.distribution.space, rollout).items():
            metrics[f"curriculum/{key}"] = value  # q as sampled, q̂ as experienced
        if feedback.num_completed:
            metrics["curriculum/completed/mean_return"] = float(feedback.returns.mean())
            metrics["curriculum/completed/mean_cost"] = float(feedback.costs.mean())
            metrics["curriculum/completed/mean_length"] = float(feedback.lengths.mean())
        return env_state, metrics
