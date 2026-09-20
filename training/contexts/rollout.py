"""The transitions of one training round, as the distribution sees them.

The trainer records, for every transition, the context it was generated in
and EpisodeWrapper's bookkeeping (``episode_done`` marks the transitions that
ended an episode; ``episode_metrics`` then holds that episode's totals). On the
host this becomes a :class:`RoundRollout`, from which two things are derived:

* the realised curriculum q̂ — a histogram over *all* transitions' contexts
  (:mod:`realised_curriculum`);
* the :class:`EpisodeFeedback` a distribution learns from — one row per
  *completed* episode (:func:`completed_episodes`).
"""
from __future__ import annotations

import dataclasses
from typing import Mapping

import numpy as np

from training.contexts.distribution import EpisodeFeedback
from training.contexts.wrapper import TRANSITION_CONTEXT_KEY

# The state.info keys a round hook needs the trainer to record per transition.
ROLLOUT_FIELDS = (TRANSITION_CONTEXT_KEY, "episode_metrics", "episode_done")


@dataclasses.dataclass(frozen=True)
class RoundRollout:
    """All transitions of one round, flattened to a leading axis of length T."""

    round_index: int
    contexts: np.ndarray  # [T, D]  context each transition was generated in
    episode_done: np.ndarray  # [T]  True where an episode ended
    episode_return: np.ndarray  # [T]  total reward of the episode ending there (else partial sum)
    episode_cost: np.ndarray  # [T]
    episode_length: np.ndarray  # [T]

    @classmethod
    def from_recorded_fields(cls, recorded: Mapping[str, np.ndarray], round_index: int) -> "RoundRollout":
        """Flatten the arrays the trainer recorded (any leading batch axes)."""
        contexts = np.asarray(recorded[TRANSITION_CONTEXT_KEY])
        metrics = recorded["episode_metrics"]
        flat = lambda x: np.asarray(x).reshape(-1)
        return cls(
            round_index=round_index,
            contexts=contexts.reshape(-1, contexts.shape[-1]),
            episode_done=flat(recorded["episode_done"]).astype(bool),
            episode_return=flat(metrics["sum_reward"]),
            episode_cost=flat(metrics["cost"]),
            episode_length=flat(metrics["length"]),
        )

    @property
    def num_transitions(self) -> int:
        return int(self.contexts.shape[0])

    @property
    def num_completed_episodes(self) -> int:
        return int(self.episode_done.sum())


def completed_episodes(rollout: RoundRollout) -> EpisodeFeedback:
    """One row per episode that ended during the round."""
    done = rollout.episode_done
    return EpisodeFeedback(
        contexts=rollout.contexts[done],
        returns=rollout.episode_return[done],
        costs=rollout.episode_cost[done],
        lengths=rollout.episode_length[done],
        round_index=rollout.round_index,
    )
