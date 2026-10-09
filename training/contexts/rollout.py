"""What one training round produced, as a distribution sees it.

The trainer records, for every transition, the context it was generated in and
EpisodeWrapper's bookkeeping (``episode_done`` marks the transitions that ended
an episode; ``episode_metrics`` then holds that episode's totals), and hands
over the learner's view of the same transitions (``LearningSignals``: the reward
advantage under the pre-update value function). On the host this becomes a
:class:`RoundRollout` with two views of the same data:

* :class:`Transitions` — every transition of the round, laid out as
  ``[num_slots, steps_per_slot]`` in the order each slot experienced them. The
  realised curriculum q̂ and the per-context value error are read from here.
* :class:`CompletedEpisodes` — one row per episode that *ended* in the round,
  with its context, totals and the mean |reward advantage| over the **whole**
  episode, including the part that ran in earlier rounds
  (:class:`training.contexts.episodes.EpisodeTracker` carries it over). The
  empirical q and PLR's score are read from here.
"""
from __future__ import annotations

import dataclasses
from typing import Mapping

import numpy as np

from training.rounds import LearningSignals

# The state.info keys the context wrapper writes per slot (training/contexts/wrapper.py):
# the context the *last transition* was generated in (it differs from the running
# episode's context only on the step an episode ends), and the slot's own index.
TRANSITION_CONTEXT_KEY = "transition_context"
SLOT_INDEX_KEY = "slot_index"

# The state.info keys a round hook needs the trainer to record per transition.
ROLLOUT_FIELDS = (TRANSITION_CONTEXT_KEY, SLOT_INDEX_KEY, "episode_metrics", "episode_done")


@dataclasses.dataclass(frozen=True)
class Transitions:
    """Every transition of one round, ``[num_slots, steps_per_slot]``, chronological per slot."""

    contexts: np.ndarray  # [S, L, D]  context each transition was generated in
    episode_done: np.ndarray  # [S, L]  True where an episode ended
    episode_return: np.ndarray  # [S, L]  total reward of the episode ending there (else partial sum)
    episode_cost: np.ndarray  # [S, L]
    episode_length: np.ndarray  # [S, L]
    reward_advantage: np.ndarray  # [S, L]  GAE of the reward at that transition, pre-update value function

    @classmethod
    def from_recorded_fields(
        cls, recorded: Mapping[str, np.ndarray], learning_signals: LearningSignals, num_slots: int
    ) -> "Transitions":
        """From the trainer's layout to per-slot time series.

        The trainer records every field as ``[rows, unroll_length, ...]``: one row per
        (unroll, slot), in whatever order it flattens them. Each row carries the slot
        it came from (``SLOT_INDEX_KEY``, written by the wrapper), so the rows of a
        slot are found by that label, not by assuming the trainer's order. Within a
        slot the rows are in time order, as the trainer's scan produced them.
        """
        for field in ROLLOUT_FIELDS:
            if field not in recorded:
                raise KeyError(f"the trainer did not record {field!r}; is the hook's extra_fields wired into train()?")
        order = _rows_by_slot(np.asarray(recorded[SLOT_INDEX_KEY]), num_slots)
        metrics = recorded["episode_metrics"]
        per_slot = lambda x: _per_slot(np.asarray(x), order)
        return cls(
            contexts=per_slot(recorded[TRANSITION_CONTEXT_KEY]),
            episode_done=per_slot(recorded["episode_done"]).astype(bool),
            episode_return=per_slot(metrics["sum_reward"]),
            episode_cost=per_slot(metrics["cost"]),
            episode_length=per_slot(metrics["length"]),
            reward_advantage=per_slot(learning_signals.reward_advantage),
        )

    def __post_init__(self) -> None:
        shape = self.contexts.shape[:2]
        for name in ("episode_done", "episode_return", "episode_cost", "episode_length", "reward_advantage"):
            if getattr(self, name).shape != shape:
                raise ValueError(f"Transitions.{name} has shape {getattr(self, name).shape}, expected {shape}")
        # The layout invariant: within a slot the context changes only on the step after
        # an episode ended. A wrong slot/time layout breaks this at once.
        changed = np.any(self.contexts[:, 1:] != self.contexts[:, :-1], axis=-1)
        if np.any(changed & ~self.episode_done[:, :-1]):
            raise ValueError("a slot's context changed without its episode ending: the slot/time layout is wrong")

    @property
    def num_slots(self) -> int:
        return int(self.contexts.shape[0])

    @property
    def steps_per_slot(self) -> int:
        return int(self.contexts.shape[1])

    @property
    def num_transitions(self) -> int:
        return self.num_slots * self.steps_per_slot

    @property
    def flat_contexts(self) -> np.ndarray:
        """``[S × L, D]``: one row per transition, slot order irrelevant."""
        return self.contexts.reshape(-1, self.contexts.shape[-1])

    @property
    def flat_reward_advantage(self) -> np.ndarray:
        return self.reward_advantage.reshape(-1)


@dataclasses.dataclass(frozen=True)
class CompletedEpisodes:
    """One row per episode that ended in the round; ``value_loss`` spans the whole episode."""

    contexts: np.ndarray  # [N, D]
    returns: np.ndarray  # [N]  undiscounted episodic return
    costs: np.ndarray  # [N]  undiscounted episodic cost
    lengths: np.ndarray  # [N]  steps from the episode's first observed transition to its last
    value_loss: np.ndarray  # [N]  mean |reward advantage| over all of the episode's transitions (PLR's L1 value loss)

    def __post_init__(self) -> None:
        count = self.contexts.shape[0]
        for name in ("returns", "costs", "lengths", "value_loss"):
            if getattr(self, name).shape != (count,):
                raise ValueError(f"CompletedEpisodes.{name} has shape {getattr(self, name).shape}, expected ({count},)")

    @property
    def count(self) -> int:
        return int(self.contexts.shape[0])


@dataclasses.dataclass(frozen=True)
class RoundRollout:
    """One round: all its transitions, and the episodes that ended in it."""

    round_index: int
    transitions: Transitions
    completed_episodes: CompletedEpisodes


def _rows_by_slot(slot_index: np.ndarray, num_slots: int) -> np.ndarray:
    """``[S, U]``: the recorded rows of each slot, in the order they were recorded.

    ``slot_index`` is ``[rows, unroll_length]``, constant along each row. Raises when
    the labels do not describe ``num_slots`` slots with the same number of rows each.
    """
    if slot_index.ndim != 2:
        raise ValueError(f"{SLOT_INDEX_KEY} has shape {slot_index.shape}; expected [rows, unroll_length]")
    if np.any(slot_index != slot_index[:, :1]):
        raise ValueError(f"{SLOT_INDEX_KEY} changes within a recorded row: a row must hold one slot's unroll")
    labels = slot_index[:, 0].astype(np.int64)
    counts = np.bincount(labels, minlength=num_slots)
    if labels.min() < 0 or labels.max() >= num_slots or np.any(counts != counts[0]):
        raise ValueError(f"{SLOT_INDEX_KEY} does not label {num_slots} slots with equally many rows each: counts {counts}")
    return np.argsort(labels, kind="stable").reshape(num_slots, counts[0])


def _per_slot(x: np.ndarray, rows_by_slot: np.ndarray) -> np.ndarray:
    """``[rows, T, *rest] -> [S, U × T, *rest]`` using the row order from :func:`_rows_by_slot`."""
    num_slots, num_unrolls = rows_by_slot.shape
    gathered = x[rows_by_slot]  # [S, U, T, *rest]
    return gathered.reshape(num_slots, num_unrolls * x.shape[1], *x.shape[2:])
