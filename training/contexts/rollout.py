"""What one training round produced, as a distribution sees it.

The trainer records, for every transition, the context it was generated in and
EpisodeWrapper's bookkeeping (``episode_done`` marks the transitions that ended
an episode; ``episode_metrics`` then holds that episode's totals), and hands
over the learner's view of the same transitions: one array per
:class:`~training.rounds.Signal` the agent produces, pre-update critics. On the
host this becomes a :class:`RoundRollout` with two views of the same data:

* :class:`Transitions` — every transition of the round, laid out as
  ``[num_slots, steps_per_slot]`` in the order each slot experienced them. The
  realised curriculum q̂ and the per-context value error are read from here.
* :class:`CompletedEpisodes` — one row per episode that *ended* in the round,
  with its context, totals and every requested
  :class:`~training.contexts.statistics.EpisodeStatistic` over the **whole**
  episode, including the part that ran in earlier rounds
  (:class:`training.contexts.episodes.EpisodeTracker` carries it over). The
  empirical q and PLR's score are read from here.

Nothing here knows what a signal or a statistic means: both are mappings keyed
by the vocabulary in ``training/rounds.py`` and ``training/contexts/statistics.py``
(``docs/acl/design/learning_signals.md``).
"""
from __future__ import annotations

import dataclasses
from typing import Mapping

import numpy as np

from training.contexts.statistics import EpisodeStatistic
from training.rounds import Signal

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
    signals: Mapping[Signal, np.ndarray]  # [S, L] each  the learner's signals at that transition, pre-update critics

    @classmethod
    def from_recorded_fields(
        cls, recorded: Mapping[str, np.ndarray], learning_signals: Mapping[Signal, np.ndarray], num_slots: int
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
            signals={Signal(signal): per_slot(values) for signal, values in learning_signals.items()},
        )

    def __post_init__(self) -> None:
        shape = self.contexts.shape[:2]
        for name in ("episode_done", "episode_return", "episode_cost", "episode_length"):
            if getattr(self, name).shape != shape:
                raise ValueError(f"Transitions.{name} has shape {getattr(self, name).shape}, expected {shape}")
        for signal, values in self.signals.items():
            if not isinstance(signal, Signal):
                raise TypeError(f"Transitions.signals is keyed by {signal!r}; keys must be Signal members")
            if values.shape != shape:
                raise ValueError(f"Transitions.signals[{signal.name}] has shape {values.shape}, expected {shape}")
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
    def flat_signals(self) -> Mapping[Signal, np.ndarray]:
        """``[S × L]`` per signal: one value per transition, in the order of :attr:`flat_contexts`."""
        return {signal: values.reshape(-1) for signal, values in self.signals.items()}


@dataclasses.dataclass(frozen=True)
class CompletedEpisodes:
    """One row per episode that ended in the round; the statistics span the whole episode."""

    contexts: np.ndarray  # [N, D]
    returns: np.ndarray  # [N]  undiscounted episodic return
    costs: np.ndarray  # [N]  undiscounted episodic cost
    lengths: np.ndarray  # [N]  steps from the episode's first observed transition to its last
    statistics: Mapping[EpisodeStatistic, np.ndarray]  # [N] each  the requested statistics over all of the episode's transitions

    def __post_init__(self) -> None:
        count = self.contexts.shape[0]
        for name in ("returns", "costs", "lengths"):
            if getattr(self, name).shape != (count,):
                raise ValueError(f"CompletedEpisodes.{name} has shape {getattr(self, name).shape}, expected ({count},)")
        for statistic, values in self.statistics.items():
            if values.shape != (count,):
                raise ValueError(f"CompletedEpisodes.statistics[{statistic.name}] has shape {values.shape}, expected ({count},)")

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
