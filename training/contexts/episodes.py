"""From per-slot transitions to completed episodes, across round boundaries.

An episode is ~1000 steps; a slot contributes ~80 steps per round. An episode
therefore spans about twelve rounds, and a statistic that is meant to describe
the *whole* episode (PLR's L1 value loss, ``docs/acl/design/prioritized_level_replay.md``
fact 4) has to be accumulated over all of them. :class:`EpisodeTracker` holds,
per slot and per requested :class:`~training.contexts.statistics.EpisodeStatistic`,
the fold's running value since the slot's last episode ended, and closes it into
:class:`CompletedEpisodes` wherever ``episode_done`` fires.

The arithmetic does not know what it reduces. :class:`EpisodeEnds` is the
geometry of one round — which transitions belong to which completed episode,
what is left over per slot — and reduces any ``[slots, steps]`` array over that
geometry with any associative ufunc and its identity. The tracker applies it,
per statistic, to ``fold.map(signal)`` with ``fold.combine`` / ``fold.identity``,
and once more with ``np.add`` / 0 over ones for the transition count. A new
statistic is a new ``EpisodeStatistic``; nothing here changes
(``docs/acl/design/learning_signals.md``).

The trainer never resets the environment between rounds while a round hook is
installed (``ppo/train.py`` refuses ``num_resets_per_eval``), so slot ``s`` in
round ``k + 1`` continues the time series of slot ``s`` in round ``k``, which is
what makes the carry valid.
"""
from __future__ import annotations

import dataclasses
from typing import Dict, FrozenSet, Tuple

import numpy as np

from training.contexts.rollout import CompletedEpisodes, Transitions
from training.contexts.statistics import EpisodeStatistic


@dataclasses.dataclass(frozen=True)
class EpisodeEnds:
    """Where the episodes of one round ended, and the segments of transitions that implies.

    Every done, in (slot, time) order, is one completed episode. Its *segment* is
    the transitions after the previous done in the same slot up to and including
    the done — or, for the first done of a slot this round, from the start of the
    round; that episode also owns the slot's carry from earlier rounds. What
    follows a slot's last done (or the whole round, if it had none) is the slot's
    *tail*, the next carry; a slot whose last done was its last step has an empty
    tail, whose reduction is the fold's identity.
    """

    num_slots: int
    steps: int
    slots: np.ndarray  # [N] slot of each completed episode
    times: np.ndarray  # [N] step its done fired at
    first_in_slot: np.ndarray  # [N] True for the first done of its slot this round
    has_done: np.ndarray  # [S]
    _segment_starts: np.ndarray  # flat [S × L] indices where every segment and non-empty tail begins, increasing
    _closed_segments: np.ndarray  # [N] index into _segment_starts of each completed episode's segment
    _tail_segments: np.ndarray  # [S] index into _segment_starts of each slot's tail (−1 where empty)

    @classmethod
    def find(cls, episode_done: np.ndarray) -> "EpisodeEnds":
        num_slots, steps = episode_done.shape
        slots, times = np.nonzero(episode_done)
        first_in_slot = np.ones(slots.shape, dtype=bool)
        first_in_slot[1:] = slots[1:] != slots[:-1]
        has_done = episode_done.any(axis=1)
        last_done = np.where(has_done, steps - 1 - np.argmax(episode_done[:, ::-1], axis=1), -1)

        # Every segment start in flat [S × L] order: each completed episode's, then each
        # slot's non-empty tail (right after its last done; the whole row where none).
        # Increasing by construction, which is what ufunc.reduceat needs.
        previous_time = np.full(slots.shape, -1)
        previous_time[1:] = np.where(first_in_slot[1:], -1, times[:-1])
        closed_starts = slots * steps + previous_time + 1
        tail_slots = np.flatnonzero(last_done < steps - 1)
        tail_starts = tail_slots * steps + last_done[tail_slots] + 1
        starts = np.concatenate([closed_starts, tail_starts])
        order = np.argsort(starts, kind="stable")
        position = np.argsort(order)  # where each start landed in the sorted array
        tail_segments = np.full(num_slots, -1, dtype=np.int64)
        tail_segments[tail_slots] = position[slots.size:]
        return cls(num_slots, steps, slots, times, first_in_slot, has_done,
                   starts[order], position[: slots.size], tail_segments)

    def reduce(
        self, per_transition: np.ndarray, combine: np.ufunc, identity: float, carry: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Reduce ``per_transition`` ``[S, L]`` with the monoid (``combine``, ``identity``)
        over each completed episode, joining ``carry`` ``[S]`` onto the first episode of
        each slot.

        Returns the per-episode values ``[N]`` and the next carry ``[S]``: the tail after
        the slot's last done (``identity`` if empty), or ``carry`` joined with the whole
        round where no episode ended.
        """
        if per_transition.shape != (self.num_slots, self.steps) or carry.shape != (self.num_slots,):
            raise ValueError(
                f"expected a [{self.num_slots}, {self.steps}] quantity with a [{self.num_slots}] carry, "
                f"got {per_transition.shape} and {carry.shape}"
            )
        segments = combine.reduceat(per_transition.reshape(-1), self._segment_starts)
        closed = segments[self._closed_segments]
        closed = np.where(self.first_in_slot, combine(carry[self.slots], closed), closed)
        tail = np.where(self._tail_segments >= 0, segments[np.maximum(self._tail_segments, 0)], identity)
        # A slot with a done has handed its carry to that episode; one without keeps combining.
        next_carry = combine(np.where(self.has_done, identity, carry), tail)
        return closed, next_carry


class EpisodeTracker:
    """Per-slot running folds of the episode in progress, carried between rounds.

    ``statistics`` is what every round's :class:`CompletedEpisodes` will carry: the
    distribution's requirements. The tracker adds nothing of its own.
    """

    def __init__(self, num_slots: int, statistics: FrozenSet[EpisodeStatistic]):
        self.num_slots = num_slots
        self.statistics = statistics
        self._carry: Dict[EpisodeStatistic, np.ndarray] = {
            statistic: np.full(num_slots, statistic.fold.identity, dtype=np.float64) for statistic in statistics
        }
        self._count_carry = np.zeros(num_slots, dtype=np.int64)

    def complete(self, transitions: Transitions) -> CompletedEpisodes:
        """The episodes that ended in ``transitions``, each statistic over its whole length; advances the carry."""
        if transitions.num_slots != self.num_slots:
            raise ValueError(f"EpisodeTracker was built for {self.num_slots} slots, got a round with {transitions.num_slots}")
        missing = {statistic.signal for statistic in self.statistics} - set(transitions.signals)
        if missing:
            raise KeyError(f"the round's signals lack {sorted(signal.name for signal in missing)}, which the requested statistics need")
        ends = EpisodeEnds.find(transitions.episode_done)
        ones = np.ones((self.num_slots, transitions.steps_per_slot), dtype=np.int64)
        count, self._count_carry = ends.reduce(ones, np.add, 0, self._count_carry)

        statistics: Dict[EpisodeStatistic, np.ndarray] = {}
        for statistic in self.statistics:
            fold = statistic.fold
            mapped = fold.map(transitions.signals[statistic.signal]).astype(np.float64)
            combined, self._carry[statistic] = ends.reduce(mapped, fold.combine, fold.identity, self._carry[statistic])
            statistics[statistic] = fold.finalise(combined, count)

        return CompletedEpisodes(
            contexts=transitions.contexts[ends.slots, ends.times],
            returns=transitions.episode_return[ends.slots, ends.times],
            costs=transitions.episode_cost[ends.slots, ends.times],
            lengths=transitions.episode_length[ends.slots, ends.times],
            statistics=statistics,
        )
