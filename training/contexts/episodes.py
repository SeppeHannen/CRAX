"""From per-slot transitions to completed episodes, across round boundaries.

An episode is ~1000 steps; a slot contributes ~80 steps per round. An episode
therefore spans about twelve rounds, and a score that is meant to describe the
*whole* episode (PLR's L1 value loss, ``docs/acl/design/prioritized_level_replay.md``
fact 4) has to be accumulated over all of them. :class:`EpisodeTracker` holds,
per slot, the running sum of |reward advantage| and the number of transitions
since the slot's last episode ended, and closes those sums into
:class:`CompletedEpisodes` wherever ``episode_done`` fires.

The trainer never resets the environment between rounds while a round hook is
installed (``ppo/train.py`` refuses ``num_resets_per_eval``), so slot ``s`` in
round ``k + 1`` continues the time series of slot ``s`` in round ``k``, which is
what makes the carry valid.
"""
from __future__ import annotations

import numpy as np

from training.contexts.rollout import CompletedEpisodes, Transitions


class EpisodeTracker:
    """Per-slot partial sums of the episode in progress, carried between rounds."""

    def __init__(self, num_slots: int):
        self.num_slots = num_slots
        self._advantage_sum = np.zeros(num_slots, dtype=np.float64)  # Σ |reward advantage| since the slot's last done
        self._transition_count = np.zeros(num_slots, dtype=np.int64)

    def complete(self, transitions: Transitions) -> CompletedEpisodes:
        """The episodes that ended in ``transitions``, scored over their whole length; advances the carry."""
        if transitions.num_slots != self.num_slots:
            raise ValueError(f"EpisodeTracker was built for {self.num_slots} slots, got a round with {transitions.num_slots}")
        magnitude = np.abs(transitions.reward_advantage).astype(np.float64)
        cumulative = np.cumsum(magnitude, axis=1)  # [S, L]
        done = transitions.episode_done
        steps = transitions.steps_per_slot

        # Every done, in (slot, time) order. The episode ending at a done consists of the
        # transitions after the previous done in the same slot — or, for the first done of
        # a slot this round, the carry plus the transitions from the start of the round.
        slots, times = np.nonzero(done)
        first_in_slot = np.ones(slots.shape, dtype=bool)
        first_in_slot[1:] = slots[1:] != slots[:-1]
        # `cumulative` up to and including the previous done of the same slot (0 where none).
        before_segment = np.zeros(slots.shape, dtype=np.float64)
        before_segment[1:] = np.where(first_in_slot[1:], 0.0, cumulative[slots[:-1], times[:-1]])
        previous_time = np.full(slots.shape, -1)
        previous_time[1:] = np.where(first_in_slot[1:], -1, times[:-1])
        episode_sum = cumulative[slots, times] - before_segment + np.where(first_in_slot, self._advantage_sum[slots], 0.0)
        episode_count = (times - previous_time) + np.where(first_in_slot, self._transition_count[slots], 0)

        # Carry: what is left after the last done in each slot (or everything, if none).
        has_done = done.any(axis=1)
        last_done = np.where(has_done, steps - 1 - np.argmax(done[:, ::-1], axis=1), -1)
        after_last = cumulative[:, -1] - np.where(has_done, cumulative[np.arange(self.num_slots), np.maximum(last_done, 0)], 0.0)
        self._advantage_sum = np.where(has_done, after_last, self._advantage_sum + cumulative[:, -1])
        self._transition_count = np.where(has_done, steps - 1 - last_done, self._transition_count + steps)

        return CompletedEpisodes(
            contexts=transitions.contexts[slots, times],
            returns=transitions.episode_return[slots, times],
            costs=transitions.episode_cost[slots, times],
            lengths=transitions.episode_length[slots, times],
            value_loss=episode_sum / episode_count,
        )
