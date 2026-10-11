"""Episode statistics: one number per completed episode, from one signal.

A distribution that learns from the learner's view of the rollout asks for
:class:`EpisodeStatistic`\\ s — *which* :class:`~training.rounds.Signal`, reduced
*how* (:class:`Fold`) over all of an episode's transitions. PLR's L1 value loss
is the mean |advantage|: :class:`MeanAbsolute` over an advantage signal. The
episode tracker (``training/contexts/episodes.py``) computes whatever
statistics it is given and knows nothing about what they mean.

Why a fold. Episodes outlast rounds and transitions are not stored, so a
statistic is accumulated across rounds in a per-slot carry. That is only
possible when the reduction is an **associative** combine of a per-transition
map with an **identity** (a monoid): the result must not depend on where the
round boundaries fell, and a slot that has nothing carried holds the identity.
Mean, sum, min, max, moment-based variance are; an exact median is not, and
cannot be written against :class:`Fold`. ``docs/acl/design/learning_signals.md``
states the limitation and the way out (buffer the episode) should a method ever
need one.
"""
from __future__ import annotations

import dataclasses
from typing import ClassVar, Protocol, runtime_checkable

import numpy as np

from training.rounds import Signal


@runtime_checkable
class Fold(Protocol):
    """A reduction over an episode's transitions that can be carried across rounds.

    ``combine`` is an associative binary NumPy ufunc (``np.add``, ``np.maximum``,
    ...) and ``identity`` its neutral element (0 for ``np.add``, ``-inf`` for
    ``np.maximum``): a slot that has nothing carried from earlier rounds holds the
    identity, so the tracker joins the carry onto every slot's first episode
    without asking whether there was one. ``map`` turns the raw signal into what
    is combined; ``finalise`` turns the combined value and the number of
    transitions it covers into the statistic. Instances must be hashable (frozen,
    no array fields): they are part of a statistic's identity.
    """

    combine: ClassVar[np.ufunc]
    identity: ClassVar[float]

    def map(self, signal: np.ndarray) -> np.ndarray:
        """Per-transition: the quantity to combine."""

    def finalise(self, combined: np.ndarray, count: np.ndarray) -> np.ndarray:
        """Per episode: the statistic from the combined value and the transition count."""


@dataclasses.dataclass(frozen=True)
class MeanAbsolute:
    """mean |x| over the episode — PLR's L1 value loss when ``x`` is an advantage."""

    combine: ClassVar[np.ufunc] = np.add
    identity: ClassVar[float] = 0.0

    def map(self, signal: np.ndarray) -> np.ndarray:
        return np.abs(signal)

    def finalise(self, combined: np.ndarray, count: np.ndarray) -> np.ndarray:
        return combined / count


@dataclasses.dataclass(frozen=True)
class EpisodeStatistic:
    """What a distribution asks the episode tracker for, and the key it reads the answer under."""

    signal: Signal
    fold: Fold

    def __post_init__(self) -> None:
        if not isinstance(self.signal, Signal):
            raise TypeError(f"signal must be a Signal, got {self.signal!r}")
        if not isinstance(self.fold, Fold):
            raise TypeError(f"fold must implement Fold (combine, identity, map, finalise), got {self.fold!r}")
        if not isinstance(self.fold.combine, np.ufunc):
            raise TypeError(f"{type(self.fold).__name__}.combine must be a NumPy ufunc, got {self.fold.combine!r}")
        probe = np.array([-7.0, 7.0])
        if not np.array_equal(self.fold.combine(self.fold.identity, probe), probe):
            raise ValueError(
                f"{type(self.fold).__name__}.identity = {self.fold.identity!r} is not neutral for "
                f"{self.fold.combine.__name__}: combine(identity, x) must equal x"
            )

    @property
    def name(self) -> str:
        """``<signal>/<fold>``, for messages and metric keys."""
        return f"{self.signal.value}/{type(self.fold).__name__}"
