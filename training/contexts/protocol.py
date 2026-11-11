"""Evaluation protocols: which contexts a run trains on, which are held out to
test it on, and the cells a trained policy is scored on afterwards.

The words are Kirk et al.'s (2023, §4.2; ``docs/acl/literature/``): an
evaluation protocol is the training and the testing context sets. Here it is
three things over one region of Ω, the protocol's **contexts**:

* the **training** contexts, which every training distribution samples from;
* the **held-out** regions, each a named test (:class:`HeldOutRegion`);
* the **cells** (:class:`Cell`): a finer cut of the same contexts, one value of
  each factor the protocol is about (one hazard total, one band of drag), on
  which ``training/score.py`` measures cost and return after training.

Training and the held-out regions are a partition of the contexts, and every
cell lies in exactly one of them; both are checked when a protocol is built.
Which regions a suite has is the registry's (``registry.py``); what a run does
with a protocol is ``setup.py``'s. The design is
``docs/acl/design/generalization_evaluation.md``.
"""
from __future__ import annotations

import dataclasses
from typing import Sequence, Tuple

from training.contexts.space import ContextSpace

# The protocol every suite has: train on all of Ω, hold nothing out.
NO_PROTOCOL = "none"
# What a protocol's training contexts are called wherever a region is named: the evaluation
# on them (``evaluation/training/*``) and the region column of a score table.
TRAINING_REGION = "training"

Coordinates = Tuple[Tuple[str, float], ...]
"""A cell's value of each factor, in the protocol's factor order: ``(("hazard_total", 12.0),)``."""


@dataclasses.dataclass(frozen=True)
class HeldOutRegion:
    """A named region of contexts no training distribution may sample, on which the policy is
    scored. ``name`` becomes the metric prefix ``evaluation/<name>/``; ``label`` is what the
    dashboard calls it."""

    name: str
    space: ContextSpace
    label: str


@dataclasses.dataclass(frozen=True)
class Cell:
    """A small box of contexts with one value of each of a protocol's factors."""

    coordinates: Coordinates
    space: ContextSpace


def split_by_integer_total(cells: Sequence[Cell], factor: str) -> Tuple[Cell, ...]:
    """Every cell cut into one cell per value of its integer total; the value is the cell's
    coordinate on ``factor``."""
    split = []
    for cell in cells:
        totals = cell.space.integer_points.sum(axis=1)
        for total in range(int(totals.min()), int(totals.max()) + 1):
            split.append(Cell((*cell.coordinates, (factor, float(total))), cell.space.with_integer_total(total, total)))
    return tuple(split)


def split_by_interval(cells: Sequence[Cell], dimension: str, edges: Sequence[float]) -> Tuple[Cell, ...]:
    """Every cell cut along the continuous coordinate ``dimension`` at ``edges``, which run from
    the cell's lower bound to its upper bound; the middle of a band is the cell's coordinate on
    the factor named after the dimension."""
    split = []
    for cell in cells:
        whole = cell.space.dimensions[cell.space.index(dimension)]
        if edges[0] != whole.low or edges[-1] != whole.high:
            raise ValueError(f"split_by_interval: edges {edges[0]}..{edges[-1]} do not span {dimension!r}, [{whole.low}, {whole.high}]")
        for low, high in zip(edges[:-1], edges[1:]):
            split.append(Cell((*cell.coordinates, (dimension, (low + high) / 2.0)), cell.space.with_interval(dimension, low, high)))
    return tuple(split)


@dataclasses.dataclass(frozen=True)
class EvaluationProtocol:
    """Which contexts a run trains on, which are held out to test it on, and the cells it is
    scored on (module docstring).

    ``contexts`` is everything the protocol is about, a region of Ω: the networks scale a
    context by its bounds and the Lagrange function has a factor per coordinate that varies
    in it. ``training`` is the part of it every training distribution samples from;
    ``held_out`` are the test regions. A **reference** run trains on all of ``contexts``,
    test regions included; it shows what was achievable on them.
    """

    name: str
    contexts: ContextSpace
    training: ContextSpace
    held_out: Tuple[HeldOutRegion, ...]
    cells: Tuple[Cell, ...]

    def __post_init__(self) -> None:
        names = [region.name for region in self.held_out]
        if len(set(names)) != len(names) or TRAINING_REGION in names:
            raise ValueError(f"protocol {self.name!r}: held-out regions need distinct names other than {TRAINING_REGION!r}, got {names}")
        regions = self._regions
        for name, region in regions:
            if not region.is_within(self.contexts):
                raise ValueError(f"protocol {self.name!r}: {name} is not within the protocol's contexts\n{region.describe()}\n{self.contexts.describe()}")
        for index, (first_name, first) in enumerate(regions):
            for second_name, second in regions[index + 1:]:
                if not first.is_disjoint_from(second):
                    raise ValueError(f"protocol {self.name!r}: {first_name} and {second_name} overlap\n{first.describe()}\n{second.describe()}")
        self._require_all_of_the_contexts("training and held-out regions", [region for _, region in regions])
        for cell in self.cells:
            self.region_of(cell)
        self._require_all_of_the_contexts("cells", [cell.space for cell in self.cells])

    @property
    def _regions(self) -> Tuple[Tuple[str, ContextSpace], ...]:
        return ((TRAINING_REGION, self.training), *((region.name, region.space) for region in self.held_out))

    def _require_all_of_the_contexts(self, what: str, pieces: Sequence[ContextSpace]) -> None:
        """Pieces known not to overlap cover the contexts when their measures add up to the contexts'."""
        covered = sum(piece.measure for piece in pieces)
        if abs(covered - self.contexts.measure) > 1e-6 * self.contexts.measure:
            raise ValueError(
                f"protocol {self.name!r}: the {what} cover {covered:g} of the protocol's contexts ({self.contexts.measure:g}); "
                f"they must cover every context exactly once"
            )

    def region_of(self, cell: Cell) -> str:
        """The one region ``cell`` lies in: :data:`TRAINING_REGION` or a held-out region's name."""
        containing = [name for name, region in self._regions if cell.space.is_within(region)]
        if len(containing) != 1:
            raise ValueError(
                f"protocol {self.name!r}: the cell at {dict(cell.coordinates)} lies in {containing or 'no region'}; a cell lies "
                f"within exactly one of {[name for name, _ in self._regions]}\n{cell.space.describe()}"
            )
        return containing[0]

    def training_contexts(self, reference: bool) -> ContextSpace:
        """What a run's training distribution may sample: the training contexts, or for a
        reference run all of the protocol's contexts."""
        return self.contexts if reference else self.training


def no_protocol(space: ContextSpace) -> EvaluationProtocol:
    """Train on all of ``space``, hold nothing out; scored as one cell."""
    return EvaluationProtocol(NO_PROTOCOL, contexts=space, training=space, held_out=(), cells=(Cell((), space),))
