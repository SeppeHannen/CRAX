"""Context spaces: the set Ω of tasks an environment suite can instantiate.

A context ω ∈ Ω is a fixed-length vector of floats. Each coordinate is a
:class:`Dimension` with a name and bounds. Integer and categorical dimensions
are stored as floats too (rounded / used as indices when decoded), so that a
batch of contexts is always one ``[num_envs, D]`` float array and can live in
``state.info["context"]`` inside a compiled JAX program.

Only *values* live here. Anything that would change an array shape or the
MuJoCo model (number of hazards, object types) is not a dimension of Ω in this
sense; such suites are parameterised through a fixed-size mask (see
``docs/acl/design/per_slot_constraints.md``).

Ω is a box, optionally cut by one constraint: bounds on the **sum of the integer
dimensions**. A hazard suite's integer dimensions are per-kind active counts,
and the arena can hold only so many hazards at once; the cap states that
limit so that "uniform over Ω" never asks for a layout that does not fit.

A **region** of Ω — the contexts a run trains on when some are held out for
testing, or a held-out band itself — is a :class:`ContextSpace` over the same
dimensions with tighter bounds (:meth:`ContextSpace.with_integer_total`,
:meth:`ContextSpace.with_interval`), or a :class:`ContextUnion` of such spaces.
A region is a ``ContextSpace`` to everything that samples from it or checks
membership.
"""
from __future__ import annotations

import dataclasses
import itertools
from typing import Dict, Literal, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np

Context = jax.Array  # shape [D]
Contexts = jax.Array  # shape [N, D]

DimensionKind = Literal["continuous", "integer"]
DifficultyDirection = Literal["higher", "lower"]


@dataclasses.dataclass(frozen=True)
class Dimension:
    """One coordinate of a context space.

    ``harder_when`` says which way difficulty runs along this coordinate: ``"higher"``
    when a larger value makes the task harder (more hazards, a wider gap), ``"lower"``
    when a smaller one does (a smaller goal, a tighter speed limit, a lower ceiling).
    A fact of the suite, stated once here; the additive multiplier
    (``training/contexts/constraints/additive.py``) is monotone in this direction.

    ``hidden`` marks a coordinate a deployed robot could not know (a property of the
    physics, not of the task it was given). No policy is told a hidden coordinate,
    whatever the learner; the critics and the Lagrange function exist only in training
    and read it (``docs/acl/design/generalization_evaluation.md``).
    """

    name: str
    low: float
    high: float
    kind: DimensionKind = "continuous"
    description: str = ""
    harder_when: DifficultyDirection = "higher"
    hidden: bool = False

    def __post_init__(self) -> None:
        if not self.low <= self.high:
            raise ValueError(f"Dimension {self.name!r}: low ({self.low}) > high ({self.high})")

    @property
    def is_degenerate(self) -> bool:
        return self.low == self.high


@dataclasses.dataclass(frozen=True)
class Bounds:
    """Ω's bounds per dimension as plain floats: what a network needs to scale a context, in a
    form a checkpoint's JSON config can hold (a :class:`ContextSpace` cannot be a factory keyword)."""

    low: Tuple[float, ...]
    high: Tuple[float, ...]

    def __post_init__(self) -> None:
        if len(self.low) != len(self.high):
            raise ValueError(f"Bounds: {len(self.low)} lower bounds against {len(self.high)} upper bounds")
        if any(l > h for l, h in zip(self.low, self.high)):
            raise ValueError(f"Bounds: a lower bound exceeds its upper bound: low={self.low}, high={self.high}")


def scale_to_bounds(contexts: jax.Array, bounds: Bounds) -> jax.Array:
    """Contexts ``[..., D]`` mapped to [-1, 1] per dimension over ``bounds``; 0 on a degenerate dimension.

    The one way a context enters a network. Its scale is known before training — Ω is
    declared — and so does not depend on what the curriculum samples; a running
    normaliser's does, and under one context per stage (``staged``) its standard deviation
    collapses to the floor and every other context maps to ~10⁶ (found 2026-10-15).
    """
    low = jnp.asarray(bounds.low, dtype=jnp.float32)
    high = jnp.asarray(bounds.high, dtype=jnp.float32)
    middle = (high + low) / 2.0
    half_range = (high - low) / 2.0
    degenerate = half_range == 0.0
    return jnp.where(degenerate, 0.0, (contexts - middle) / jnp.where(degenerate, 1.0, half_range))


@dataclasses.dataclass(frozen=True)
class ContextSpace:
    """A box-shaped Ω with named coordinates, optionally bounded in its integer total.

    ``integer_total_cap`` / ``integer_total_floor``: if set, only points whose
    integer coordinates sum to at most / at least this value belong to Ω.
    Instances are immutable and hashable, so they can be static arguments to
    jitted functions.
    """

    dimensions: Tuple[Dimension, ...]
    integer_total_cap: Optional[int] = None
    integer_total_floor: Optional[int] = None

    def __post_init__(self) -> None:
        names = [d.name for d in self.dimensions]
        if len(set(names)) != len(names):
            raise ValueError(f"Duplicate dimension names: {names}")
        if not self.dimensions:
            raise ValueError("A context space needs at least one dimension")
        if (self.integer_total_cap is not None or self.integer_total_floor is not None) and not self.integer_dimensions:
            raise ValueError("an integer total bound was given but Ω has no integer dimension")
        if self.integer_total_cap is not None:
            least = sum(int(d.low) for d in self.integer_dimensions)
            if self.integer_total_cap < least:
                raise ValueError(f"integer_total_cap {self.integer_total_cap} is below the smallest possible total {least}")
        if self.integer_total_floor is not None:
            most = sum(int(d.high) for d in self.integer_dimensions)
            if self.integer_total_floor > most:
                raise ValueError(f"integer_total_floor {self.integer_total_floor} is above the largest possible total {most}")
            if self.integer_total_cap is not None and self.integer_total_floor > self.integer_total_cap:
                raise ValueError(f"integer_total_floor {self.integer_total_floor} exceeds integer_total_cap {self.integer_total_cap}")

    # ---- regions of this space -------------------------------------------- #

    def with_integer_total(self, low: int, high: int) -> "ContextSpace":
        """The part of this space whose integer coordinates sum to a value in ``[low, high]``:
        the same box, fewer admissible integer tuples."""
        if low > high:
            raise ValueError(f"with_integer_total: low ({low}) > high ({high})")
        totals = self.integer_points.sum(axis=1)
        if low < totals.min() or high > totals.max():
            raise ValueError(
                f"with_integer_total([{low}, {high}]) reaches outside this space, whose integer totals run from "
                f"{int(totals.min())} to {int(totals.max())}"
            )
        return dataclasses.replace(self, integer_total_floor=low, integer_total_cap=high)

    def with_interval(self, name: str, low: float, high: float) -> "ContextSpace":
        """The part of this space whose continuous coordinate ``name`` lies in ``[low, high]``;
        ``low == high`` fixes it at that value."""
        index = self.index(name)
        dimension = self.dimensions[index]
        if dimension.kind != "continuous":
            raise ValueError(f"with_interval: {name!r} is not a continuous coordinate")
        if not dimension.low <= low <= high <= dimension.high:
            raise ValueError(f"with_interval({name!r}, {low}, {high}) is not an interval within [{dimension.low}, {dimension.high}]")
        narrowed = dataclasses.replace(dimension, low=low, high=high)
        return dataclasses.replace(self, dimensions=(*self.dimensions[:index], narrowed, *self.dimensions[index + 1:]))

    @property
    def boxes(self) -> Tuple["ContextSpace", ...]:
        """The box-shaped pieces this space is made of: itself. A :class:`ContextUnion` has several."""
        return (self,)

    @property
    def measure(self) -> float:
        """How much of the box this space covers, for weighing regions against each other: the
        number of admissible integer tuples times the length of every continuous interval that
        has one (a degenerate interval, a fixed value, counts as 1)."""
        lengths = [d.high - d.low for d in self.dimensions if d.kind == "continuous" and not d.is_degenerate]
        return float(len(self.integer_points) * np.prod(lengths))

    def is_within(self, other: "ContextSpace") -> bool:
        """Every box of this space lies inside one box of ``other`` (a box that straddles two
        boxes of ``other`` is not recognised; cut it at their border)."""
        return all(any(_box_within(box, container) for container in other.boxes) for box in self.boxes)

    def is_disjoint_from(self, other: "ContextSpace") -> bool:
        """No context belongs to both (two continuous intervals that only touch do not count)."""
        return all(_boxes_disjoint(first, second) for first in self.boxes for second in other.boxes)

    # ---- introspection --------------------------------------------------- #

    @property
    def size(self) -> int:
        return len(self.dimensions)

    @property
    def names(self) -> Tuple[str, ...]:
        return tuple(d.name for d in self.dimensions)

    @property
    def integer_dimensions(self) -> Tuple[Dimension, ...]:
        return tuple(d for d in self.dimensions if d.kind == "integer")

    @property
    def integer_points(self) -> np.ndarray:
        """Every admissible value of the integer coordinates, ``[K, I]`` (``I`` = number of
        integer dimensions), in lexicographic order. The total bounds, when set, are applied here.
        A space without integer dimensions has one, the empty tuple: ``[1, 0]``."""
        ranges = [range(int(d.low), int(d.high) + 1) for d in self.integer_dimensions]
        tuples = list(itertools.product(*ranges))
        points = np.asarray(tuples, dtype=np.int32).reshape(len(tuples), len(ranges))
        if self.integer_total_cap is not None:
            points = points[points.sum(axis=1) <= self.integer_total_cap]
        if self.integer_total_floor is not None:
            points = points[points.sum(axis=1) >= self.integer_total_floor]
        return points

    @property
    def low(self) -> jax.Array:
        return jnp.asarray([d.low for d in self.dimensions], dtype=jnp.float32)

    @property
    def high(self) -> jax.Array:
        return jnp.asarray([d.high for d in self.dimensions], dtype=jnp.float32)

    @property
    def bounds(self) -> Bounds:
        """Ω's bounds as plain numbers, for whoever scales a context without holding the space."""
        return Bounds(low=tuple(float(d.low) for d in self.dimensions), high=tuple(float(d.high) for d in self.dimensions))

    def index(self, name: str) -> int:
        return self.names.index(name)

    # ---- conversion ------------------------------------------------------ #

    def encode(self, **values: float) -> Context:
        """Named values -> context vector. Every dimension must be given."""
        missing = set(self.names) - set(values)
        extra = set(values) - set(self.names)
        if missing or extra:
            raise ValueError(f"encode(): missing {sorted(missing)}, unexpected {sorted(extra)}")
        return jnp.asarray([float(values[name]) for name in self.names], dtype=jnp.float32)

    def decode(self, context: Context) -> Dict[str, float]:
        """Context vector -> named Python floats (integers rounded)."""
        out: Dict[str, float] = {}
        for i, dimension in enumerate(self.dimensions):
            value = float(context[i])
            out[dimension.name] = round(value) if dimension.kind == "integer" else value
        return out

    def component(self, contexts: Contexts, name: str) -> jax.Array:
        """Select one named coordinate from a batch: ``[N, D] -> [N]``."""
        return contexts[..., self.index(name)]

    def scale(self, contexts: Contexts) -> jax.Array:
        """Contexts mapped to [-1, 1] per dimension over Ω's box; see :func:`scale_to_bounds`."""
        return scale_to_bounds(contexts, self.bounds)

    # ---- sampling / checking (JAX, usable inside jit) --------------------- #

    def sample_uniform(self, key: jax.Array, n: int) -> Contexts:
        """``n`` contexts drawn uniformly from Ω.

        Continuous coordinates are uniform on their interval; the integer
        coordinates are one admissible tuple drawn uniformly from
        :attr:`integer_points`, so every tuple under the cap is equally likely
        (rounding a uniform float would give the endpoints half weight).
        """
        continuous_key, integer_key = jax.random.split(key)
        u = jax.random.uniform(continuous_key, (n, self.size), dtype=jnp.float32)
        contexts = self.low + u * (self.high - self.low)
        integer_indices = [i for i, d in enumerate(self.dimensions) if d.kind == "integer"]
        if not integer_indices:
            return contexts
        points = jnp.asarray(self.integer_points, dtype=jnp.float32)
        chosen = points[jax.random.randint(integer_key, (n,), 0, points.shape[0])]
        return contexts.at[:, jnp.asarray(integer_indices)].set(chosen)

    def contains(self, contexts: Contexts, atol: float = 1e-6) -> jax.Array:
        """Boolean ``[N]``: inside the box (with tolerance) and within the integer total bounds."""
        inside = jnp.all((contexts >= self.low - atol) & (contexts <= self.high + atol), axis=-1)
        if self.integer_total_cap is None and self.integer_total_floor is None:
            return inside
        integer_mask = jnp.asarray([d.kind == "integer" for d in self.dimensions])
        total = jnp.sum(jnp.where(integer_mask, contexts, 0.0), axis=-1)
        if self.integer_total_cap is not None:
            inside = inside & (total <= self.integer_total_cap + atol)
        if self.integer_total_floor is not None:
            inside = inside & (total >= self.integer_total_floor - atol)
        return inside

    # ---- convenience ----------------------------------------------------- #

    def describe(self) -> str:
        rows = [f"  {d.name:<24} [{d.low:g}, {d.high:g}] {d.kind}{' hidden' if d.hidden else ''}" + (f"  — {d.description}" if d.description else "")
                for d in self.dimensions]
        names = " + ".join(d.name for d in self.integer_dimensions)
        if self.integer_total_cap is not None:
            rows.append(f"  with {names} ≤ {self.integer_total_cap}")
        if self.integer_total_floor is not None:
            rows.append(f"  with {names} ≥ {self.integer_total_floor}")
        return "ContextSpace(\n" + "\n".join(rows) + "\n)"


def _box_within(inner: ContextSpace, outer: ContextSpace) -> bool:
    """Both box-shaped: every admissible integer tuple of ``inner`` is one of ``outer``'s, and
    every continuous interval of ``inner`` lies inside ``outer``'s."""
    if inner.names != outer.names:
        return False
    integer_inside = {tuple(p) for p in inner.integer_points.tolist()} <= {tuple(p) for p in outer.integer_points.tolist()}
    continuous_inside = all(
        o.low <= i.low and i.high <= o.high for i, o in zip(inner.dimensions, outer.dimensions) if i.kind == "continuous"
    )
    return integer_inside and continuous_inside


def _intervals_overlap(first: Dimension, second: Dimension) -> bool:
    """Two continuous intervals share more than an endpoint, or are the same fixed value."""
    if first.is_degenerate or second.is_degenerate:
        return max(first.low, second.low) <= min(first.high, second.high)
    return first.low < second.high and second.low < first.high


def _boxes_disjoint(first: ContextSpace, second: ContextSpace) -> bool:
    """Both box-shaped over the same dimensions: they share no admissible integer tuple, or some
    continuous interval of one does not overlap the other's."""
    if first.names != second.names:
        raise ValueError(f"cannot compare spaces over different dimensions: {first.names} and {second.names}")
    share_integer_tuples = bool({tuple(p) for p in first.integer_points.tolist()} & {tuple(p) for p in second.integer_points.tolist()})
    continuous_overlap = all(
        _intervals_overlap(a, b) for a, b in zip(first.dimensions, second.dimensions) if a.kind == "continuous"
    )
    return not (share_integer_tuples and continuous_overlap)


@dataclasses.dataclass(frozen=True)
class ContextUnion(ContextSpace):
    """A region made of several disjoint parts, each a box-shaped :class:`ContextSpace`: the
    training contexts on either side of a held-out band, or everything but a held-out corner.

    ``dimensions`` are those of the box the parts lie in, so everything that scales, bins or
    names a context sees the same box whichever region a run samples from. Sampling is
    uniform over the union: a part is drawn in proportion to its
    :attr:`~ContextSpace.measure`, then a context within it. Build one with :func:`union`.
    """

    parts: Tuple[ContextSpace, ...] = dataclasses.field(kw_only=True)

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.integer_total_cap is not None or self.integer_total_floor is not None:
            raise ValueError("a ContextUnion's integer totals are bounded by its parts, not on the union")
        if len(self.parts) < 2:
            raise ValueError(f"a ContextUnion has at least two parts, got {len(self.parts)}; one part is that ContextSpace itself")
        enclosing = ContextSpace(self.dimensions)
        for part in self.parts:
            if len(part.boxes) != 1:
                raise ValueError("the parts of a ContextUnion are box-shaped; flatten a union of unions")
            if not _box_within(part, enclosing):
                raise ValueError(f"a part of the union reaches outside its box\n{part.describe()}\n{enclosing.describe()}")
        for index, first in enumerate(self.parts):
            for second in self.parts[index + 1:]:
                if not _boxes_disjoint(first, second):
                    raise ValueError(f"two parts of the union overlap\n{first.describe()}\n{second.describe()}")
        fixed = {tuple(d.is_degenerate for d in part.dimensions) for part in self.parts}
        if len(fixed) != 1:
            raise ValueError("the parts of a ContextUnion must fix the same dimensions; a fixed value has no length to weigh against an interval")

    @property
    def boxes(self) -> Tuple[ContextSpace, ...]:
        return self.parts

    @property
    def measure(self) -> float:
        return float(sum(part.measure for part in self.parts))

    @property
    def integer_points(self) -> np.ndarray:
        """Every integer tuple admissible in some part, in lexicographic order."""
        return np.unique(np.concatenate([part.integer_points for part in self.parts]), axis=0)

    def sample_uniform(self, key: jax.Array, n: int) -> Contexts:
        """``n`` contexts drawn uniformly from the union."""
        part_key, *draw_keys = jax.random.split(key, 1 + len(self.parts))
        draws = jnp.stack([part.sample_uniform(draw_key, n) for part, draw_key in zip(self.parts, draw_keys)])  # [P, n, D]
        weights = jnp.asarray([part.measure for part in self.parts], dtype=jnp.float32)
        chosen = jax.random.categorical(part_key, jnp.log(weights), shape=(n,))
        return draws[chosen, jnp.arange(n)]

    def contains(self, contexts: Contexts, atol: float = 1e-6) -> jax.Array:
        return jnp.any(jnp.stack([part.contains(contexts, atol) for part in self.parts]), axis=0)

    def describe(self) -> str:
        return "ContextUnion of\n" + "\nand\n".join(part.describe() for part in self.parts)


def union(enclosing: ContextSpace, *parts: ContextSpace) -> ContextUnion:
    """The region of the box ``enclosing`` covered by ``parts``, which are disjoint and lie within it."""
    return ContextUnion(dimensions=enclosing.dimensions, parts=tuple(parts))


def box(**bounds: Tuple[float, float]) -> ContextSpace:
    """Shorthand: ``box(threshold=(0.5, 2.6))``."""
    return ContextSpace(tuple(Dimension(name, low, high) for name, (low, high) in bounds.items()))
