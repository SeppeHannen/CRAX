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

Ω is a box, optionally cut by one constraint: a cap on the **sum of the integer
dimensions**. A hazard suite's integer dimensions are per-kind active counts,
and the arena can hold only so many hazards at once; the cap states that
limit so that "uniform over Ω" never asks for a layout that does not fit.
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
    """

    name: str
    low: float
    high: float
    kind: DimensionKind = "continuous"
    description: str = ""
    harder_when: DifficultyDirection = "higher"

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
        integer dimensions), in lexicographic order. The total bounds, when set, are applied here."""
        ranges = [range(int(d.low), int(d.high) + 1) for d in self.integer_dimensions]
        points = np.asarray(list(itertools.product(*ranges)), dtype=np.int32).reshape(-1, len(ranges))
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
        rows = [f"  {d.name:<24} [{d.low:g}, {d.high:g}] {d.kind}" + (f"  — {d.description}" if d.description else "")
                for d in self.dimensions]
        names = " + ".join(d.name for d in self.integer_dimensions)
        if self.integer_total_cap is not None:
            rows.append(f"  with {names} ≤ {self.integer_total_cap}")
        if self.integer_total_floor is not None:
            rows.append(f"  with {names} ≥ {self.integer_total_floor}")
        return "ContextSpace(\n" + "\n".join(rows) + "\n)"


def box(**bounds: Tuple[float, float]) -> ContextSpace:
    """Shorthand: ``box(threshold=(0.5, 2.6))``."""
    return ContextSpace(tuple(Dimension(name, low, high) for name, (low, high) in bounds.items()))
