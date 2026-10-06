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


@dataclasses.dataclass(frozen=True)
class Dimension:
    """One coordinate of a context space."""

    name: str
    low: float
    high: float
    kind: DimensionKind = "continuous"
    description: str = ""

    def __post_init__(self) -> None:
        if not self.low <= self.high:
            raise ValueError(f"Dimension {self.name!r}: low ({self.low}) > high ({self.high})")

    @property
    def is_degenerate(self) -> bool:
        return self.low == self.high


@dataclasses.dataclass(frozen=True)
class ContextSpace:
    """A box-shaped Ω with named coordinates, optionally capped in its integer total.

    ``integer_total_cap``: if set, only points whose integer coordinates sum to at
    most this value belong to Ω. Instances are immutable and hashable, so they can
    be static arguments to jitted functions.
    """

    dimensions: Tuple[Dimension, ...]
    integer_total_cap: Optional[int] = None

    def __post_init__(self) -> None:
        names = [d.name for d in self.dimensions]
        if len(set(names)) != len(names):
            raise ValueError(f"Duplicate dimension names: {names}")
        if not self.dimensions:
            raise ValueError("A context space needs at least one dimension")
        if self.integer_total_cap is not None:
            if not self.integer_dimensions:
                raise ValueError("integer_total_cap given but Ω has no integer dimension")
            least = sum(int(d.low) for d in self.integer_dimensions)
            if self.integer_total_cap < least:
                raise ValueError(f"integer_total_cap {self.integer_total_cap} is below the smallest possible total {least}")

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
        integer dimensions), in lexicographic order. The cap, when set, is applied here."""
        ranges = [range(int(d.low), int(d.high) + 1) for d in self.integer_dimensions]
        points = np.asarray(list(itertools.product(*ranges)), dtype=np.int32).reshape(-1, len(ranges))
        if self.integer_total_cap is not None:
            points = points[points.sum(axis=1) <= self.integer_total_cap]
        return points

    @property
    def low(self) -> jax.Array:
        return jnp.asarray([d.low for d in self.dimensions], dtype=jnp.float32)

    @property
    def high(self) -> jax.Array:
        return jnp.asarray([d.high for d in self.dimensions], dtype=jnp.float32)

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
        """Boolean ``[N]``: inside the box (with tolerance) and under the cap."""
        inside = jnp.all((contexts >= self.low - atol) & (contexts <= self.high + atol), axis=-1)
        if self.integer_total_cap is None:
            return inside
        integer_mask = jnp.asarray([d.kind == "integer" for d in self.dimensions])
        total = jnp.sum(jnp.where(integer_mask, contexts, 0.0), axis=-1)
        return inside & (total <= self.integer_total_cap + atol)

    # ---- convenience ----------------------------------------------------- #

    def describe(self) -> str:
        rows = [f"  {d.name:<24} [{d.low:g}, {d.high:g}] {d.kind}" + (f"  — {d.description}" if d.description else "")
                for d in self.dimensions]
        if self.integer_total_cap is not None:
            names = " + ".join(d.name for d in self.integer_dimensions)
            rows.append(f"  with {names} ≤ {self.integer_total_cap}")
        return "ContextSpace(\n" + "\n".join(rows) + "\n)"


def box(**bounds: Tuple[float, float]) -> ContextSpace:
    """Shorthand: ``box(threshold=(0.5, 2.6))``."""
    return ContextSpace(tuple(Dimension(name, low, high) for name, (low, high) in bounds.items()))
