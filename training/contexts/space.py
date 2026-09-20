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
"""
from __future__ import annotations

import dataclasses
from typing import Dict, Literal, Mapping, Sequence, Tuple

import jax
import jax.numpy as jnp

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
    """A box-shaped Ω with named coordinates.

    Instances are immutable and hashable, so they can be static arguments to
    jitted functions.
    """

    dimensions: Tuple[Dimension, ...]

    def __post_init__(self) -> None:
        names = [d.name for d in self.dimensions]
        if len(set(names)) != len(names):
            raise ValueError(f"Duplicate dimension names: {names}")
        if not self.dimensions:
            raise ValueError("A context space needs at least one dimension")

    # ---- introspection --------------------------------------------------- #

    @property
    def size(self) -> int:
        return len(self.dimensions)

    @property
    def names(self) -> Tuple[str, ...]:
        return tuple(d.name for d in self.dimensions)

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
        """``n`` contexts drawn uniformly from the box (integers rounded)."""
        u = jax.random.uniform(key, (n, self.size), dtype=jnp.float32)
        contexts = self.low + u * (self.high - self.low)
        return self._round_integers(contexts)

    def clip(self, contexts: Contexts) -> Contexts:
        return self._round_integers(jnp.clip(contexts, self.low, self.high))

    def contains(self, contexts: Contexts, atol: float = 1e-6) -> jax.Array:
        """Boolean ``[N]``: inside the box (with tolerance)."""
        return jnp.all((contexts >= self.low - atol) & (contexts <= self.high + atol), axis=-1)

    def _round_integers(self, contexts: Contexts) -> Contexts:
        integer_mask = jnp.asarray([d.kind == "integer" for d in self.dimensions])
        return jnp.where(integer_mask, jnp.round(contexts), contexts)

    # ---- convenience ----------------------------------------------------- #

    def describe(self) -> str:
        rows = [f"  {d.name:<24} [{d.low:g}, {d.high:g}] {d.kind}" + (f"  — {d.description}" if d.description else "")
                for d in self.dimensions]
        return "ContextSpace(\n" + "\n".join(rows) + "\n)"


def box(**bounds: Tuple[float, float]) -> ContextSpace:
    """Shorthand: ``box(threshold=(0.5, 2.6))``."""
    return ContextSpace(tuple(Dimension(name, low, high) for name, (low, high) in bounds.items()))
