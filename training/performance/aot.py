"""Ahead-of-time compilation helpers.

``jax.jit`` compiles on the first call, which mixes compile time into the
first epoch's wall-clock.  Compiling explicitly beforehand gives clean
compile-vs-run timings and access to the compiler's cost and memory analyses,
which are *deterministic* for a given program and therefore a noise-free way
to detect that a code change altered the amount of work per epoch.

JAX documents ``cost_analysis()`` / ``memory_analysis()`` as best-effort and
not stable across versions.  We only ever compare numbers produced by the same
pinned environment, so that is acceptable.
"""
from __future__ import annotations

import dataclasses
import time
from typing import Any, Callable, Dict, Optional


@dataclasses.dataclass
class ProgramStatistics:
    """Compiler-reported statistics of a compiled XLA program."""

    lower_time_seconds: float
    compile_time_seconds: float
    flops: Optional[float] = None
    bytes_accessed: Optional[float] = None
    transcendentals: Optional[float] = None
    argument_size_bytes: Optional[int] = None
    output_size_bytes: Optional[int] = None
    temp_size_bytes: Optional[int] = None
    generated_code_size_bytes: Optional[int] = None

    @property
    def peak_memory_bytes(self) -> Optional[int]:
        parts = [self.argument_size_bytes, self.output_size_bytes, self.temp_size_bytes]
        if any(p is None for p in parts):
            return None
        return sum(parts)  # type: ignore[arg-type]

    def as_dict(self) -> Dict[str, Any]:
        data = dataclasses.asdict(self)
        data["peak_memory_bytes"] = self.peak_memory_bytes
        return data


def _first_present(mapping: Any, *keys: str) -> Optional[float]:
    """XLA cost analysis is a flat dict but the exact key spelling varies."""
    if not mapping:
        return None
    if isinstance(mapping, (list, tuple)):  # some backends return a per-device list
        mapping = mapping[0] if mapping else None
        if mapping is None:
            return None
    for key in keys:
        if key in mapping:
            value = mapping[key]
            try:
                return float(value)
            except (TypeError, ValueError):
                return None
    return None


def _memory_field(analysis: Any, name: str) -> Optional[int]:
    if analysis is None:
        return None
    value = getattr(analysis, name, None)
    if value is None and isinstance(analysis, dict):
        value = analysis.get(name)
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def ahead_of_time_compile(
    jitted_function: Callable[..., Any],
    *arguments: Any,
    **keyword_arguments: Any,
):
    """Lower and compile ``jitted_function`` for the given example arguments.

    Args:
      jitted_function: the result of ``jax.jit(...)`` (or ``jax.pmap``; pmap
        also exposes ``.lower``).
      *arguments, **keyword_arguments: concrete arrays/pytrees with the shapes
        and dtypes the function will be called with.  Values are not used.

    Returns:
      ``(compiled, statistics)`` where ``compiled`` is a ``jax.stages.Compiled``
      callable that accepts the same arguments, and ``statistics`` is a
      :class:`ProgramStatistics`.
    """
    start = time.perf_counter()
    lowered = jitted_function.lower(*arguments, **keyword_arguments)
    lowered_at = time.perf_counter()
    compiled = lowered.compile()
    compiled_at = time.perf_counter()

    cost = None
    memory = None
    try:
        cost = compiled.cost_analysis()
    except Exception:  # pragma: no cover - backend dependent
        pass
    try:
        memory = compiled.memory_analysis()
    except Exception:  # pragma: no cover - backend dependent
        pass

    statistics = ProgramStatistics(
        lower_time_seconds=lowered_at - start,
        compile_time_seconds=compiled_at - lowered_at,
        flops=_first_present(cost, "flops"),
        bytes_accessed=_first_present(cost, "bytes accessed", "bytes_accessed"),
        transcendentals=_first_present(cost, "transcendentals"),
        argument_size_bytes=_memory_field(memory, "argument_size_in_bytes"),
        output_size_bytes=_memory_field(memory, "output_size_in_bytes"),
        temp_size_bytes=_memory_field(memory, "temp_size_in_bytes"),
        generated_code_size_bytes=_memory_field(memory, "generated_code_size_in_bytes"),
    )
    return compiled, statistics
