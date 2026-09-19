"""Thin wrapper around ``jax.profiler`` tracing.

A trace captures host events, XLA operations and CUDA kernels (through CUPTI)
for everything executed inside the window.  Overhead is paid only inside the
window, so we trace one or two steady-state epochs rather than a whole run.

Traces are written in the layout XProf expects::

    <output_dir>/plugins/profile/<session_name>/*.xplane.pb

so that ``xprof --logdir <output_dir>`` (web UI) or
``xprof get_overview <output_dir>`` (JSON on the terminal) work directly.
"""
from __future__ import annotations

import contextlib
import os
from typing import Iterator, Optional


def _profiler_options(host_tracer_level: int, python_tracer_level: int):
    import jax

    options_class = getattr(jax.profiler, "ProfileOptions", None)
    if options_class is None:  # older JAX
        return None
    options = options_class()
    options.host_tracer_level = host_tracer_level
    options.python_tracer_level = python_tracer_level
    # CRAX epochs are launch-bound: ~2 M kernel launches per 1.3 M-step epoch at
    # 2048 envs. XProf's CUPTI buffers default to 2*1024*1024 events, which a
    # slightly larger epoch would silently overflow (dropped events). Raise them.
    try:
        options.advanced_configuration = {
            "gpu_max_activity_api_events": 32 * 1024 * 1024,
            "gpu_max_callback_api_events": 32 * 1024 * 1024,
            "gpu_max_annotation_strings": 4 * 1024 * 1024,
        }
    except Exception:  # option names vary across JAX/XProf versions; not fatal
        pass
    return options


@contextlib.contextmanager
def trace_window(
    output_dir: str,
    *,
    host_tracer_level: int = 1,
    python_tracer_level: int = 0,
    create_perfetto_link: bool = False,
) -> Iterator[str]:
    """Context manager that records a ``jax.profiler`` trace into ``output_dir``.

    Args:
      output_dir: directory to write the trace to. Created if missing.  JAX
        adds the ``plugins/profile/<timestamp>/`` structure itself.
      host_tracer_level: 0 disables host tracing, 1 records only user
        annotations (``jax.named_scope`` / ``TraceAnnotation``), 2 adds
        expensive XLA ops (JAX default), 3 adds cheap ops too.  We default to
        1 to keep traces small; the device timeline is unaffected.
      python_tracer_level: 1 records every Python function call.  Expensive
        and useless for jitted code; default 0.
      create_perfetto_link: if True, JAX prints a ui.perfetto.dev link and
        *blocks* until it is opened.  Off by default; use XProf instead.

    Yields:
      The output directory.
    """
    import jax

    os.makedirs(output_dir, exist_ok=True)
    options = _profiler_options(host_tracer_level, python_tracer_level)
    keyword_arguments = {"create_perfetto_link": create_perfetto_link}
    if options is not None:
        keyword_arguments["profiler_options"] = options
    jax.profiler.start_trace(output_dir, **keyword_arguments)
    try:
        yield output_dir
    finally:
        jax.profiler.stop_trace()


def find_trace_files(output_dir: str) -> list[str]:
    """Return all ``*.xplane.pb`` files under ``output_dir``."""
    found = []
    for root, _dirs, files in os.walk(output_dir):
        for name in files:
            if name.endswith(".xplane.pb"):
                found.append(os.path.join(root, name))
    return sorted(found)
