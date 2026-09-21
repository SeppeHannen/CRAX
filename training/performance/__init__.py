"""Performance measurement for CRAX training.

This package answers two questions about a training run without slowing it
down:

1. *Is it faster or slower than before?*  Per-epoch wall-clock and
   steps-per-second, compile time separated from run time, and deterministic
   program statistics (FLOPs, bytes, peak memory) from the XLA compiler.
2. *Where does the time go?*  An optional ``jax.profiler`` trace of a few
   steady-state epochs, viewable with XProf.

Design
------
* A single process-wide :class:`PerformanceTracker` is installed once by the
  entry point (``install(...)``).  Library code fetches it with
  :func:`get_tracker` and calls it unconditionally.  When nothing has been
  installed, :func:`get_tracker` returns a :class:`NullTracker` whose methods
  are no-ops, so the trainer carries no ``if profiling:`` branches.
* Compilation is observed passively through JAX's monitoring hooks; no call
  site has to be wrapped to count or time recompiles.
* Weights & Biases is the system of record.  Per-epoch scalars
  (``performance/*``) are reported through the same progress callback the
  trainer logs with, so they take the one path every metric takes (and pass
  the dashboard registry); the summary goes to ``run.summary`` and the
  profiler trace is uploaded as an ``xprof-trace`` artifact so it can be
  fetched from a cluster run with ``python -m training.performance.fetch_traces
  <run>``.  A JSON copy is also written locally and attached to the run's
  files.

Typical use::

    from training import performance

    # entry point, once per run, after wandb.init
    performance.install(run_name="...", output_dir="runs/performance",
                        report_metrics=progress_fn, profile_epochs=[3, 4])

    # library code, anywhere
    tracker = performance.get_tracker()
    with tracker.phase("environment_reset"):
        ...
    with tracker.epoch(index, environment_steps=n):
        ...
    tracker.finish()
"""

from training.performance.tracker import (
    NullTracker,
    PerformanceTracker,
    get_tracker,
    install,
    uninstall,
)
from training.performance.aot import ahead_of_time_compile, ProgramStatistics
from training.performance.trace import trace_window

__all__ = [
    "NullTracker",
    "PerformanceTracker",
    "ProgramStatistics",
    "ahead_of_time_compile",
    "get_tracker",
    "install",
    "trace_window",
    "uninstall",
]
