"""The process-wide performance tracker.

See the package docstring for the design rationale.
"""
from __future__ import annotations

import contextlib
import dataclasses
import os
import platform
import statistics
import subprocess
import sys
import time
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence

import jax

from training.performance.aot import ProgramStatistics, ahead_of_time_compile
from training.performance.sinks import JsonSink, WandbSink
from training.performance.trace import find_trace_files, trace_window

# Per-round scalars are reported through this: (environment_steps, {"performance/<name>": value}).
ReportMetrics = Callable[[int, Dict[str, float]], None]
METRIC_PREFIX = "performance/"

# --------------------------------------------------------------------------- #
# Null object
# --------------------------------------------------------------------------- #


class NullTracker:
    """Does nothing.  Returned by :func:`get_tracker` when no tracker is installed.

    Every public method of :class:`PerformanceTracker` exists here with the same
    signature so that call sites never need to check which one they hold.
    """

    enabled = False

    @contextlib.contextmanager
    def phase(self, name: str) -> Iterator[None]:
        yield

    @contextlib.contextmanager
    def epoch(self, index: int, environment_steps: int) -> Iterator[None]:
        yield

    @contextlib.contextmanager
    def scope(self, name: str) -> Iterator[None]:
        # jax.named_scope is metadata-only and free; keep it even when disabled so
        # ad-hoc `jax.profiler.trace` calls elsewhere still get readable names.
        import jax

        with jax.named_scope(name):
            yield

    def compile_epoch_function(self, jitted_function: Callable, *arguments: Any) -> Callable:
        return jitted_function

    def record(self, **scalars: float) -> None:
        pass

    def note(self, message: str) -> None:
        pass

    def finish(self) -> Dict[str, Any]:
        return {}


# --------------------------------------------------------------------------- #
# Real tracker
# --------------------------------------------------------------------------- #


@dataclasses.dataclass
class EpochRecord:
    index: int
    environment_steps: int
    wall_seconds: float
    steps_per_second: float
    compiles_during_epoch: int
    compile_seconds_during_epoch: float
    traced: bool

    # An epoch is steady unless a *substantial* program was compiled during it.
    # JAX lazily compiles ~10 tiny helper programs per epoch (copy, squeeze,
    # broadcast_in_dim, ... from the host-side glue around the epoch call), each
    # ~10 ms; those must not disqualify an epoch. A recompiled epoch program
    # costs seconds. So: total compile time in the epoch must be small in absolute
    # terms AND not a large fraction of a (possibly very short) epoch.
    STEADY_MAX_COMPILE_SECONDS = 0.5
    STEADY_MAX_COMPILE_FRACTION = 0.10

    @property
    def is_steady(self) -> bool:
        """Usable for throughput statistics: not the compile epoch, not traced,
        and no substantial compilation happened during it."""
        if self.index == 0 or self.traced or self.wall_seconds <= 0:
            return False
        return (
            self.compile_seconds_during_epoch <= self.STEADY_MAX_COMPILE_SECONDS
            and self.compile_seconds_during_epoch <= self.STEADY_MAX_COMPILE_FRACTION * self.wall_seconds
        )


@dataclasses.dataclass
class CompileEvent:
    seconds: float
    wall_time: float  # time.time() at completion, to attribute to phases/epochs


class PerformanceTracker:
    """Collects timing, compilation and program statistics for one training run.

    Instances are normally created through :func:`install`, not directly.
    """

    enabled = True

    def __init__(
        self,
        run_name: str,
        output_dir: str,
        report_metrics: ReportMetrics,
        *,
        profile_epochs: Sequence[int] = (),
        log_compiles: bool = False,
        verbose: bool = True,
    ):
        self.run_name = run_name
        self.output_dir = os.path.join(output_dir, run_name)
        self.report_metrics = report_metrics
        self.profile_epochs = set(int(i) for i in profile_epochs)
        self.verbose = verbose

        self._started_at = time.time()
        self._epochs: List[EpochRecord] = []
        self._phases: Dict[str, float] = {}
        self._compile_events: List[CompileEvent] = []
        # One entry per compiled epoch program.  A plain run has one; a
        # curriculum has one per stage.
        self._program_statistics: List[ProgramStatistics] = []
        self._epoch_offset = 0  # makes epoch indices unique across train() calls
        self._extra_scalars: Dict[str, float] = {}
        self._notes: List[str] = []
        self._finished = False

        os.makedirs(self.output_dir, exist_ok=True)
        self._json_sink = JsonSink(os.path.join(self.output_dir, "performance.json"))
        self._wandb_sink = WandbSink()

        self._install_compile_listener()
        if log_compiles:
            import jax

            jax.config.update("jax_log_compiles", True)

        self.metadata = collect_run_metadata()

    # ---- passive compile observation ------------------------------------- #

    def _install_compile_listener(self) -> None:
        """Count and time every XLA backend compile via JAX's monitoring hooks.

        This is the mechanism JAX itself uses for its metrics; it has no effect
        on the compiled programs and costs one Python call per compilation.
        """
        from jax._src import monitoring
        from jax._src.dispatch import BACKEND_COMPILE_EVENT

        def listener(event: str, duration: float, **_unused: Any) -> None:
            if event == BACKEND_COMPILE_EVENT:
                self._compile_events.append(CompileEvent(seconds=duration, wall_time=time.time()))

        monitoring.register_event_duration_secs_listener(listener)
        self._compile_listener = listener

    def _compiles_between(self, start: float, end: float) -> List[CompileEvent]:
        return [event for event in self._compile_events if start <= event.wall_time <= end]

    # ---- public API ------------------------------------------------------ #

    @contextlib.contextmanager
    def phase(self, name: str) -> Iterator[None]:
        """Time a one-off, host-side phase (environment reset, initial eval, ...)."""
        start = time.time()
        try:
            yield
        finally:
            elapsed = time.time() - start
            self._phases[name] = self._phases.get(name, 0.0) + elapsed
            compiles = self._compiles_between(start, time.time())
            compile_seconds = sum(event.seconds for event in compiles)
            self._say(
                f"phase {name}: {elapsed:.1f}s ({len(compiles)} compiles, {compile_seconds:.1f}s compiling)"
            )

    @contextlib.contextmanager
    def epoch(self, index: int, environment_steps: int) -> Iterator[None]:
        """Time one training epoch; optionally record a profiler trace for it.

        ``index`` is local to the current ``train()`` call (0 = first epoch of
        that call).  When one tracker spans several calls (curriculum stages)
        the stored ``index`` is offset so it stays unique; ``profile_epochs`` is
        matched against the global index.

        The body must block on its results (the trainer already calls
        ``block_until_ready``), otherwise both the wall-clock and the trace end
        before the device does.
        """
        if index == 0 and self._epochs:
            # A new train() call started: offset indices past the previous ones.
            self._epoch_offset = self._epochs[-1].index + 1
        global_index = self._epoch_offset + index
        traced = global_index in self.profile_epochs
        trace_dir = os.path.join(self.output_dir, "trace")
        start = time.time()
        if traced:
            self._say(f"epoch {global_index}: recording jax.profiler trace -> {trace_dir}")
            context: Any = trace_window(trace_dir)
        else:
            context = contextlib.nullcontext()
        with context:
            # XProf's Overview Page / step-time breakdown are computed per "step",
            # delimited by StepTraceAnnotation; without one they read 0. An epoch
            # is the smallest unit we can mark from Python (training_step lives
            # inside lax.scan), so one XProf step == one epoch.
            with jax.profiler.StepTraceAnnotation("epoch", step_num=global_index):
                yield
        wall = time.time() - start
        compiles = self._compiles_between(start, time.time())
        compile_seconds = sum(event.seconds for event in compiles)
        record = EpochRecord(
            index=global_index,
            environment_steps=environment_steps,
            wall_seconds=wall,
            steps_per_second=environment_steps / wall if wall > 0 else float("nan"),
            compiles_during_epoch=len(compiles),
            compile_seconds_during_epoch=compile_seconds,
            traced=traced,
        )
        self._epochs.append(record)
        scalars = {
            "epoch_wall_seconds": wall,
            "epoch_steps_per_second": record.steps_per_second,
            "epoch_compiles": len(compiles),
            "epoch_compile_seconds": compile_seconds,
        }
        total_steps = self._total_environment_steps()
        self._json_sink.log_scalars(scalars, environment_steps=total_steps)
        self.report_metrics(total_steps, {METRIC_PREFIX + key: value for key, value in scalars.items()})
        if index > 0 and not record.is_steady:
            self._say(
                f"epoch {global_index}: {len(compiles)} compile(s) took {compile_seconds:.1f}s "
                f"of {wall:.1f}s -> not steady state. Did shapes or pytree structure change?"
            )

    @contextlib.contextmanager
    def scope(self, name: str) -> Iterator[None]:
        """Name a region inside jitted code so profiler tables roll up to it."""
        import jax

        with jax.named_scope(name):
            yield

    def compile_epoch_function(self, jitted_function: Callable, *arguments: Any) -> Callable:
        """Compile the epoch program ahead of time and record its statistics.

        Returns a callable with the same signature.  Call this once, right after
        ``jax.jit`` / ``jax.pmap``, with the arguments of the first epoch.
        """
        self._say("compiling epoch program ahead of time ...")
        start = time.time()
        compiled, program_statistics = ahead_of_time_compile(jitted_function, *arguments)
        self._program_statistics.append(program_statistics)
        self._phases["epoch_compile"] = self._phases.get("epoch_compile", 0.0) + (time.time() - start)
        self._say(
            f"epoch program: lower {program_statistics.lower_time_seconds:.1f}s, "
            f"compile {program_statistics.compile_time_seconds:.1f}s, "
            f"flops/epoch {program_statistics.flops or float('nan'):.3e}, "
            f"bytes/epoch {program_statistics.bytes_accessed or float('nan'):.3e}, "
            f"peak memory {_format_bytes(program_statistics.peak_memory_bytes)}"
        )
        return compiled

    def record(self, **scalars: float) -> None:
        """Attach arbitrary extra scalars to the run summary."""
        self._extra_scalars.update({key: float(value) for key, value in scalars.items()})

    def note(self, message: str) -> None:
        """Free-text note stored in the JSON (replaces ad-hoc debug prints)."""
        self._notes.append(f"{time.time() - self._started_at:8.1f}s  {message}")
        self._say(message)

    def finish(self) -> Dict[str, Any]:
        """Aggregate, write all sinks, and return the summary dictionary."""
        if self._finished:
            return {}
        self._finished = True

        steady = [e for e in self._epochs if e.is_steady]
        steady_sps = [e.steps_per_second for e in steady]
        summary: Dict[str, Any] = {
            "run_name": self.run_name,
            "total_wall_seconds": time.time() - self._started_at,
            "num_epochs": len(self._epochs),
            "num_steady_epochs": len(steady),
            "total_compiles": len(self._compile_events),
            "total_compile_seconds": sum(e.seconds for e in self._compile_events),
            "steady_steps_per_second_median": statistics.median(steady_sps) if steady_sps else None,
            "steady_steps_per_second_iqr": _interquartile_range(steady_sps),
            "steady_epoch_wall_seconds_median": (
                statistics.median(e.wall_seconds for e in steady) if steady else None
            ),
            "first_epoch_wall_seconds": self._epochs[0].wall_seconds if self._epochs else None,
            **{f"phase_{name}_seconds": seconds for name, seconds in self._phases.items()},
            **self._extra_scalars,
        }
        if self._program_statistics:
            # Summary carries the first program (the common single-stage case);
            # the JSON keeps all of them under "programs".
            summary["num_programs"] = len(self._program_statistics)
            summary.update(
                {f"program_{key}": value for key, value in self._program_statistics[0].as_dict().items()}
            )

        trace_dir = os.path.join(self.output_dir, "trace")
        trace_files = find_trace_files(trace_dir)

        # Trace -> W&B artifact, so it can be pulled to a laptop from a cluster run
        # with `python -m training.performance.fetch_traces <run>`.
        trace_artifact = None
        if trace_files:
            sessions_dir = os.path.join(trace_dir, "plugins", "profile")
            trace_artifact = self._wandb_sink.log_trace_artifact(
                sessions_dir,
                self.run_name,
                metadata={
                    "traced_epochs": sorted(e.index for e in self._epochs if e.traced),
                    "num_files": len(trace_files),
                    "bytes": sum(os.path.getsize(f) for f in trace_files),
                    **{k: self.metadata.get(k) for k in ("git_sha_short", "device_kind", "jax")},
                },
            )
            summary["trace_artifact"] = trace_artifact

        json_sink = self._json_sink
        json_sink.set_summary(summary)
        json_sink.set_section("metadata", self.metadata)
        json_sink.set_section("epochs", [dataclasses.asdict(e) for e in self._epochs])
        json_sink.set_section("programs", [p.as_dict() for p in self._program_statistics])
        json_sink.set_section("compile_events", [dataclasses.asdict(e) for e in self._compile_events])
        json_sink.set_section("notes", self._notes)
        json_sink.set_section("trace_files", trace_files)
        json_path = json_sink.flush()

        self._wandb_sink.set_summary(summary)
        self._wandb_sink.save_file(json_path)

        self._say(self._format_summary(summary, trace_files, trace_artifact))
        return summary

    # ---- helpers --------------------------------------------------------- #

    def _total_environment_steps(self) -> int:
        return sum(e.environment_steps for e in self._epochs)

    def _say(self, message: str) -> None:
        if self.verbose:
            print(f"[performance] {message}", flush=True)

    def _format_summary(
        self, summary: Dict[str, Any], trace_files: List[str], trace_artifact: Optional[str]
    ) -> str:
        median = summary.get("steady_steps_per_second_median")
        lines = [
            "summary",
            f"  steady-state SPS      : {median:,.0f}" if median else "  steady-state SPS      : n/a",
            f"  steady epochs         : {summary['num_steady_epochs']} / {summary['num_epochs']}",
            f"  compiles (total)      : {summary['total_compiles']} ({summary['total_compile_seconds']:.1f}s)",
            f"  total wall            : {summary['total_wall_seconds']:.1f}s",
            f"  json                  : {os.path.join(self.output_dir, 'performance.json')}",
        ]
        if trace_files:
            lines.append(f"  trace (local)         : xprof --logdir {os.path.join(self.output_dir, 'trace')}")
        if trace_artifact:
            lines.append(f"  trace (W&B artifact)  : {trace_artifact}")
        return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Module-level installation
# --------------------------------------------------------------------------- #

_TRACKER: Any = NullTracker()


def install(
    run_name: str,
    output_dir: str,
    report_metrics: ReportMetrics,
    *,
    profile_epochs: Sequence[int] = (),
    log_compiles: bool = False,
    verbose: bool = True,
) -> PerformanceTracker:
    """Create the process-wide tracker.  Call once per run, before training.

    ``report_metrics`` receives the per-round scalars (``performance/*``) at
    the run's environment-step count: pass the same progress callback the
    trainer logs with, so every metric takes one path to W&B. Call *after*
    ``wandb.init`` so the summary and trace artifact land in that run.
    """
    global _TRACKER
    if isinstance(_TRACKER, PerformanceTracker) and not _TRACKER._finished:
        _TRACKER.finish()
    _TRACKER = PerformanceTracker(
        run_name,
        output_dir,
        report_metrics,
        profile_epochs=profile_epochs,
        log_compiles=log_compiles,
        verbose=verbose,
    )
    return _TRACKER


def uninstall() -> None:
    """Finish and remove the current tracker (restores the null tracker)."""
    global _TRACKER
    if isinstance(_TRACKER, PerformanceTracker):
        _TRACKER.finish()
    _TRACKER = NullTracker()


def get_tracker() -> Any:
    """Return the installed :class:`PerformanceTracker`, or a :class:`NullTracker`."""
    return _TRACKER


# --------------------------------------------------------------------------- #
# Metadata
# --------------------------------------------------------------------------- #


def collect_run_metadata() -> Dict[str, Any]:
    """Everything needed to know whether two runs are comparable."""
    metadata: Dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "hostname": platform.node(),
        "xla_flags": os.environ.get("XLA_FLAGS", ""),
        "jax_compilation_cache_dir": os.environ.get("JAX_COMPILATION_CACHE_DIR", ""),
    }
    try:
        import jax

        metadata["jax"] = jax.__version__
        metadata["jax_backend"] = jax.default_backend()
        metadata["devices"] = [str(d) for d in jax.devices()]
        try:
            metadata["device_kind"] = jax.devices()[0].device_kind
        except Exception:
            pass
    except Exception:
        pass
    for module_name in ("mujoco", "flax", "optax"):
        try:
            module = __import__(module_name)
            metadata[module_name] = getattr(module, "__version__", "?")
        except Exception:
            pass
    metadata.update(_git_metadata())
    return metadata


def _git_metadata() -> Dict[str, Any]:
    def run(*args: str) -> Optional[str]:
        try:
            return subprocess.check_output(["git", *args], stderr=subprocess.DEVNULL, text=True).strip()
        except Exception:
            return None

    sha = run("rev-parse", "HEAD")
    if sha is None:
        return {}
    dirty = run("status", "--porcelain")
    return {
        "git_sha": sha,
        "git_sha_short": sha[:8],
        "git_branch": run("rev-parse", "--abbrev-ref", "HEAD"),
        "git_dirty": bool(dirty),
    }


# --------------------------------------------------------------------------- #
# Formatting
# --------------------------------------------------------------------------- #


def _interquartile_range(values: Sequence[float]) -> Optional[float]:
    if len(values) < 4:
        return None
    ordered = sorted(values)
    quartiles = statistics.quantiles(ordered, n=4)
    return quartiles[2] - quartiles[0]


def _format_bytes(count: Optional[int]) -> str:
    if count is None:
        return "n/a"
    value = float(count)
    for unit in ("B", "KiB", "MiB", "GiB"):
        if value < 1024 or unit == "GiB":
            return f"{value:.1f} {unit}"
        value /= 1024
    return f"{value:.1f} GiB"
