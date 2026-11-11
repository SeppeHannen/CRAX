# Profiling CRAX training: options review

*Author: Copilot for Giuseppe Hannen — 2026-09-19. Status: **proposal, for review**.*

Goal: be able to say, with confidence, "this change made training X% slower/faster,
and here is *where* the time went" — without the measurement itself slowing training
down, and without building a bespoke tool from scratch.

---

## 0. TL;DR recommendation

| Layer | Tool | Custom code we write | Overhead during normal training | What you look at |
|---|---|---|---|---|
| **A. Always-on regression signal** | Already in CRAX (`training/sps`, `training/walltime` per epoch) + W&B | ~0 — just make sure both are logged to W&B and add `compile_time_s` as a summary | none | W&B chart: `training/sps` vs step, runs overlaid; run table sorted by `median_sps` |
| **B. Deterministic "did the program change?" check** | JAX AOT: `compiled.cost_analysis()`, `memory_analysis()`, `jax_log_compiles` | ~60 lines: one helper that logs these numbers to W&B summary | none (computed once, at compile time) | W&B run table columns: `flops_per_epoch`, `bytes_per_epoch`, `peak_mem_mb`, `n_compiles` |
| **C. Deep dive (occasionally)** | `jax.profiler.trace` + **XProf** (`pip install xprof`) | ~20 lines: `--profile` flag that traces 1–2 steady-state epochs | 0 when off; <5% during the traced window only | XProf web UI: Overview page, Trace Viewer, HLO Op Stats, Roofline; or `xprof get_overview <dir>` JSON on the terminal |
| **D. CUDA-level (rarely)** | `nsys` (already installed, 2024.6) | 0 — wrap the command | 0 when off | Nsight Systems GUI or `nsys stats` |

**W&B is layer A + the storage/comparison layer for B**, not a profiler. It is exactly right
for "is run X slower than run Y, across many runs, over weeks". It is the wrong tool
for "which kernel / which part of the epoch is slow" — that's XProf.

The "custom" part is small: a single `training/profiling.py` module (~150 lines) that
(1) wraps the AOT compile and records timings + cost analysis, (2) optionally opens a
`jax.profiler.trace` window, (3) pushes the scalars to W&B. Everything else is
off-the-shelf.

---

## 1. Why the standard Python toolbox does not apply

CRAX compiles **one epoch into a single XLA program** (`jax.jit(vmap(training_epoch))`,
see `training/agents/ppo/train.py`). Python dispatches it once per epoch and blocks.
Measured on this machine (RTX 4080, `safe_goal_point` L1, 2048 envs): a steady-state
epoch of 655k env-steps runs in ~4.5–6 s of GPU time with essentially no Python in between.

Consequences:

- `cProfile`, `py-spy`, `line_profiler`: see one opaque call. Useless for steady state.
  (Still useful for the *host-side* setup: env construction, XML generation, etc.)
- Wall-clock around the epoch call (which CRAX already does) is an accurate,
  zero-overhead measure of throughput. Good for regression, blind to *where*.
- Anything inside the program can only be seen via the XLA/CUDA-level profilers
  (JAX profiler → XProf/Perfetto, or Nsight).

And the elephant: **compilation**. In the baseline run, ~320 s of a 441 s run was
compilation (reset JIT 26 s, eval JIT 178 s, epoch JIT ~106 s) versus ~25 s of actual
training. Any timing that mixes these two is meaningless. Tristan's "2 min per level
switch" is exactly this: `train_curriculum.py` makes a new env and calls `train()` again
→ full recompile. Separating compile time from run time is therefore the first
requirement of the tooling, and it's the thing the stock scripts don't do.

---

## 2. The candidates, evaluated

### 2.1 What CRAX already has

- `training_epoch_with_timing` in `ppo/train.py`: `block_until_ready()` + wall clock →
  `training/sps`, `training/walltime`. Correct and free. The first epoch's value
  includes compile time and must be discarded (the existing
  `scripts/benchmark_crax_training.py` already does this).
- `MetricsLogger` (`training/logger.py`): computes a second SPS from the time between
  log flushes. Runs inside a `jax.debug.callback`, i.e. a host round-trip per training
  step. **This callback is unconditional** (only the second one is gated by
  `log_training_metrics`) — it is itself a candidate perf cost we should measure.
- `scripts/benchmark_crax_training.py`: sweeps `num_envs` × env, records mean/max SPS
  and CSV/PNG/LaTeX. Good bones for a regression harness, but it re-runs full
  `train()` (and so recompiles) for each config and doesn't record git SHA, compile
  time, or memory.
- Leftover `_dbg` prints in `train.py` (upstream). Harmless noise; they actually
  gave us the compile-phase timings above.

### 2.2 Weights & Biases

**What it gives for free** (we already use it):
- Per-run scalar history with overlays across runs, grouping, filtering, run table
  with summary columns. Ideal for A/B of throughput across many runs and weeks.
- System metrics via `nvidia-smi` sampling every ~10–15 s: GPU util %, memory,
  power, temperature, CPU, disk, network. Useful sanity check ("GPU util is 60% ⇒
  we're host-bound") but too coarse to attribute anything, and `nvidia-smi` "util" is
  a weak proxy (it means "a kernel was resident", not "SMs were busy").
- Code saving + git diff, `requirements.txt` capture ⇒ reproducibility for free.
- Artifacts: can store arbitrary files (e.g. the XProf `.xplane.pb` trace, a `.nsys-rep`).
  W&B **cannot render** those traces though; you download and open them in XProf.

**What it doesn't do:** no JAX/XLA profiler integration (the `sync_tensorboard`
feature syncs scalar summaries, not the profiler plugin), no compile-vs-run separation,
no per-op timing. It's a dashboard, not a profiler.

**Verdict:** use W&B as the *system of record* for throughput metrics and the place to
compare runs. Feed it a few more scalars than today (see §3). Don't try to make it
do attribution.

### 2.3 JAX profiler + XProf (recommended for attribution)

- `jax.profiler.trace(dir)` / `start_trace` / `stop_trace` around the code you care
  about. Captures host events, XLA ops, CUDA kernels (via CUPTI — present in our venv:
  `nvidia/cuda_cupti/lib/libcupti.so.12`), memory events.
- Overhead: only while tracing; OpenXLA quotes **<5% on GPU during the profiling
  window**. Zero otherwise. We'd trace 1–2 steady-state epochs (~10 s).
- Trace size: tens–hundreds of MB for a 5 s epoch with thousands of scanned steps.
  Manageable; keep `host_tracer_level=1` (user annotations + expensive ops) and
  `python_tracer_level=0` (default).
- Viewer: **XProf** (`pip install xprof`, 2.23.2, has a py3.13 manylinux wheel;
  standalone `xprof --logdir …`, no TensorBoard/TensorFlow needed). Tools:
  - *Overview page*: GPU step-time breakdown into device compute / kernel launch /
    host compute / "all other (Python)" / compile. Directly answers "are we
    GPU-bound or host-bound".
  - *Trace Viewer*: timeline of host + GPU streams. Shows the `debug.callback`
    sync stalls if they matter.
  - *HLO Op Stats / Framework Op Stats*: table of ops by self-time. With
    `jax.named_scope("rollout")`, `("sgd")`, `("env_step")`, etc. in the trainer, ops
    roll up to our own names.
  - *Roofline*: compute-bound vs memory-bound per op.
  - *Memory Viewer / Memory Profile*: peak memory and what's in it.
- **CLI**: every XProf tool is also a subcommand emitting JSON, e.g.
  `xprof get_overview <logdir>`, `get_kpi_metrics`, `get_top_hlo_ops --limit 15`,
  `get_kernel_stats --include_summary`, `get_step_trace`, `get_memory_profile`. This
  means I (Copilot) can read a profile without you screenshotting a UI, and we can
  script "before vs after" diffs (XProf ships a documented *diff-sessions*
  procedure: overview → kpi → kernel stats → top ops → HLO text diff).
- Caveats: XProf's UI loads Google Charts from the internet (charts blank when fully
  offline). The `jax.profiler` doc warns to `block_until_ready()` inside the trace
  window. Perfetto (`create_perfetto_link=True`) is an alternative viewer that needs no
  install but blocks the script until you click a link — less convenient; XProf can
  also export to Perfetto format if wanted.
- Named scopes: `jax.named_scope`/`jax.profiler.TraceAnnotation` add metadata only;
  they do not change generated code.

### 2.4 JAX AOT API (recommended for deterministic regression checks)

`jax.jit(f).lower(*args).compile()` gives us:
- `compiled.cost_analysis()` → `flops`, `bytes accessed`, transcendentals, etc. of the
  optimized program. **Deterministic**: same code + same shapes ⇒ same number, no
  GPU noise. If a change adds 3% FLOPs to the epoch, we see exactly 3%, and we see it
  *before running anything*.
- `compiled.memory_analysis()` → argument/output/temp/generated-code bytes ⇒ peak
  device memory of the program.
- `compiled.as_text()` → optimized HLO; can be diffed between commits to see *what*
  changed structurally.
- Timing `lower()` and `compile()` separately gives clean **compile time**, which is
  the metric for the ACL "does context switching recompile?" question.

JAX explicitly documents these analyses as *not stable across versions*. Fine for us:
we compare within one pinned environment.

Caveat: `training_epoch` is `jit(vmap(...))` and its inputs are the full
`TrainingState` + env state pytrees; lowering needs concrete shapes but not values, so
we can pass the real objects. This is a small refactor of the first-epoch path in
`train.py` (or a wrapper that reaches in) — see §3.

### 2.5 Recompile detection

- `jax.config.update("jax_log_compiles", True)` logs every compilation with the
  function name and shapes. `jax_explain_cache_misses=True` says *why* a tracing cache
  missed (new shape, new static arg, new pytree structure). For ACL this is the
  guard-rail: a sampler that changes an array shape or a Python-level structure
  triggers a silent recompile per epoch, and SPS falls off a cliff with no other
  symptom.
- **Persistent compilation cache** (`jax_compilation_cache_dir`,
  `jax_persistent_cache_min_compile_time_secs`): keyed on HLO + jaxlib version +
  XLA flags + GPU name. Second run of identical code skips the ~5 min compile. Big
  quality-of-life win for a benchmark harness that runs the same config repeatedly;
  also relevant for a curriculum that cycles between a fixed set of structural
  contexts (the cache would make "switch back to level 1" free).

### 2.6 Nsight Systems (`nsys`) / Nsight Compute

- `nsys` 2024.6 is on the machine. `nsys profile -t cuda,nvtx,osrt python …` gives a
  CUDA-API-level timeline (kernels, memcpy, sync, CPU threads). Overhead is low
  (a few %); the report is a self-contained `.nsys-rep`, viewable in the Nsight GUI
  or summarised with `nsys stats`.
- Adds over XProf: exact host↔device synchronisation points, CPU thread states,
  driver overhead. Loses: JAX op names (unless we emit NVTX ranges), the HLO-level
  tools.
- Verdict: keep as the second-opinion tool for host/device stall questions. Not the
  daily driver.

### 2.7 Things considered and rejected for now

- **PyTorch-style step profilers / `torch.profiler` W&B integration**: n/a for JAX.
- **`jax.profiler.device_memory_profile()` + pprof**: fine but XProf's memory viewer
  covers it.
- **TensorBoard**: only needed as a container for XProf; standalone `xprof` is enough.
- **Custom Perfetto trace writer / bespoke timeline**: unnecessary — XProf already
  produces Perfetto-compatible traces.
- **W&B profiler artifacts as primary**: W&B can *store* a trace but the viewing
  loop (download → xprof) is worse than pointing xprof at a local `runs/profiles`
  dir. Store to W&B only for sharing with supervisors.

---

## 3. Proposed concrete design (small)

### 3.1 Where numbers live

| Metric | Where computed | Where stored |
|---|---|---|
| `training/sps`, `training/walltime` (per epoch) | already in `train.py` | W&B history (already) |
| `perf/compile_time_s`, `perf/lower_time_s`, `perf/first_epoch_s` | new helper around first epoch | W&B summary + JSON |
| `perf/flops_per_epoch`, `perf/bytes_per_epoch`, `perf/peak_device_mem_mb` | `compiled.cost_analysis()`, `memory_analysis()` | W&B summary + JSON |
| `perf/n_compiles` (over the run) | counter fed by `jax_log_compiles` logging handler | W&B summary + JSON |
| `perf/median_sps`, `perf/iqr_sps` (epochs ≥ 2) | end of run | W&B summary + JSON |
| git SHA, dirty flag, jax/mujoco versions, GPU name, XLA flags | at start | W&B config + JSON |
| XProf trace (`.xplane.pb`) | `--profile` only | `runs/profiles/<run_name>/plugins/profile/<session>/`; optionally W&B artifact |

The JSON sidecar (one file per run under `runs/profiles/`) exists so we can compare
runs offline and so results don't depend on W&B being reachable.

### 3.2 Interface

```
# Normal training, extra scalars appear in W&B automatically (no cost):
crax-train --env_name safe_goal_point --alg ppo_lag --num_envs 2048 --use_wandb

# Same, plus an XProf trace of epochs 3–4 (only those epochs pay <5% overhead):
crax-train ... --profile --profile_epochs 3 4

# Look at it:
xprof --logdir runs/profiles/<run_name>          # web UI at localhost:8791
xprof get_overview runs/profiles/<run_name>      # JSON summary in the terminal
xprof get_top_hlo_ops runs/profiles/<run_name> --limit 15

# Regression harness (fixed config, fixed seed, N epochs, JSON out) — reuses the
# above; this is scripts/benchmark_crax_training.py with the perf scalars added:
python scripts/benchmark_crax_training.py --envs safe_goal_point --num_envs 2048 \
    --num_epochs 10 --out runs/profiles/baseline.json
python scripts/compare_perf.py runs/profiles/baseline.json runs/profiles/candidate.json
```

Programmatic entry point (`training/profiling.py`):

```python
@dataclass
class PerfRecord: ...   # the fields in §3.1

with perf_tracker(run_name, wandb=use_wandb) as perf:      # collects compile counts, git, versions
    compiled = perf.aot_compile(training_epoch, training_state, env_state, keys)  # times lower/compile, logs cost/memory
    for epoch in range(...):
        with perf.epoch(epoch):                             # wall-clock; opens jax.profiler.trace if epoch in profile_epochs
            ...
perf.summary()  # median/IQR SPS, writes JSON, pushes W&B summary
```

`train.py` changes are limited to: (a) calling `perf.aot_compile` instead of letting
the first call compile implicitly, (b) wrapping the epoch call, (c) a handful of
`jax.named_scope(...)` around rollout / GAE / SGD / env-step so XProf tables are
readable. None of these alter the compiled program.

### 3.3 Measurement protocol (so numbers are comparable)

1. Fixed config + seed; `num_evals` chosen so that one epoch ≈ 5–10 s.
2. Discard epoch 0 (compile). Report **median and IQR** of epochs 1..N (N ≥ 8).
   Median, not mean: GPU clocks/thermal produce right-skewed outliers.
3. Record `flops_per_epoch` and `bytes_per_epoch`. If a change moves wall time but
   not these, it's scheduling/launch/host; if both move, it's the program.
4. `n_compiles` must be constant between baseline and candidate; if not, stop —
   the comparison is invalid until the extra recompile is understood.
5. Same GPU, same XLA flags, nothing else on the GPU (check `nvidia-smi`).
6. First experiment with the new tooling: quantify the unconditional
   `jax.debug.callback` in `training_step` (env-metrics logging) by comparing
   with/without. It's the most likely existing host↔device sync.

---

## 4. Open questions for Giuseppe

1. **W&B as primary store** — fine, or do you prefer local JSON + a small pandas/plot
   script (no network dependency, easier to version)? I'd do both, JSON always.
2. **Cluster**: do you have SLURM/GPU-cluster access for the real experiments? If so,
   the tooling should write self-contained artifacts (JSON + `.xplane.pb`) that we
   `rsync` back and inspect locally with `xprof`. Also affects whether the
   persistent compilation cache should live on shared storage.
3. **Upstream-ability**: keep `training/profiling.py` purely additive (opt-in flags,
   default behaviour unchanged) so it could be PR'd to Tristan's repo? I'd assume yes.
4. **Trace privacy**: XProf UI fetches Google Charts JS from the web (no data leaves
   the machine); is that acceptable on the machines you'll use?

---

## 5. Sources

- JAX profiling guide (Perfetto, XProf, options, GPU/CUPTI troubleshooting):
  https://docs.jax.dev/en/latest/profiling.html
- JAX AOT (`lower`/`compile`/`cost_analysis`/`memory_analysis`):
  https://docs.jax.dev/en/latest/aot.html
- JAX persistent compilation cache: https://docs.jax.dev/en/latest/persistent_compilation_cache.html
- XProf: https://openxla.org/xprof (features, <5% GPU overhead claim), overview-page
  GPU breakdown: https://openxla.org/xprof/overview_page, repo/install/CLI:
  https://github.com/openxla/xprof, CLI reference & diff-session procedure:
  https://github.com/openxla/xprof/tree/master/skills/xprof
- W&B logging & auto-logged system metrics: https://docs.wandb.ai/models/track/log
- Local measurements: see `/memories/repo/profiling.md` (baseline run 2026-09-19).
