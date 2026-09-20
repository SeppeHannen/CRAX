#!/usr/bin/env python3
"""Sweep num_envs to find where the launch-bound regime ends on this GPU.

For each (environment, num_envs) pair this runs a short PPO-Lag training with
``--measure_performance`` so W&B receives ``performance/steady_steps_per_second_median``
etc. Every run is tagged so the curve can be plotted in W&B directly:

    line plot  x = config.num_envs  (log2)   y = performance/steady_steps_per_second_median
    group by   config.env_name                filter tags: sweep-num-envs

Each run is a separate process so a CUDA OOM at large num_envs cannot take the
rest of the sweep down; failures are recorded and the sweep continues.

Usage (defaults: Point / Ant / Humanoid morphologies, 2048..32768):

    python scripts/sweep_num_envs.py
    python scripts/sweep_num_envs.py --envs safe_goal_point --num_envs 2048 4096 8192
    python scripts/sweep_num_envs.py --dry_run

A summary table is printed and written to runs/logs/sweep_num_envs_<timestamp>.md.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
PYTHON = sys.executable

# One representative per morphology class: 2-DoF puck, quadruped with contacts,
# high-DoF humanoid. Everything else in CRAX sits between these.
DEFAULT_ENVS = ["safe_goal_point", "safe_velocity_ant", "safe_velocity_humanoid"]
DEFAULT_NUM_ENVS = [2048, 4096, 8192, 16384, 32768]

# Same PPO shape as the baseline measurements. Keep batch_size*num_minibatches
# fixed so data-per-update is constant; only the parallelism changes.
BASE_ARGS = [
    "--alg", "ppo_lag", "--difficulty", "1", "--seeds", "0",
    "--episode_length", "1000",
    "--unroll_length", "20", "--batch_size", "1024", "--num_minibatches", "32",
    "--num_updates_per_batch", "4", "--num_eval_envs", "128",
    "--store_model", "false", "--skip_rollout", "--skip_video", "--quiet",
    "--measure_performance",
]
ENV_STEPS_PER_TRAINING_STEP = 1024 * 32 * 20  # 655,360


def run_one(env_name: str, num_envs: int, epochs: int, tag: str, log_dir: Path) -> dict:
    # num_evals - 1 epochs of one training step each (num_timesteps rounds up).
    num_timesteps = ENV_STEPS_PER_TRAINING_STEP * epochs
    run_stem = f"sweep_{env_name}_{num_envs}"
    log_path = log_dir / f"{run_stem}.log"
    cmd = [
        PYTHON, "-m", "training.train_env",
        "--env_name", env_name, "--num_envs", str(num_envs),
        "--num_timesteps", str(num_timesteps), "--num_evals", str(epochs + 1),
        "--wandb_group", "sweep-num-envs", "--wandb_tags", "sweep-num-envs", tag, env_name,
        "--performance_dir", str(REPO / "runs" / "performance"),
        *BASE_ARGS,
    ]
    print(f"\n=== {env_name} @ {num_envs} envs  ({epochs} epochs, log: {log_path.name})", flush=True)
    start = time.time()
    with open(log_path, "w") as log:
        log.write(" ".join(cmd) + "\n\n")
        log.flush()
        proc = subprocess.run(cmd, cwd=REPO, stdout=log, stderr=subprocess.STDOUT,
                              env={**os.environ, "MUJOCO_GL": "egl"})
    wall = time.time() - start
    result = {"env_name": env_name, "num_envs": num_envs, "wall_s": wall, "exit": proc.returncode}

    # Pick up the JSON the tracker wrote (newest matching run dir).
    perf_dirs = sorted((REPO / "runs" / "performance").glob(f"{env_name}_Level_1_ppo_lag_seed0_*"),
                       key=lambda p: p.stat().st_mtime)
    if proc.returncode == 0 and perf_dirs:
        try:
            document = json.load(open(perf_dirs[-1] / "performance.json"))
            summary = document["summary"]
            epochs = document.get("epochs", [])
            # Fall back to the median over all non-compile epochs if the tracker's
            # steadiness filter rejected everything; flag it so the table shows it.
            sps = summary.get("steady_steps_per_second_median")
            fallback = False
            if sps is None and len(epochs) > 1:
                import statistics
                sps = statistics.median(e["steps_per_second"] for e in epochs[1:])
                fallback = True
            result.update({
                "sps_median": sps,
                "sps_iqr": summary.get("steady_steps_per_second_iqr"),
                "steady_epochs": summary.get("num_steady_epochs"),
                "sps_fallback": fallback,
                "epoch_compile_s": summary.get("phase_epoch_compile_seconds"),
                "peak_mem_gib": (summary.get("program_peak_memory_bytes") or 0) / 2**30,
                "run_name": summary.get("run_name"),
            })
            print(f"    SPS {_fmt(sps)}{' (all-epoch fallback)' if fallback else ''}  "
                  f"(IQR {_fmt(result['sps_iqr'])})  steady {result['steady_epochs']}  "
                  f"peak mem {result['peak_mem_gib']:.1f} GiB  wall {wall:.0f}s", flush=True)
        except Exception as exc:  # never let a bookkeeping problem end the sweep
            result["error"] = f"could not read performance.json: {exc}"
            print(f"    ran, but {result['error']}", flush=True)
    else:
        tail = log_path.read_text().splitlines()[-15:]
        oom = any("RESOURCE_EXHAUSTED" in l or "out of memory" in l.lower() for l in tail)
        result["error"] = "OOM" if oom else f"exit {proc.returncode}"
        print(f"    FAILED ({result['error']}) after {wall:.0f}s — see {log_path}", flush=True)
    return result


def _fmt(value, spec: str = ",.0f") -> str:
    return "n/a" if value is None else format(value, spec)


def write_report(results: list, path: Path, tag: str) -> None:
    lines = [f"# num_envs sweep — {tag}", "",
             f"Machine: {os.uname().nodename}. Tag `sweep-num-envs`, W&B group `sweep-num-envs`.", "",
             "| env | num_envs | steady SPS (median) | IQR | steady epochs | epoch compile (s) | peak mem (GiB) | wall (s) |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for r in results:
        if r.get("sps_median") is not None:
            note = " (all-epoch median)" if r.get("sps_fallback") else ""
            lines.append(f"| {r['env_name']} | {r['num_envs']} | {_fmt(r['sps_median'])}{note} | "
                         f"{_fmt(r.get('sps_iqr'))} | {r.get('steady_epochs')} | {_fmt(r.get('epoch_compile_s'), '.1f')} | "
                         f"{_fmt(r.get('peak_mem_gib'), '.2f')} | {r['wall_s']:.0f} |")
        else:
            lines.append(f"| {r['env_name']} | {r['num_envs']} | FAILED: {r.get('error')} | | | | | {r['wall_s']:.0f} |")
    path.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nreport: {path}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--envs", nargs="+", default=DEFAULT_ENVS)
    parser.add_argument("--num_envs", nargs="+", type=int, default=DEFAULT_NUM_ENVS)
    parser.add_argument("--epochs", type=int, default=8, help="steady epochs per run (plus the compile epoch)")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    tag = datetime.now().strftime("sweep-%Y%m%d-%H%M")
    log_dir = REPO / "runs" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    plan = [(e, n) for e in args.envs for n in args.num_envs]
    print(f"{len(plan)} runs, ~{len(plan) * 3} min estimated. Tag: {tag}")
    if args.dry_run:
        for e, n in plan:
            print(f"  {e} @ {n}")
        return 0

    results = []
    for env_name, num_envs in plan:
        results.append(run_one(env_name, num_envs, args.epochs, tag, log_dir))
    write_report(results, log_dir / f"sweep_num_envs_{tag}.md", tag)
    return 0


if __name__ == "__main__":
    sys.exit(main())
