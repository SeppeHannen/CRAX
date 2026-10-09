"""Pull the full per-round scalar history of every run in a W&B group to local parquet files.

    .venv/bin/python scripts/analysis/pull_group_history.py <group> <output_dir>

One file per run, named by the run's training distribution (the `_ctx_<spec>_` part
of the run name). `scan_history` is slow (~3 min per 780-round run); pull once, then
analyse and plot from the files.
"""
from __future__ import annotations

import argparse
import pathlib

import pandas as pd
import wandb

from training.config import WANDB_ENTITY, WANDB_PROJECT


def training_distribution_of(run_name: str) -> str:
    """``safe_goal_point_ctx_staged123_ppo_lag_seed0_...`` -> ``staged123``."""
    return run_name.split("_ctx_")[1].split("_ppo")[0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("group")
    parser.add_argument("output_dir", type=pathlib.Path)
    arguments = parser.parse_args()
    arguments.output_dir.mkdir(parents=True, exist_ok=True)

    api = wandb.Api(timeout=300)
    runs = api.runs(f"{WANDB_ENTITY}/{WANDB_PROJECT}", filters={"group": arguments.group})
    for run in runs:
        rows = [{key: value for key, value in row.items() if not isinstance(value, (dict, list))} for row in run.scan_history()]
        frame = pd.DataFrame(rows)
        path = arguments.output_dir / f"{training_distribution_of(run.name)}.parquet"
        frame.to_parquet(path)
        print(f"{run.name}: {frame.shape[0]} rows, {frame.shape[1]} scalar columns -> {path}")


if __name__ == "__main__":
    main()
