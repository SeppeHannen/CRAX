"""Pull the full per-round scalar history of every run in a W&B group to local parquet files.

    .venv/bin/python scripts/analysis/pull_group_history.py <group> <output_dir>

One file per training run, named by the arm the run declares in its config (the key the
dashboard groups lines by, e.g. ``uniform scalar blind`` -> ``uniform_scalar_blind``); the
group's budget reference run has no history worth pulling and is skipped. `scan_history` is
slow (~3 min per 780-round run); pull once, then analyse and plot from the files.
"""
from __future__ import annotations

import argparse
import pathlib

import pandas as pd
import wandb

from training.config import WANDB_ENTITY, WANDB_PROJECT
from training.dashboard.budget_run import JOB_TYPE as BUDGET_JOB_TYPE
from training.dashboard.view import GROUP_RUNS_BY


def file_stem_of(run: wandb.apis.public.Run) -> str:
    """The run's arm as a file name: ``uniform scalar blind`` -> ``uniform_scalar_blind``."""
    return str(run.config[GROUP_RUNS_BY]).replace(" ", "_").replace(":", "").replace(",", "")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("group")
    parser.add_argument("output_dir", type=pathlib.Path)
    arguments = parser.parse_args()
    arguments.output_dir.mkdir(parents=True, exist_ok=True)

    api = wandb.Api(timeout=300)
    runs = api.runs(f"{WANDB_ENTITY}/{WANDB_PROJECT}", filters={"group": arguments.group})
    for run in runs:
        if run.job_type == BUDGET_JOB_TYPE:
            continue
        rows = [{key: value for key, value in row.items() if not isinstance(value, (dict, list))} for row in run.scan_history()]
        frame = pd.DataFrame(rows)
        path = arguments.output_dir / f"{file_stem_of(run)}.parquet"
        frame.to_parquet(path)
        print(f"{run.name}: {frame.shape[0]} rows, {frame.shape[1]} scalar columns -> {path}")


if __name__ == "__main__":
    main()
