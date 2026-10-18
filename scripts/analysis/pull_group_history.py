"""Pull the per-round scalar history of every run in a W&B group to local parquet files.

    .venv/bin/python scripts/analysis/pull_group_history.py <group> <output_dir>
    .venv/bin/python scripts/analysis/pull_group_history.py <group> <output_dir> --samples 1000

One file per training run, named by the arm the run declares in its config (the key the
dashboard groups lines by, e.g. ``uniform scalar blind`` -> ``uniform_scalar_blind``); the
group's budget reference run has no history worth pulling and is skipped.

Without ``--samples`` every logged row is pulled (`scan_history`: ~3 min per 780-round run,
~25 min per 6000-round run). With it, every evaluation row is pulled exactly and the training
rows are an even sample of that many, which is what a figure over 500 M steps resolves
anyway; seconds per run. Both layouts are one row per logged step with NaN where a key was
not logged, so the same analysis reads either.
"""
from __future__ import annotations

import argparse
import pathlib
from typing import List, Mapping

import pandas as pd
import wandb

from training.config import WANDB_ENTITY, WANDB_PROJECT
from training.dashboard.budget_run import JOB_TYPE as BUDGET_JOB_TYPE
from training.dashboard.view import GROUP_RUNS_BY


def file_stem_of(run: wandb.apis.public.Run) -> str:
    """The run's arm as a file name: ``uniform scalar blind`` -> ``uniform_scalar_blind``."""
    return str(run.config[GROUP_RUNS_BY]).replace(" ", "_").replace(":", "").replace(",", "")


def _scalars(row: Mapping[str, object]) -> dict:
    """The numeric entries of a logged row: histograms (dicts), lists and strings (a run name a
    performance tracker logs) are not scalars and a sampled history mixes them with NaN."""
    return {key: value for key, value in row.items() if isinstance(value, (int, float)) and not isinstance(value, bool)}


def every_row(run: wandb.apis.public.Run) -> pd.DataFrame:
    return pd.DataFrame([_scalars(row) for row in run.scan_history()])


def sampled_rows(run: wandb.apis.public.Run, samples: int) -> pd.DataFrame:
    """Every evaluation row (there are few, and the end state is read from them) plus ``samples``
    evenly spaced rows of everything, merged on the W&B step."""
    evaluation_keys: List[str] = [key for key, value in run.summary.items() if key.startswith("evaluation/") and isinstance(value, (int, float))]
    if not evaluation_keys:
        raise ValueError(f"{run.name} logged no evaluation/* scalars")
    evaluations = pd.DataFrame([_scalars(row) for row in run.history(keys=evaluation_keys + ["environment_steps"], samples=100_000, pandas=False)])
    training = pd.DataFrame([_scalars(row) for row in run.history(samples=samples, pandas=False)])
    merged = pd.concat([training, evaluations]).drop_duplicates(subset="_step", keep="last").sort_values("_step")
    return merged.reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("group")
    parser.add_argument("output_dir", type=pathlib.Path)
    parser.add_argument("--samples", type=int, default=None, help="pull every evaluation row and this many training rows per run instead of every row")
    arguments = parser.parse_args()
    arguments.output_dir.mkdir(parents=True, exist_ok=True)

    api = wandb.Api(timeout=300)
    runs = api.runs(f"{WANDB_ENTITY}/{WANDB_PROJECT}", filters={"group": arguments.group})
    for run in runs:
        if run.job_type == BUDGET_JOB_TYPE:
            continue
        frame = every_row(run) if arguments.samples is None else sampled_rows(run, arguments.samples)
        path = arguments.output_dir / f"{file_stem_of(run)}.parquet"
        frame.to_parquet(path)
        print(f"{run.name}: {frame.shape[0]} rows, {frame.shape[1]} scalar columns -> {path}")


if __name__ == "__main__":
    main()
