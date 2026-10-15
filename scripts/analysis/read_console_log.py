"""Read a training run's console log as a per-round table, while the run is on.

The W&B history is the record; this is the glance at a running run before it
finishes. ``train_env`` prints, once per round, every logged key whose name
mentions λ, cost or reward (``training/run_utils.py``, ``CONSOLE_KEY_TOKENS``),
so those keys can be read back without touching W&B.

    .venv/bin/python -m scripts.analysis.read_console_log runs/logs/<run>.log [--from-round 40] [--every 1]
"""
from __future__ import annotations

import argparse
import math
import pathlib
import re
from typing import Dict, Mapping

STEPS_PER_ROUND = 655_360  # the 500 M protocol: batch 1024 × 32 minibatches × unroll 20

COLUMNS: Mapping[str, str] = {
    "training/lambda_lagr": "λ",
    "training/lambda_lagr/std": "λ std",
    "training/lambda_lagr/min": "λ min",
    "training/lambda_lagr/max": "λ max",
    "training/lambda_lagr/fit_error": "fit error",
    "training/lambda_lagr/target_spread": "target sd",
    "training/cost_violation": "violation",
    "episodic/cost": "train cost",
    "evaluation/deployment/episode_cost": "cost L3",
    "evaluation/deployment/episode_reward": "return L3",
    "evaluation/uniform/episode_cost": "cost Ω",
    "evaluation/uniform/episode_reward": "return Ω",
}


def rounds_from_log(text: str, steps_per_round: int) -> Dict[int, Dict[str, float]]:
    """``{round: {key: value}}`` from the ``Step N:`` blocks; the last value of a key at a step wins."""
    rows: Dict[int, Dict[str, float]] = {}
    parts = re.split(r"\nStep (\d+):", text)
    for step, body in zip(parts[1::2], parts[2::2]):
        row = rows.setdefault(int(step) // steps_per_round, {})
        for key in COLUMNS:
            match = re.search(re.escape(key) + r": ([-\d.e+]+)\n", body)
            if match:
                row[key] = float(match.group(1))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("log", type=pathlib.Path)
    parser.add_argument("--from-round", type=int, default=0)
    parser.add_argument("--every", type=int, default=1, help="print every n-th round (evaluation rounds always)")
    parser.add_argument("--steps-per-round", type=int, default=STEPS_PER_ROUND)
    arguments = parser.parse_args()
    rows = rounds_from_log(arguments.log.read_text(), arguments.steps_per_round)
    print(f"{len(rows)} rounds logged")
    header = f"{'round':>5} " + " ".join(f"{name:>10}" for name in COLUMNS.values())
    print(header)
    for round_index in sorted(rows):
        row = rows[round_index]
        is_evaluation = "evaluation/deployment/episode_cost" in row
        if round_index < arguments.from_round or (round_index % arguments.every and not is_evaluation):
            continue
        cells = [f"{row.get(key, math.nan):>10.3f}" for key in COLUMNS]
        print(f"{round_index:>5} " + " ".join(cells))


if __name__ == "__main__":
    main()
