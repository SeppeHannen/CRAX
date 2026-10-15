"""Run an experiment's jobs one after another on this machine.

    .venv/bin/python -m training.experiments.local scripts/experiments/<name>.py

Each job's console output goes to ``runs/logs/<group>_<label>_<timestamp>.log``.
A job that fails stops the chain: the remaining arms would join a group whose
result is already in question. Run detached so the chain survives the terminal:

    setsid nohup .venv/bin/python -m training.experiments.local <definition> \\
        > runs/logs/<group>_chain.log 2>&1 < /dev/null & disown
"""
from __future__ import annotations

import argparse
import datetime
import pathlib
import subprocess
import sys

from training.experiments.definition import load_experiment, prepare_group

LOG_DIRECTORY = pathlib.Path("runs/logs")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("definition", type=pathlib.Path, help="Python file exporting EXPERIMENT")
    arguments = parser.parse_args()

    experiment = load_experiment(arguments.definition)
    jobs = experiment.jobs()
    prepare_group(experiment)
    LOG_DIRECTORY.mkdir(parents=True, exist_ok=True)
    for job in jobs:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        log = LOG_DIRECTORY / f"{experiment.group}_{job.label}_{timestamp}.log"
        print(f"{timestamp} starting {job.label} -> {log}", flush=True)
        with log.open("w") as output:
            completed = subprocess.run([sys.executable, "-m", "training.train_env", *job.argv], stdout=output, stderr=subprocess.STDOUT)
        print(f"{datetime.datetime.now():%Y%m%d_%H%M%S} finished {job.label} (exit {completed.returncode})", flush=True)
        if completed.returncode != 0:
            raise SystemExit(f"{job.label} failed with exit code {completed.returncode}; see {log}")


if __name__ == "__main__":
    main()
