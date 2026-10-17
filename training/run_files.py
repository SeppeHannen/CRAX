"""The one door by which a file a run writes reaches W&B.

W&B uploads only the files it is told about, and only ones that exist when it is told
(a glob registered before the files appear uploads nothing; verified 2026-10-15 for
checkpoints, 2026-10-17 for episode records). So every producer of a run artefact —
the checkpoint writer, the episode recorder — calls :func:`hand_to_run` once per
written path. A path outside the active run's directory, or written with no run
active (a test, the transfer benchmark's working directory), stays local.
"""
from __future__ import annotations

import pathlib
from typing import Union

import wandb
from etils import epath


def hand_to_run(path: Union[pathlib.Path, epath.Path]) -> None:
    """Register ``path`` (a file, or a directory with everything below it) with the active
    W&B run for upload now, if it lies in the run's directory."""
    if wandb.run is None:
        return
    run_directory = pathlib.Path(wandb.run.dir).resolve()
    resolved = pathlib.Path(str(path)).resolve()
    if run_directory not in resolved.parents:
        return
    glob = str(resolved / "**") if resolved.is_dir() else str(resolved)
    wandb.run.save(glob, base_path=str(run_directory), policy="now")
