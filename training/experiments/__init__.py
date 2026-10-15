"""An experiment defined once, run anywhere.

An :class:`Experiment` is a W&B group, the flags every arm shares and the arms
that differ from each other; it expands into one :class:`Job` per arm and seed,
each the exact argument list of ``python -m training.train_env``. Two launchers
read the same definition: :mod:`training.experiments.local` runs the jobs one
after another on this machine, :mod:`training.experiments.snellius` submits one
SLURM job per arm on Snellius. Both create the group's dashboard view before
the first job starts, so concurrently starting arms never race to create it.

A definition is a Python file exporting ``EXPERIMENT`` (``scripts/experiments/``).
"""
from training.experiments.definition import Arm, Experiment, Job, load_experiment, prepare_group, run_facts

__all__ = ["Arm", "Experiment", "Job", "load_experiment", "prepare_group", "run_facts"]
