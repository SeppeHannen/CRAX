"""What an experiment is, and the checks every launcher runs before starting it.

The flags are kept as the strings ``train_env`` would receive, not mirrored into
a second typed configuration: ``training.config.build_base_parser`` is the one
definition of the flags, and every job is parsed with it here, so an unknown or
malformed flag fails at load time, before anything is submitted.
"""
from __future__ import annotations

import argparse
import dataclasses
import importlib.util
import pathlib
import re
from typing import Tuple

from training import contexts, dashboard
from training.config import build_base_parser
from training.dashboard import RunFacts

LABEL_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
RESERVED_FLAGS = ("--wandb_group", "--seeds")
EXPORTED_NAME = "EXPERIMENT"


@dataclasses.dataclass(frozen=True)
class Arm:
    """One training distribution under one learner: the flags that make it differ from the other arms."""

    label: str  # file- and job-name safe; names the arm in logs and job names
    flags: Tuple[str, ...]

    def __post_init__(self) -> None:
        if not LABEL_PATTERN.match(self.label):
            raise ValueError(f"arm label {self.label!r} must match {LABEL_PATTERN.pattern} (it names files and jobs)")


@dataclasses.dataclass(frozen=True)
class Job:
    """One process: one arm under one seed, with the full argument list of ``training.train_env``."""

    label: str
    argv: Tuple[str, ...]
    arguments: argparse.Namespace


@dataclasses.dataclass(frozen=True)
class Experiment:
    """A W&B group of arms that share ``common`` and differ in their own flags, each run once per seed."""

    group: str
    common: Tuple[str, ...]
    arms: Tuple[Arm, ...]
    seeds: Tuple[int, ...] = (0,)

    def __post_init__(self) -> None:
        if not LABEL_PATTERN.match(self.group):
            raise ValueError(f"group {self.group!r} must match {LABEL_PATTERN.pattern}")
        if not self.arms:
            raise ValueError(f"experiment {self.group!r} has no arms")
        if not self.seeds:
            raise ValueError(f"experiment {self.group!r} has no seeds")
        labels = [arm.label for arm in self.arms]
        if len(set(labels)) != len(labels):
            raise ValueError(f"arm labels must be unique, got {labels}")
        for flag in RESERVED_FLAGS:
            if flag in self.common or any(flag in arm.flags for arm in self.arms):
                raise ValueError(f"{flag} is set from the experiment's `group`/`seeds` fields, not passed as a flag")

    def jobs(self) -> Tuple[Job, ...]:
        """Every arm under every seed, parsed with the real CLI so a bad flag fails here."""
        parser = build_base_parser()
        jobs = []
        for arm in self.arms:
            for seed in self.seeds:
                argv = (*self.common, *arm.flags, "--wandb_group", self.group, "--seeds", str(seed))
                label = arm.label if len(self.seeds) == 1 else f"{arm.label}_seed{seed}"
                jobs.append(Job(label=label, argv=argv, arguments=parser.parse_args(list(argv))))
        return tuple(jobs)

    def facts(self) -> RunFacts:
        """The dashboard facts all jobs share; raises if two jobs would not be allowed into one group."""
        facts_by_job = {job.label: run_facts(job.arguments) for job in self.jobs()}
        distinct = set(facts_by_job.values())
        if len(distinct) != 1:
            raise ValueError(
                f"the arms of {self.group!r} differ in the facts the dashboard states; one group cannot hold them:\n"
                + "\n".join(f"  {label}: {facts}" for label, facts in facts_by_job.items())
            )
        return distinct.pop()


def run_facts(arguments: argparse.Namespace) -> RunFacts:
    """The dashboard facts of a run with these flags, from the flags alone (no environment, no GPU).

    The same facts ``train_env`` derives at start-up, which is why ``--episode_length``
    must be explicit: without it the run reads the length off the environment.
    """
    if arguments.context_distribution == contexts.NO_DISTRIBUTION:
        raise ValueError("an experiment group has a dashboard view, which needs --context_distribution other than 'none'")
    if arguments.episode_length is None:
        raise ValueError("--episode_length must be given explicitly: the dashboard states it before an environment exists")
    setup = contexts.context_training_setup(
        arguments.env_name,
        arguments.context_distribution,
        arguments.deployment_distribution,
        contexts.Learner(constraint=arguments.constraint, policy_context=arguments.policy_context),
        num_timesteps=int(arguments.num_timesteps),
        num_envs=arguments.num_envs,
        batch_size=arguments.batch_size,
        unroll_length=arguments.unroll_length,
        num_minibatches=arguments.num_minibatches,
    )
    assert setup is not None  # NO_DISTRIBUTION was excluded above
    return RunFacts.from_wandb_config({**vars(arguments), "episode_length": arguments.episode_length, **setup.wandb_config()})


def prepare_group(experiment: Experiment) -> None:
    """Create the group's view and budget run (or check they match) before any job starts."""
    jobs = experiment.jobs()
    entities = {(job.arguments.wandb_entity, job.arguments.wandb_project) for job in jobs}
    if len(entities) != 1:
        raise ValueError(f"all arms must log to one W&B entity/project, got {sorted(entities)}")
    (entity, project), = entities
    dashboard.ensure_view(entity, project, experiment.group, experiment.facts())


def load_experiment(path: pathlib.Path) -> Experiment:
    """Import a definition file and return its ``EXPERIMENT``."""
    if not path.is_file():
        raise FileNotFoundError(f"no experiment definition at {path}")
    specification = importlib.util.spec_from_file_location(f"experiment_{path.stem}", path)
    if specification is None or specification.loader is None:
        raise ImportError(f"cannot import {path}")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    if not hasattr(module, EXPORTED_NAME):
        raise AttributeError(f"{path} must define {EXPORTED_NAME}")
    experiment = getattr(module, EXPORTED_NAME)
    if not isinstance(experiment, Experiment):
        raise TypeError(f"{path}:{EXPORTED_NAME} is {type(experiment).__name__}, expected Experiment")
    return experiment
