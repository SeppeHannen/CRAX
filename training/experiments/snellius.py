"""Submit an experiment to Snellius: one SLURM job per arm and seed, all at once.

    .venv/bin/python -m training.experiments.snellius scripts/experiments/<name>.py \\
        --partition gpu_a100 --time 04:00:00 [--dry-run]

A job runs a commit, never a working tree: the submitter takes the commit this
checkout is at, requires it to be pushed and the tree to be clean, checks the
commit out on the cluster into its own worktree and submits a job script whose
argument list is the job's. Every run therefore knows its exact code (W&B
records the commit) and two experiments at different commits never share a
checkout. The cluster side is fixed by :class:`Cluster`: a clone to fetch from,
one Python environment for every commit (``scripts/snellius/install_environment.sh``),
one worktree per commit, one log directory per group.

The job script is generated here rather than kept as a template so it is read
in one place and can be printed exactly as submitted (``--dry-run``).
"""
from __future__ import annotations

import argparse
import dataclasses
import pathlib
import shlex
import subprocess
from typing import Sequence, Tuple

from training.experiments.definition import Experiment, Job, load_experiment, prepare_group

SHORT_COMMIT_LENGTH = 12


@dataclasses.dataclass(frozen=True)
class Cluster:
    """Where things live on Snellius; ``host`` is the ``~/.ssh/config`` alias."""

    host: str
    home: str

    @property
    def clone(self) -> str:
        return f"{self.home}/CRAX"

    @property
    def python(self) -> str:
        return f"{self.home}/crax-venv/bin/python"

    def checkout(self, commit: str) -> str:
        return f"{self.home}/crax-runs/{commit}"

    def log_directory(self, group: str) -> str:
        return f"{self.home}/crax-logs/{group}"

    def run(self, command: str, *, input: str | None = None) -> str:
        """Run a shell command on the cluster; raises with its stderr on failure."""
        completed = subprocess.run(["ssh", self.host, command], input=input, capture_output=True, text=True)
        if completed.returncode != 0:
            raise RuntimeError(f"on {self.host}, `{command}` failed ({completed.returncode}):\n{completed.stderr}")
        return completed.stdout

    @classmethod
    def connect(cls, host: str) -> "Cluster":
        home = subprocess.run(["ssh", host, 'printf %s "$HOME"'], capture_output=True, text=True, check=True).stdout
        if not home.startswith("/"):
            raise RuntimeError(f"could not read $HOME on {host}: {home!r}")
        return cls(host=host, home=home)


@dataclasses.dataclass(frozen=True)
class Resources:
    """What each job asks SLURM for: one GPU on ``partition`` for ``time`` (``HH:MM:SS``)."""

    partition: str
    time: str


def pushed_commit(repository: pathlib.Path) -> str:
    """The commit this checkout is at, if the tree is clean and the commit is on a remote branch."""
    git = ["git", "-C", str(repository)]
    dirty = subprocess.run([*git, "status", "--porcelain"], capture_output=True, text=True, check=True).stdout
    if dirty:
        raise RuntimeError(f"a job runs a commit, and these files are not in one; commit or stash them first:\n{dirty}")
    commit = subprocess.run([*git, "rev-parse", f"--short={SHORT_COMMIT_LENGTH}", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    remote_branches = subprocess.run([*git, "branch", "--remotes", "--contains", commit], capture_output=True, text=True, check=True).stdout
    if not remote_branches.strip():
        raise RuntimeError(f"commit {commit} is on no remote branch; push it so the cluster can fetch it")
    return commit


def job_script(cluster: Cluster, resources: Resources, experiment: Experiment, job: Job, commit: str) -> str:
    """The sbatch script for one job, complete."""
    checkout = cluster.checkout(commit)
    command = shlex.join([cluster.python, "-m", "training.train_env", *job.argv])
    return "\n".join([
        "#!/usr/bin/env bash",
        f"#SBATCH --job-name={experiment.group}__{job.label}",
        f"#SBATCH --partition={resources.partition}",
        "#SBATCH --gpus=1",
        f"#SBATCH --time={resources.time}",
        f"#SBATCH --output={cluster.log_directory(experiment.group)}/{job.label}_%j.out",
        "set -euo pipefail",
        f'echo "job $SLURM_JOB_ID on $(hostname), commit {commit}, $(nvidia-smi --query-gpu=name --format=csv,noheader)"',
        f"cd {shlex.quote(checkout)}",
        f"export PYTHONPATH={shlex.quote(checkout)}",
        f"exec {command}",
        "",
    ])


def ensure_checkout(cluster: Cluster, commit: str) -> None:
    """Fetch and check the commit out into its worktree on the cluster, if it is not there yet."""
    checkout = cluster.checkout(commit)
    cluster.run(
        f"cd {shlex.quote(cluster.clone)} && git fetch --quiet origin && "
        f"([ -d {shlex.quote(checkout)} ] || git worktree add --quiet {shlex.quote(checkout)} {commit})"
    )


def submit(cluster: Cluster, script: str) -> str:
    """Submit a job script through stdin; returns the job id."""
    output = cluster.run("sbatch --parsable", input=script)
    return output.strip().splitlines()[-1]


def submit_experiment(
    cluster: Cluster, resources: Resources, experiment: Experiment, commit: str, *, dry_run: bool
) -> Sequence[Tuple[Job, str]]:
    """Prepare the group, the checkout and the log directory, then submit every job. Returns (job, job id) pairs."""
    jobs = experiment.jobs()
    scripts = [job_script(cluster, resources, experiment, job, commit) for job in jobs]
    if dry_run:
        for script in scripts:
            print(script)
        return []
    prepare_group(experiment)
    ensure_checkout(cluster, commit)
    cluster.run(f"mkdir -p {shlex.quote(cluster.log_directory(experiment.group))}")
    submitted = []
    for job, script in zip(jobs, scripts):
        job_id = submit(cluster, script)
        print(f"submitted {job.label} as job {job_id}")
        submitted.append((job, job_id))
    return submitted


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("definition", type=pathlib.Path, help="Python file exporting EXPERIMENT")
    parser.add_argument("--partition", required=True, help="SLURM partition, e.g. gpu_a100 or gpu_h100")
    parser.add_argument("--time", required=True, help="wall-time limit per job, HH:MM:SS")
    parser.add_argument("--host", default="snellius", help="ssh host alias of the cluster")
    parser.add_argument("--dry-run", action="store_true", help="print the job scripts; submit nothing, create nothing")
    arguments = parser.parse_args()

    repository = pathlib.Path(__file__).resolve().parents[2]
    experiment = load_experiment(arguments.definition)
    commit = pushed_commit(repository)
    cluster = Cluster.connect(arguments.host)
    resources = Resources(partition=arguments.partition, time=arguments.time)
    print(f"{experiment.group}: {len(experiment.jobs())} jobs at commit {commit} on {cluster.host} ({resources.partition}, {resources.time})")
    submit_experiment(cluster, resources, experiment, commit, dry_run=arguments.dry_run)


if __name__ == "__main__":
    main()
