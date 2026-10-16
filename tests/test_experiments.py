"""The experiment definition expands to the right jobs, refuses what cannot form one group,
and the Snellius submitter only ever submits a pushed commit. CPU only; no W&B, no ssh."""
from __future__ import annotations

import pathlib
import subprocess

import pytest

from training.experiments import Arm, Experiment, load_experiment, run_facts
from training.experiments import snellius

REPOSITORY = pathlib.Path(__file__).resolve().parents[1]

COMMON = (
    "--env_name", "safe_goal_point", "--alg", "ppo_lag", "--deployment_distribution", "level:3",
    "--num_envs", "64", "--num_timesteps", "2_560", "--num_evals", "2", "--episode_length", "100",
    "--unroll_length", "5", "--batch_size", "32", "--num_minibatches", "4",
)
UNIFORM = Arm("uniform", ("--context_distribution", "uniform"))
STAGED = Arm("staged123", ("--context_distribution", "staged:1,2,3"))


def test_an_experiment_expands_to_one_job_per_arm_and_seed_with_the_full_argument_list():
    experiment = Experiment(group="g", common=COMMON, arms=(UNIFORM, STAGED), seeds=(0, 7))
    jobs = experiment.jobs()
    assert [job.label for job in jobs] == ["uniform_seed0", "uniform_seed7", "staged123_seed0", "staged123_seed7"]
    assert jobs[1].argv == (*COMMON, "--context_distribution", "uniform", "--wandb_group", "g", "--seeds", "7")
    assert jobs[1].arguments.seeds == [7] and jobs[1].arguments.wandb_group == "g"
    assert jobs[3].arguments.context_distribution == "staged:1,2,3"
    assert [job.label for job in Experiment(group="g", common=COMMON, arms=(UNIFORM,)).jobs()] == ["uniform"]


def test_a_malformed_flag_fails_at_load_time():
    with pytest.raises(SystemExit):
        Experiment(group="g", common=(*COMMON, "--no_such_flag"), arms=(UNIFORM,)).jobs()


@pytest.mark.parametrize(
    "kwargs, message",
    [
        (dict(common=(*COMMON, "--wandb_group", "x"), arms=(UNIFORM,)), "--wandb_group"),
        (dict(common=COMMON, arms=(Arm("a", ("--seeds", "1")),)), "--seeds"),
        (dict(common=COMMON, arms=(UNIFORM, Arm("uniform", ()))), "unique"),
        (dict(common=COMMON, arms=()), "no arms"),
        (dict(common=COMMON, arms=(UNIFORM,), seeds=()), "no seeds"),
    ],
)
def test_the_definition_refuses_what_the_launchers_set_themselves(kwargs, message):
    with pytest.raises(ValueError, match=message):
        Experiment(group="g", **kwargs)


def test_labels_name_files_and_jobs_so_they_are_restricted():
    with pytest.raises(ValueError, match="label"):
        Arm("staged:1,2,3", ())
    with pytest.raises(ValueError, match="group"):
        Experiment(group="a/b", common=COMMON, arms=(UNIFORM,))


def test_the_facts_are_those_of_the_dashboard_and_all_arms_must_share_them():
    experiment = Experiment(group="g", common=COMMON, arms=(UNIFORM, STAGED))
    facts = experiment.facts()
    assert facts.num_rounds == 4 and facts.environment_steps_per_round == 640 and facts.episode_length == 100
    assert [name for name, _ in facts.evaluations][:2] == ["deployment", "uniform"]

    longer = Arm("longer", ("--context_distribution", "uniform", "--num_timesteps", "5_120"))
    with pytest.raises(ValueError, match="differ in the facts"):
        Experiment(group="g", common=COMMON, arms=(UNIFORM, longer)).facts()


def test_facts_need_what_the_run_would_otherwise_read_off_the_environment():
    without_length = tuple(flag for flag in COMMON if flag not in ("--episode_length", "100"))
    with pytest.raises(ValueError, match="--episode_length"):
        run_facts(Experiment(group="g", common=without_length, arms=(UNIFORM,)).jobs()[0].arguments)
    with pytest.raises(ValueError, match="context_distribution"):
        run_facts(Experiment(group="g", common=COMMON, arms=(Arm("none", ("--context_distribution", "none")),)).jobs()[0].arguments)


@pytest.mark.parametrize("name", ["snellius_smoke", "goal_point_learners_500M"])
def test_the_committed_definitions_load_and_their_arms_share_facts(name: str):
    experiment = load_experiment(REPOSITORY / "scripts" / "experiments" / f"{name}.py")
    assert len(experiment.jobs()) >= 2
    experiment.facts()


def test_load_experiment_names_what_is_wrong(tmp_path: pathlib.Path):
    with pytest.raises(FileNotFoundError):
        load_experiment(tmp_path / "missing.py")
    (tmp_path / "no_export.py").write_text("x = 1\n")
    with pytest.raises(AttributeError, match="EXPERIMENT"):
        load_experiment(tmp_path / "no_export.py")
    (tmp_path / "wrong_type.py").write_text("EXPERIMENT = 'not an experiment'\n")
    with pytest.raises(TypeError, match="Experiment"):
        load_experiment(tmp_path / "wrong_type.py")


def test_the_job_script_runs_the_job_from_the_commit_checkout_with_the_exact_arguments():
    cluster = snellius.Cluster(host="snellius", home="/home/user")
    experiment = Experiment(group="g", common=COMMON, arms=(STAGED,))
    script = snellius.job_script(cluster, snellius.Resources("gpu_a100", "01:00:00"), experiment, experiment.jobs()[0], "abc123")
    lines = script.splitlines()
    assert lines[0] == "#!/usr/bin/env bash"
    assert "#SBATCH --job-name=g__staged123" in lines
    assert "#SBATCH --partition=gpu_a100" in lines and "#SBATCH --time=01:00:00" in lines and "#SBATCH --gpus=1" in lines
    assert "#SBATCH --output=/home/user/crax-logs/g/staged123_%j.out" in lines
    assert "cd /home/user/crax-runs/abc123" in lines and "export PYTHONPATH=/home/user/crax-runs/abc123" in lines
    assert lines[-1].startswith("exec /home/user/crax-venv/bin/python -m training.train_env ")
    assert lines[-1].endswith("--context_distribution staged:1,2,3 --wandb_group g --seeds 0")


def _git(repository: pathlib.Path, *arguments: str) -> str:
    return subprocess.run(["git", "-C", str(repository), *arguments], capture_output=True, text=True, check=True).stdout


def test_only_a_clean_pushed_commit_is_submitted(tmp_path: pathlib.Path):
    remote, local = tmp_path / "remote.git", tmp_path / "local"
    subprocess.run(["git", "init", "--quiet", "--bare", str(remote)], check=True)
    subprocess.run(["git", "init", "--quiet", "--initial-branch=main", str(local)], check=True)
    _git(local, "config", "user.email", "t@t"); _git(local, "config", "user.name", "t")
    _git(local, "remote", "add", "origin", str(remote))
    (local / "a.txt").write_text("a\n")
    _git(local, "add", "a.txt"); _git(local, "commit", "--quiet", "-m", "first")

    with pytest.raises(RuntimeError, match="no remote branch"):
        snellius.pushed_commit(local)
    _git(local, "push", "--quiet", "-u", "origin", "main")
    assert snellius.pushed_commit(local) == _git(local, "rev-parse", "--short=12", "HEAD").strip()

    (local / "b.txt").write_text("b\n")  # untracked counts: it would not be in the commit the job runs
    with pytest.raises(RuntimeError, match="b.txt"):
        snellius.pushed_commit(local)
