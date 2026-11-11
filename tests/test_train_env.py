"""The entry point every experiment goes through (``python -m training.train_env``) runs end to
end for one tiny round on CPU, offline W&B, and leaves what a run must leave under its run
directory — episode records and a whole-learner checkpoint — handed to the run for upload.

The other tests call ``train()`` directly; only this one exercises the argument parsing, the
W&B set-up, the recorder and checkpoint wiring and the module's imports as a job on the
cluster does (the 2026-10-16 vision smoke failed on a missing import no other test reached).

Run with: JAX_PLATFORMS=cpu pytest tests/test_train_env.py -v  (about a minute)
"""
from __future__ import annotations

import os
import pathlib
import subprocess
import sys

import pytest

from training import checkpoint, contexts
from training.evaluation import RECORDS_DIRECTORY, records_under

TINY_RUN = (
    "--env_name", "safe_goal_point", "--alg", "ppo_lag",
    "--context_distribution", "uniform", "--deployment_distribution", "level:3",
    "--constraint", "additive", "--policy_context", "told",
    "--num_envs", "8", "--batch_size", "8", "--num_minibatches", "2", "--unroll_length", "5", "--num_updates_per_batch", "1",
    "--num_timesteps", "160", "--num_evals", "2", "--num_eval_envs", "4", "--episode_length", "20",
    "--record_episodes", "2", "--store_model", "true", "--skip_rollout", "--skip_video", "--quiet", "--seeds", "0",
)
# Arms of the generalization baselines (scripts/experiments/generalization.py): the by-kind surface
# lidar (78 sensors) under an evaluation protocol, which adds an evaluation target for the training
# contexts and one per held-out region to the deployment task and the protocol's contexts.
SURFACE_LIDAR = ("--env_kwargs", '{"hazard_lidar": "surface_by_kind"}')


@pytest.mark.parametrize("extra_flags, protocol_name, sensors, targets, run_name_fragment", [
    ((), "none", 62, 2, "_ctx_uniform_additive_told_ppo_lag_"),
    (SURFACE_LIDAR, "hazard_total", 78, 5, "_ctx_uniform_additive_told_hazard_lidar_surface_by_kind_ppo_lag_"),
    (SURFACE_LIDAR, "drag", 78, 5, "_ctx_uniform_additive_told_hazard_lidar_surface_by_kind_ppo_lag_"),
    ((*SURFACE_LIDAR, "--reference"), "combination", 78, 4, "_ctx_uniform_additive_told_reference_hazard_lidar_surface_by_kind_ppo_lag_"),
])
def test_the_entry_point_runs_a_tiny_context_run_and_leaves_records_and_a_whole_learner_checkpoint(
    tmp_path: pathlib.Path, extra_flags, protocol_name, sensors, targets, run_name_fragment
):
    environment = {
        **os.environ, "JAX_PLATFORMS": "cpu", "WANDB_MODE": "offline", "WANDB_DIR": str(tmp_path), "WANDB_SILENT": "true",
        "WANDB_API_KEY": os.environ.get("WANDB_API_KEY", "0" * 40),  # the login check reads a key; offline mode never uses it
        "PYTHONPATH": str(pathlib.Path(__file__).resolve().parents[1]),
    }
    completed = subprocess.run(
        [sys.executable, "-m", "training.train_env", *TINY_RUN, *extra_flags, "--evaluation_protocol", protocol_name],
        env=environment, capture_output=True, text=True, timeout=900,
    )
    assert completed.returncode == 0, completed.stderr[-4000:]

    run_directories = list((tmp_path / "wandb").glob("offline-run-*/files"))
    assert len(run_directories) == 1, run_directories
    files = run_directories[0]

    records = list(records_under(files / RECORDS_DIRECTORY))
    assert len(records) == 2 * targets * 2, f"2 episodes x {targets} evaluation targets x 2 evaluations"
    # every episode recorded on a region of the protocol ran in a context of that region
    protocol = contexts.suite_contexts("safe_goal_point").protocol(protocol_name)
    regions = {contexts.TRAINING_REGION: protocol.training, **{region.name: region.space for region in protocol.held_out}}
    evaluation_names = {record.evaluation_name for record in records}
    assert evaluation_names == {"deployment", "uniform", *(regions if protocol.held_out else ())}
    for record in records:
        if record.evaluation_name in regions:
            assert bool(regions[record.evaluation_name].contains(record.context[None])[0]), (record.evaluation_name, record.context)
        assert bool(protocol.contexts.contains(record.context[None])[0]), record.context
    assert all(record.observation.shape[1] == sensors for record in records), "the record holds the sensors the arm observed"
    assert all(run_name_fragment in record.run_name for record in records), records[0].run_name

    steps = sorted(path for path in (files / checkpoint.RUN_CHECKPOINTS_DIRECTORY).iterdir() if path.is_dir())
    assert steps, "no checkpoint written"
    assert (steps[-1] / checkpoint.LEARNER_STATE_DIRECTORY).is_dir(), "the checkpoint holds the policy only"
    assert len(checkpoint.load(steps[-1])) == 3, "(normaliser, policy, value)"

    # W&B uploads only the files it is handed; an offline run's transaction log names exactly those
    # (verified: a file written into the run directory but not handed does not appear). The vision
    # smoke of 2026-10-17 finished with every record on disk and none in W&B.
    handed = next(files.parent.glob("*.wandb")).read_bytes()
    for path in sorted((files / RECORDS_DIRECTORY).iterdir()):
        assert path.name.encode() in handed, f"{path.name} was written but not handed to the run"
    assert f"{steps[-1].name}/{checkpoint.LEARNER_STATE_DIRECTORY}/".encode() in handed, "the learner state was not handed to the run"
