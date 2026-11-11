"""A trained policy is scored on every cell of its evaluation protocol, in one rollout.

Run with: JAX_PLATFORMS=cpu pytest tests/test_score.py -v  (about two minutes)
"""
from __future__ import annotations

import argparse
import csv
import pathlib

import numpy as np

from crax import envs
from training import contexts, replay, score
from training.agents.ppo_lag.train import train as train_ppo_lag

ENV_NAME = "safe_goal_point"
TINY = dict(num_envs=8, batch_size=8, num_minibatches=2, unroll_length=5, num_updates_per_batch=1, episode_length=20,
            num_eval_envs=4, num_evals=2, learning_rate=1e-3, normalize_observations=True)
NUM_TIMESTEPS = 8 * 5 * 2 * 4


def test_a_checkpointed_policy_is_scored_on_every_cell_of_its_protocol(tmp_path: pathlib.Path):
    """A tiny run under goal-point's hazard-total protocol: one score per hazard total, each
    from episodes with exactly that total, labelled with the region the total lies in; the
    table carries the run's facts and the cell's coordinate as columns."""
    learner_spec = contexts.Learner(constraint="additive", policy_context="told")
    setup = contexts.context_training_setup(
        ENV_NAME, "uniform", "level:3", learner_spec, protocol_name=contexts.HAZARD_TOTAL_PROTOCOL, reference=False,
        environment_overrides={}, num_timesteps=NUM_TIMESTEPS, num_envs=8, batch_size=8, unroll_length=5, num_minibatches=2,
    )
    train_ppo_lag(environment=envs.get_environment(ENV_NAME, level=1), num_timesteps=NUM_TIMESTEPS, seed=0,
                  progress_fn=lambda *_: None, safety_bound=25.0, initial_lambda_lagr=0.1, **TINY, **setup.train_kwargs(),
                  save_checkpoint_path=str(tmp_path / "checkpoints"))
    run_config = argparse.Namespace(
        env_name=ENV_NAME, difficulty=1, env_kwargs={}, episode_length=TINY["episode_length"], safety_bound=0.5,
        constraint=learner_spec.constraint, policy_context=learner_spec.policy_context,
        evaluation_protocol=contexts.HAZARD_TOTAL_PROTOCOL, reference=False, seed=0, **{
            key: setup.wandb_config()[key] for key in ("arm", "training_distribution")
        },
    )
    checkpoint_directory = sorted(p for p in (tmp_path / "checkpoints").iterdir() if p.is_dir())[-1]
    learner = replay.learner_from_checkpoint(checkpoint_directory, run_config)

    scores = score.score_cells(run_config, learner, deterministic=False, episodes=3, seed=0)
    assert [dict(s.cell.coordinates)["hazard_total"] for s in scores] == [float(total) for total in range(21)]
    assert [s.region for s in scores] == ["training"] * 11 + ["interpolation"] * 3 + ["training"] * 3 + ["extrapolation"] * 4
    assert all(s.episodes == 3 and np.isfinite([s.return_mean, s.cost_mean, s.cost_median]).all() for s in scores)
    assert all(0.0 <= s.share_over_budget <= 1.0 and s.share_outside_arena == 0.0 for s in scores)
    assert scores[0].cost_mean == 0.0 and scores[0].share_over_budget == 0.0, "no hazard, no cost"
    again = score.score_cells(run_config, learner, deterministic=False, episodes=3, seed=0)
    assert [s.cost_mean for s in again] == [s.cost_mean for s in scores], "the same seed scores the same contexts with the same keys"

    path = tmp_path / "scores" / "tiny_run.csv"
    score.write_scores(path, "tiny_run", run_config, int(checkpoint_directory.name), scores)
    with path.open() as file:
        rows = list(csv.DictReader(file))
    assert len(rows) == 21 and rows[12]["region"] == "interpolation" and float(rows[12]["hazard_total"]) == 12.0
    assert rows[0]["run"] == "tiny_run" and rows[0]["arm"] == "uniform additive told" and rows[0]["reference"] == "False"
    assert rows[0]["evaluation_protocol"] == contexts.HAZARD_TOTAL_PROTOCOL and float(rows[0]["budget"]) == 0.5
    assert {"env_name", "constraint", "policy_context", "seed", "environment_steps", "return_mean", "cost_median", "share_over_budget"} <= set(rows[0])


def test_assigned_contexts_give_slot_i_context_i_in_every_episode():
    """What the scoring rests on: the evaluation stack under :class:`AssignedContexts` runs slot
    ``i`` in ``contexts[i]``, also after its episode has ended and restarted."""
    import jax

    suite = contexts.suite_contexts(ENV_NAME)
    assigned = suite.space.sample_uniform(jax.random.PRNGKey(0), 4)
    env = contexts.wrap_for_context_evaluation(
        envs.get_environment(ENV_NAME, level=1), contexts.AssignedContexts(suite.space, assigned), episode_length=3
    )
    state = jax.jit(env.reset)(jax.random.split(jax.random.PRNGKey(1), 4))
    step = jax.jit(env.step)
    for _ in range(5):  # past the end of the first episode
        np.testing.assert_array_equal(np.asarray(state.info[contexts.CONTEXT_KEY]), np.asarray(assigned))
        state = step(state, np.zeros((4, env.action_size), np.float32))
    import pytest
    with pytest.raises(ValueError, match="one per slot"):
        env.reset(jax.random.split(jax.random.PRNGKey(1), 3))
    with pytest.raises(ValueError, match="outside the space"):
        contexts.AssignedContexts(suite.protocol(contexts.HAZARD_TOTAL_PROTOCOL).contexts, assigned)
