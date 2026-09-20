"""Tests for the trainer wiring of context distributions (task board T1–T3).

Unit tests cover the round schedule, round data → episode feedback, and the
realised-curriculum summaries. The end-to-end tests run a tiny PPO-Lagrange
training on ``safe_velocity_ant`` on CPU with each distribution and check that

* one compiled call is one round and the hook runs after every round,
* the staged curriculum switches levels at the planned rounds without a
  recompile of the epoch program,
* the policy is evaluated on both the deployment context w and the uniform
  reference r,
* the realised curriculum q̂ is reported next to the intended one.

Run with: JAX_PLATFORMS=cpu pytest tests/test_context_training.py -v
"""
from __future__ import annotations

from typing import Any, Dict, List, Tuple

import jax
import numpy as np
import pytest

from crax import envs
from training import contexts as C
from training.contexts.realised_curriculum import NUM_BINS
from training.agents.ppo_lag.train import train as train_ppo_lag

ENV_NAME = "safe_velocity_ant"


# --------------------------------------------------------------------------- #
# Unit: schedule, round data, realised curriculum
# --------------------------------------------------------------------------- #


def test_distribution_specs_parse_and_reject_garbage():
    suite = C.suite_contexts(ENV_NAME)
    assert isinstance(C.parse_distribution("uniform", suite, 10), C.UniformDistribution)
    fixed = C.parse_distribution("level:2", suite, 10)
    assert isinstance(fixed, C.FixedContext) and float(fixed.context[0]) == float(suite.level(2)[0])
    staged = C.parse_distribution("staged:1,3", suite, 10)
    assert isinstance(staged, C.StagedContexts) and list(staged.switch_rounds) == [5]
    for bad in ("level", "uniform:2", "staged:", "gaussian"):
        with pytest.raises(ValueError):
            C.parse_distribution(bad, suite, 10)
    assert C.spec_label("staged:1,2,3") == "staged123"


def _synthetic_round(num_transitions: int = 10) -> C.RoundRollout:
    contexts = np.linspace(1.0, 2.0, num_transitions, dtype=np.float32)[:, None]
    done = np.zeros(num_transitions, dtype=bool)
    done[[2, 7]] = True
    metrics = {
        "sum_reward": np.arange(num_transitions, dtype=np.float32),
        "cost": np.full(num_transitions, 3.0, dtype=np.float32),
        "length": np.full(num_transitions, 25.0, dtype=np.float32),
    }
    recorded = {  # as the trainer records them: [rows, unroll, ...]
        C.TRANSITION_CONTEXT_KEY: contexts.reshape(2, 5, 1),
        "episode_done": done.reshape(2, 5).astype(np.float32),
        "episode_metrics": {k: v.reshape(2, 5) for k, v in metrics.items()},
    }
    return C.RoundRollout.from_recorded_fields(recorded, round_index=4)


def test_rollout_flattens_and_extracts_completed_episodes():
    data = _synthetic_round()
    assert data.num_transitions == 10
    assert data.num_completed_episodes == 2
    feedback = C.completed_episodes(data)
    assert feedback.num_completed == 2
    assert feedback.round_index == 4
    np.testing.assert_allclose(feedback.returns, [2.0, 7.0])
    np.testing.assert_allclose(feedback.costs, [3.0, 3.0])
    np.testing.assert_allclose(feedback.lengths, [25.0, 25.0])
    np.testing.assert_allclose(feedback.contexts[:, 0], data.contexts[[2, 7], 0])
    assert bool(np.all(feedback.safe(cost_threshold=5.0)))


def test_curriculum_metrics_give_sampled_and_experienced_distributions():
    import wandb

    suite = C.suite_contexts(ENV_NAME)
    data = _synthetic_round()
    metrics = C.curriculum_metrics(suite.space, data)
    for name in ("sampled", "experienced"):
        assert isinstance(metrics[f"{name}/velocity_threshold"], wandb.Histogram)
        mass = [metrics[f"{name}/velocity_threshold/bin_{i:02d}"] for i in BINS]
        assert sum(mass) == pytest.approx(1.0)
    assert metrics["num_transitions"] == 10
    assert metrics["num_completed_episodes"] == 2
    assert metrics["experienced/velocity_threshold/mean"] == pytest.approx(1.5)  # all 10 transitions
    assert metrics["sampled/velocity_threshold/mean"] == pytest.approx(1.5)  # episodes at 1.22 and 1.78
    lengths = [metrics[f"episode_length/velocity_threshold/bin_{i:02d}"] for i in BINS]
    assert sorted(l for l in lengths if not np.isnan(l)) == [25.0, 25.0]


# --------------------------------------------------------------------------- #
# End to end on CPU: tiny PPO-Lagrange run per distribution
# --------------------------------------------------------------------------- #

TINY = dict(
    num_envs=8,
    batch_size=8,
    num_minibatches=2,
    unroll_length=5,
    num_updates_per_batch=1,
    episode_length=20,
    num_eval_envs=4,
    num_evals=3,  # initial + 2 -> 2 evaluation blocks
    learning_rate=1e-3,
    normalize_observations=True,
    log_training_metrics=False,
)
STEPS_PER_ROUND = TINY["batch_size"] * TINY["unroll_length"] * TINY["num_minibatches"]  # 80
TOTAL_ROUNDS = 6
NUM_TIMESTEPS = STEPS_PER_ROUND * TOTAL_ROUNDS


BINS = range(NUM_BINS)


def _run(training_spec: str) -> Tuple[C.ContextTrainingSetup, List[Tuple[int, Dict[str, Any]]], List[float]]:
    setup = C.context_training_setup(
        ENV_NAME,
        training_spec,
        "level:3",
        num_timesteps=NUM_TIMESTEPS,
        batch_size=TINY["batch_size"],
        unroll_length=TINY["unroll_length"],
        num_minibatches=TINY["num_minibatches"],
    )
    assert setup is not None
    assert setup.total_rounds == TOTAL_ROUNDS

    from jax._src import monitoring
    from jax._src.dispatch import BACKEND_COMPILE_EVENT

    compile_seconds: List[float] = []
    monitoring.register_event_duration_secs_listener(
        lambda event, seconds, **_: compile_seconds.append(seconds) if event == BACKEND_COMPILE_EVENT else None
    )

    history: List[Tuple[int, Dict[str, Any]]] = []
    env = envs.get_environment(ENV_NAME, level=1)
    train_ppo_lag(
        environment=env,
        num_timesteps=NUM_TIMESTEPS,
        seed=0,
        progress_fn=lambda step, metrics: history.append((step, dict(metrics))),
        safety_bound=25.0,
        **TINY,
        **setup.train_kwargs(),
    )
    return setup, history, compile_seconds


def _round_entries(history):
    return [(step, m) for step, m in history if "curriculum/round" in m]


@pytest.mark.parametrize("training_spec", ["uniform", "staged:1,2,3", "level:1"])
def test_end_to_end_one_round_per_call_with_both_evaluations(training_spec):
    setup, history, _ = _run(training_spec)

    round_entries = _round_entries(history)
    assert [int(m["curriculum/round"]) for _, m in round_entries] == list(range(TOTAL_ROUNDS))
    assert [step for step, _ in round_entries] == [STEPS_PER_ROUND * (k + 1) for k in range(TOTAL_ROUNDS)]

    # q̂ logged every round as a distribution over fixed bins
    for _, metrics in round_entries:
        mass = [metrics[f"curriculum/experienced/velocity_threshold/bin_{i:02d}"] for i in BINS]
        assert sum(mass) == pytest.approx(1.0)
        assert metrics["curriculum/num_transitions"] == STEPS_PER_ROUND

    # evaluation on w and r, at the initial eval and after each of the 2 blocks
    evaluation_entries = [(s, m) for s, m in history if f"eval/{C.DEPLOYMENT_EVALUATION}/episode_reward" in m]
    assert [s for s, _ in evaluation_entries] == [0, 3 * STEPS_PER_ROUND, 6 * STEPS_PER_ROUND]
    for _, metrics in evaluation_entries:
        assert f"eval/{C.UNIFORM_EVALUATION}/episode_reward" in metrics
        assert f"eval/{C.DEPLOYMENT_EVALUATION}/episode_cost" in metrics
        assert not any(key.startswith("eval/episode_") for key in metrics), "unnamed eval metrics leaked"


def test_end_to_end_staged_switches_levels_on_schedule_without_recompiling():
    setup, history, compile_seconds = _run("staged:1,2,3")
    suite = setup.suite
    staged = setup.distribution
    assert isinstance(staged, C.StagedContexts)
    assert list(staged.switch_rounds) == [2, 4]

    round_entries = _round_entries(history)
    intended = [m["curriculum/intended/context/velocity_threshold"] for _, m in round_entries]
    # round k's intended context is the one in force during round k: L1,L1,L2,L2,L3,L3
    expected = [float(suite.level(l)[0]) for l in (1, 1, 2, 2, 3, 3)]
    assert intended == pytest.approx(expected)

    # The realised curriculum lags the intended one (intended_vs_realised_curriculum.md):
    # episodes outlast a round, so the first round of a new stage still trains on
    # transitions from episodes started under the previous stage. q̂ therefore
    # moves monotonically from L1 towards L3 but sits *between* stages at each switch.
    # A context is drawn when an episode *ends*, under the φ of that round, so the
    # first round of a stage may consist entirely of previous-stage data.
    level_1, level_3 = float(suite.level(1)[0]), float(suite.level(3)[0])
    realised_mean = [m["curriculum/experienced/velocity_threshold/mean"] for _, m in round_entries]
    assert realised_mean[0] == pytest.approx(level_1)
    assert all(level_3 <= mean <= level_1 for mean in realised_mean)
    # thresholds decrease with level: realised never runs *ahead* of intended ...
    assert all(mean >= q - 1e-5 for mean, q in zip(realised_mean, intended))
    # ... is monotone, and does lag (q̂ != q somewhere after the first switch)
    assert all(a >= b - 1e-5 for a, b in zip(realised_mean, realised_mean[1:]))
    assert realised_mean[-1] < level_1
    assert any(mean > q + 1e-3 for mean, q in zip(realised_mean[2:], intended[2:]))

    # stage switches change data only: no substantial compile after the warm-up
    # (the first epoch/eval/reset programs). Anything > 1 s after those is a recompile.
    substantial = [s for s in compile_seconds if s > 1.0]
    assert len(substantial) <= 4, f"unexpected recompiles: {substantial}"


def test_stock_trainer_path_is_unchanged_without_a_hook():
    """No round hook, no named evaluators: the benchmark behaves as before."""
    history: List[Tuple[int, Dict[str, Any]]] = []
    env = envs.get_environment(ENV_NAME, level=1)
    train_ppo_lag(
        environment=env,
        num_timesteps=NUM_TIMESTEPS,
        seed=0,
        progress_fn=lambda step, metrics: history.append((step, dict(metrics))),
        safety_bound=25.0,
        **TINY,
    )
    steps = [step for step, _ in history]
    # 2 evaluation blocks of one epoch each (3 rounds per epoch), initial eval at 0
    assert steps[0] == 0 and steps[-1] == NUM_TIMESTEPS
    assert not any("curriculum/round" in m for _, m in history)
    final = history[-1][1]
    assert "eval/episode_reward" in final and "eval/episode_cost" in final
    assert not any(key.startswith(f"eval/{C.DEPLOYMENT_EVALUATION}/") for key in final)


def test_end_to_end_uniform_realised_is_not_a_point_mass():
    _, history, _ = _run("uniform")
    last = _round_entries(history)[-1][1]
    assert last["curriculum/experienced/velocity_threshold/std"] > 0.05
    occupied = sum(1 for i in BINS if last[f"curriculum/experienced/velocity_threshold/bin_{i:02d}"] > 0)
    assert occupied >= 3
