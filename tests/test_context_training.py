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
from training.contexts.training_curriculum import NUM_BINS, SECTION as CURRICULUM
from training.agents.ppo_lag.train import train as train_ppo_lag
from training.contexts.distributions import PrioritizedLevelReplayParameters
from training.rounds import LearningSignals
from crax.envs.base import State

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
    assert isinstance(C.parse_distribution("plr", suite, 10), C.PrioritizedLevelReplay)
    for bad in ("level", "uniform:2", "staged:", "gaussian", "plr:0.5"):
        with pytest.raises(ValueError):
            C.parse_distribution(bad, suite, 10)
    assert C.spec_label("staged:1,2,3") == "staged123"


def _synthetic_round() -> C.RoundRollout:
    """Two slots, five steps, one unroll. Each slot ends an episode on step 2 and starts a new
    one; the advantage is −1 in the lower half of Ω's used range and +2 in the upper half."""
    contexts = np.asarray([[1.28, 1.28, 1.28, 1.4, 1.4], [1.72, 1.72, 1.72, 1.6, 1.6]], np.float32)[..., None]  # inside Ω = [≈1.05, ≈2.62]
    done = np.zeros((2, 5), np.float32)
    done[:, 2] = 1.0
    recorded = {  # as the trainer records them: [num_unrolls × num_slots, unroll_length, ...]
        C.TRANSITION_CONTEXT_KEY: contexts,
        C.SLOT_INDEX_KEY: np.array([[0] * 5, [1] * 5], np.int32),
        "episode_done": done,
        "episode_metrics": {
            "sum_reward": np.arange(10, dtype=np.float32).reshape(2, 5),
            "cost": np.full((2, 5), 3.0, np.float32),
            "length": np.full((2, 5), 25.0, np.float32),
        },
    }
    signals = LearningSignals(reward_advantage=np.where(contexts[..., 0] < 1.5, -1.0, 2.0).astype(np.float32))
    transitions = C.Transitions.from_recorded_fields(recorded, signals, num_slots=2)
    return C.RoundRollout(4, transitions, C.EpisodeTracker(num_slots=2).complete(transitions))


def test_rollout_has_the_transitions_per_slot_and_the_episodes_that_ended():
    data = _synthetic_round()
    assert data.round_index == 4
    transitions, episodes = data.transitions, data.completed_episodes
    assert (transitions.num_slots, transitions.steps_per_slot, transitions.num_transitions) == (2, 5, 10)
    np.testing.assert_allclose(transitions.reward_advantage, [[-1.0] * 5, [2.0] * 5])
    assert episodes.count == 2
    np.testing.assert_allclose(episodes.contexts[:, 0], [1.28, 1.72])
    np.testing.assert_allclose(episodes.returns, [2.0, 7.0])  # the recorded totals at the done transitions
    np.testing.assert_allclose(episodes.costs, [3.0, 3.0])
    np.testing.assert_allclose(episodes.lengths, [25.0, 25.0])
    np.testing.assert_allclose(episodes.value_loss, [1.0, 2.0])  # mean |advantage| over the episode's three steps
    with pytest.raises(ValueError, match="reward_advantage"):
        C.Transitions(transitions.contexts, transitions.episode_done, transitions.episode_return, transitions.episode_cost,
                      transitions.episode_length, transitions.reward_advantage[:, :3])


def _histogram_masses(histogram) -> np.ndarray:
    return np.asarray(histogram.histogram, dtype=float)


def test_training_curriculum_metrics_give_intended_sampled_experienced_and_value_loss():
    import wandb

    suite = C.suite_contexts(ENV_NAME)
    distribution = C.parse_distribution("staged:1,2,3", suite, total_rounds=6)
    data = _synthetic_round()
    metrics = C.training_curriculum_metrics(distribution, distribution.initialise(), data)
    assert all(key.startswith(CURRICULUM) for key in metrics)
    for name in ("intended", "sampled", "experienced"):
        histogram = metrics[f"{CURRICULUM}{name}/velocity_threshold"]
        assert isinstance(histogram, wandb.Histogram)
        assert len(histogram.histogram) == NUM_BINS
        assert _histogram_masses(histogram).sum() == pytest.approx(1.0)
    # intended: the stage-0 point mass, shown by sampling it
    assert metrics[CURRICULUM + "intended/velocity_threshold/mean"] == pytest.approx(float(suite.level(1)[0]))
    assert metrics[CURRICULUM + "intended/velocity_threshold/std"] == 0.0
    assert int((_histogram_masses(metrics[CURRICULUM + "intended/velocity_threshold"]) > 0).sum()) == 1
    assert metrics[CURRICULUM + "distribution/stage"] == 0
    assert metrics[CURRICULUM + "num_transitions"] == 10
    assert metrics[CURRICULUM + "num_completed_episodes"] == 2
    assert metrics[CURRICULUM + "experienced/velocity_threshold/mean"] == pytest.approx(1.5)  # all 10 transitions
    assert metrics[CURRICULUM + "sampled/velocity_threshold/mean"] == pytest.approx(1.5)  # episodes at 1.28 and 1.72
    lengths = _histogram_masses(metrics[CURRICULUM + "episode_length/velocity_threshold"])
    assert sorted(lengths[lengths > 0]) == [25.0, 25.0]  # 0 where no episode ended
    # value_loss: mean |advantage| of the transitions in each bin (|−1| below 1.5, 2 above), 0 in empty bins
    value_loss = _histogram_masses(metrics[CURRICULUM + "value_loss/velocity_threshold"])
    edges = np.linspace(float(suite.space.low[0]), float(suite.space.high[0]), NUM_BINS + 1)
    flat_contexts, flat_advantage = data.transitions.flat_contexts[:, 0], data.transitions.flat_reward_advantage
    bins = np.digitize(flat_contexts, edges) - 1
    expected = [np.abs(flat_advantage[bins == b]).mean() if np.any(bins == b) else 0.0 for b in range(NUM_BINS)]
    np.testing.assert_allclose(value_loss, expected)
    assert 0.0 in value_loss and 1.0 in value_loss and 2.0 in value_loss
    assert not any("bin_" in key or "completed_episodes/" in key for key in metrics)


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
)
STEPS_PER_ROUND = TINY["batch_size"] * TINY["unroll_length"] * TINY["num_minibatches"]  # 80
TOTAL_ROUNDS = 6
NUM_TIMESTEPS = STEPS_PER_ROUND * TOTAL_ROUNDS


class _RecordingHook(C.ContextRoundHook):
    """The hook, remembering the environment state it hands back after the last round
    (φ lives there, not on the hook)."""

    final_env_state: State

    def on_round_end(self, round_index: int, env_state: State) -> Tuple[State, Dict[str, Any]]:
        self.final_env_state, metrics = super().on_round_end(round_index, env_state)
        return self.final_env_state, metrics


def _run(training_spec: str) -> Tuple[C.ContextTrainingSetup, List[Tuple[int, Dict[str, Any]]], List[float], _RecordingHook]:
    setup = C.context_training_setup(
        ENV_NAME,
        training_spec,
        "level:3",
        num_timesteps=NUM_TIMESTEPS,
        num_envs=TINY["num_envs"],
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
    train_kwargs = setup.train_kwargs()
    hook = _RecordingHook(setup.distribution, num_slots=setup.num_envs)
    train_kwargs["round_hook"] = hook
    train_ppo_lag(
        environment=env,
        num_timesteps=NUM_TIMESTEPS,
        seed=0,
        progress_fn=lambda step, metrics: history.append((step, dict(metrics))),
        safety_bound=25.0,
        **TINY,
        **train_kwargs,
    )
    return setup, history, compile_seconds, hook


EXPERIENCED = CURRICULUM + "experienced/velocity_threshold"


def _round_entries(history):
    return [(step, m) for step, m in history if EXPERIENCED in m]


@pytest.mark.parametrize("training_spec", ["uniform", "staged:1,2,3", "level:1", "plr"])
def test_end_to_end_one_round_per_call_with_both_evaluations(training_spec):
    setup, history, _, _ = _run(training_spec)

    round_entries = _round_entries(history)
    assert [step for step, _ in round_entries] == [STEPS_PER_ROUND * (k + 1) for k in range(TOTAL_ROUNDS)]

    # q̂ logged every round as a distribution over fixed bins
    for _, metrics in round_entries:
        assert _histogram_masses(metrics[EXPERIENCED]).sum() == pytest.approx(1.0)
        assert metrics[CURRICULUM + "num_transitions"] == STEPS_PER_ROUND

    # episodes end in (almost) every round once the first ones have run their length
    rounds_with_completed_episodes = [step for step, m in round_entries if m[CURRICULUM + "num_completed_episodes"] > 0]
    assert len(rounds_with_completed_episodes) >= TOTAL_ROUNDS - 1
    # The stock logger's `episodic/*` keys are not compared against ours: its two
    # `jax.debug.callback`s share unlocked state and race (README, Known defects).

    # evaluation on w and r, at the initial eval and after each of the 2 blocks
    evaluation_entries = [(s, m) for s, m in history if f"evaluation/{C.DEPLOYMENT_EVALUATION}/episode_reward" in m]
    assert [s for s, _ in evaluation_entries] == [0, 3 * STEPS_PER_ROUND, 6 * STEPS_PER_ROUND]
    for _, metrics in evaluation_entries:
        assert f"evaluation/{C.UNIFORM_EVALUATION}/episode_reward" in metrics
        assert f"evaluation/{C.DEPLOYMENT_EVALUATION}/episode_cost" in metrics
        assert not any(key.startswith("eval/") for key in metrics), "unnamed eval metrics leaked"


def test_end_to_end_staged_switches_levels_on_schedule_without_recompiling():
    setup, history, compile_seconds, _ = _run("staged:1,2,3")
    suite = setup.suite
    staged = setup.distribution
    assert isinstance(staged, C.StagedContexts)
    assert list(staged.switch_rounds) == [2, 4]

    round_entries = _round_entries(history)
    intended = [m[CURRICULUM + "intended/velocity_threshold/mean"] for _, m in round_entries]
    # round k's intended context is the one in force during round k: L1,L1,L2,L2,L3,L3
    expected = [float(suite.level(l)[0]) for l in (1, 1, 2, 2, 3, 3)]
    assert intended == pytest.approx(expected)
    assert [m[CURRICULUM + "distribution/stage"] for _, m in round_entries] == [0, 0, 1, 1, 2, 2]

    # The realised curriculum lags the intended one (intended_vs_realised_curriculum.md):
    # episodes outlast a round, so the first round of a new stage still trains on
    # transitions from episodes started under the previous stage. q̂ therefore
    # moves monotonically from L1 towards L3 but sits *between* stages at each switch.
    # A context is drawn when an episode *ends*, under the φ of that round, so the
    # first round of a stage may consist entirely of previous-stage data.
    level_1, level_3 = float(suite.level(1)[0]), float(suite.level(3)[0])
    realised_mean = [m[CURRICULUM + "experienced/velocity_threshold/mean"] for _, m in round_entries]
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
    assert not any(key.startswith(CURRICULUM) for _, m in history for key in m)
    final = history[-1][1]
    assert "eval/episode_reward" in final and "eval/episode_cost" in final
    assert not any(key.startswith("evaluation/") for key in final)


def test_end_to_end_uniform_realised_is_not_a_point_mass():
    _, history, _, _ = _run("uniform")
    last = _round_entries(history)[-1][1]
    assert last[EXPERIENCED + "/std"] > 0.05
    assert int((_histogram_masses(last[EXPERIENCED]) > 0).sum()) >= 3


def test_end_to_end_plr_fills_its_buffer_from_the_trainers_advantages_and_replays():
    """PLR on the real trainer: the first round draws from Uniform(Ω) only; a context enters
    the buffer only once an episode in it has *ended*, scored by the advantage the trainer
    shipped over that whole episode; the intended distribution is then the replay mixture."""
    setup, history, _, hook = _run("plr")
    plr = setup.distribution
    assert isinstance(plr, C.PrioritizedLevelReplay)
    round_entries = _round_entries(history)
    DISTRIBUTION = CURRICULUM + "distribution/"

    replay_probability = [m[DISTRIBUTION + "replay_probability"] for _, m in round_entries]
    occupancy = [m[DISTRIBUTION + "buffer_occupancy"] for _, m in round_entries]
    completed = [m[CURRICULUM + "num_completed_episodes"] for _, m in round_entries]
    assert replay_probability[0] == 0.0 and occupancy[0] == 0.0  # φ_0: nothing seen yet
    assert all(p == plr.replay_probability for p in replay_probability[1:])  # the paper's constant P_D from then on
    # every buffer row needed an episode to end: rows in φ_k ≤ episodes completed in rounds < k
    rows = [o * plr.buffer_size for o in occupancy]
    assert all(rows[k] <= sum(completed[:k]) + 1e-6 for k in range(len(rows)))
    assert all(a <= b + 1e-6 for a, b in zip(rows, rows[1:])) and rows[-1] > 0  # a 1000-row buffer only fills here
    # scores are the trainer's |reward advantage|: finite and positive once the buffer has rows
    scored = [m for _, m in round_entries if m[DISTRIBUTION + "buffer_occupancy"] > 0]
    assert scored
    assert all(np.isfinite(m[DISTRIBUTION + "score/mean"]) and m[DISTRIBUTION + "score/mean"] > 0 for m in scored)
    assert all(m[DISTRIBUTION + "score/max"] >= m[DISTRIBUTION + "score/mean"] for m in scored)
    # value_loss/<d>: mean |advantage| per bin -- only bins the round had transitions in can be
    # non-zero (a bin whose transitions all had advantage exactly 0 is legitimately 0)
    for _, m in round_entries:
        value_loss = _histogram_masses(m[CURRICULUM + "value_loss/velocity_threshold"])
        experienced = _histogram_masses(m[EXPERIENCED])
        assert np.all(experienced[value_loss > 0] > 0)
        assert np.any(value_loss > 0)
    # intended is a spread-out mixture, never a point mass
    assert all(m[CURRICULUM + "intended/velocity_threshold/std"] > 0.05 for _, m in round_entries)

    # the φ left in the environment state for the round after the last: rows inside Ω,
    # scored in rounds that happened
    final = C.current_parameters(hook.final_env_state)
    assert isinstance(final, PrioritizedLevelReplayParameters)
    occupied = np.asarray(final.occupied)
    assert int(final.round_index) == TOTAL_ROUNDS
    assert occupied.mean() >= occupancy[-1] > 0  # the buffer only fills
    assert np.all(np.asarray(setup.suite.space.contains(final.contexts[occupied])))
    last_round = np.asarray(final.last_round)[occupied]
    assert np.all((0 <= last_round) & (last_round < TOTAL_ROUNDS))
