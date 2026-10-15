"""Tests for the trainer wiring of context distributions (task board T1–T3).

Unit tests cover the round schedule, round data → episode feedback, and the
realised-curriculum summaries. The end-to-end tests run a tiny PPO-Lagrange
training on ``safe_velocity_ant`` on CPU with each distribution and check that

* one compiled call is one round and the hook runs after every round,
* the staged curriculum switches levels at the planned rounds without a
  recompile of the epoch program,
* the policy is evaluated on both the deployment context w and the uniform
  reference r,
* the realised curriculum q̂ is reported next to the intended one,
* the contextual learner (``docs/acl/design/contextual_learner.md``) runs on an
  Ω with integer dimensions (``safe_lift_ant``): λ(ω) per context, told policy.

Run with: JAX_PLATFORMS=cpu pytest tests/test_context_training.py -v
"""
from __future__ import annotations

import dataclasses
from typing import Any, Callable, Dict, List, Tuple

import jax
import numpy as np
import pytest

from crax import envs
from training import contexts as C
from training.contexts.training_curriculum import NUM_BINS, SECTION as CURRICULUM
from training import dashboard
from training.agents.ppo_lag.train import train as train_ppo_lag
from training.agents.ppo_pid.train import train as train_ppo_pid
from training.agents.ppo import train as ppo_train
from training.contexts.distributions import PrioritizedLevelReplayParameters
from training.rounds import LearningSignalsFunction, Signal
from crax.envs.base import State

ENV_NAME = "safe_velocity_ant"
REWARD, COST, LAGRANGIAN = Signal.REWARD_ADVANTAGE, Signal.COST_ADVANTAGE, Signal.LAGRANGIAN_ADVANTAGE


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
    assert C.parse_distribution("plr", suite, 10).score is REWARD
    assert C.parse_distribution("plr:cost", suite, 10).score is COST
    assert C.parse_distribution("plr:lagrangian", suite, 10).score is LAGRANGIAN
    for bad in ("level", "uniform:2", "staged:", "gaussian", "plr:0.5", "plr:return", "plr:REWARD_ADVANTAGE"):
        with pytest.raises(ValueError):
            C.parse_distribution(bad, suite, 10)
    assert C.spec_label("staged:1,2,3") == "staged123"


def test_setup_evaluates_on_every_region_the_suite_registers():
    """w and r always; then one evaluator per `EvaluationRegion`, named and labelled by the
    region itself. A suite with no regions gets exactly two."""
    common = dict(num_timesteps=1000, num_envs=4, batch_size=4, unroll_length=5, num_minibatches=2)
    velocity = C.context_training_setup(ENV_NAME, "uniform", "level:3", C.DEFAULT_LEARNER, **common)
    goal = C.context_training_setup("safe_goal_point", "uniform", "level:3", C.DEFAULT_LEARNER, **common)
    assert [e.name for e in velocity.evaluations] == [C.DEPLOYMENT_EVALUATION, C.UNIFORM_EVALUATION]
    assert [e.name for e in goal.evaluations] == [C.DEPLOYMENT_EVALUATION, C.UNIFORM_EVALUATION, C.OUT_OF_DISTRIBUTION_EVALUATION]
    assert set(goal.train_kwargs()["evaluation_wrap_env_fns"]) == {e.name for e in goal.evaluations}
    region = goal.suite.region(C.OUT_OF_DISTRIBUTION_EVALUATION)
    assert goal.wandb_config()["evaluations"][C.OUT_OF_DISTRIBUTION_EVALUATION] == region.label
    out_of_distribution = goal.evaluations[2].distribution
    assert isinstance(out_of_distribution, C.UniformDistribution) and out_of_distribution.space == region.space
    # a region may not shadow the two built-in evaluations
    clashing = dataclasses.replace(region, name=C.UNIFORM_EVALUATION)
    with pytest.raises(ValueError, match="collides"):
        C.evaluations_for(dataclasses.replace(goal.suite, evaluation_regions=(clashing,)), "level:3", total_rounds=1)


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
    reward_advantage = np.where(contexts[..., 0] < 1.5, -1.0, 2.0).astype(np.float32)
    signals = {REWARD: reward_advantage, COST: -reward_advantage, LAGRANGIAN: 3 * reward_advantage}
    transitions = C.Transitions.from_recorded_fields(recorded, signals, num_slots=2)
    statistics = frozenset(C.EpisodeStatistic(signal, C.MeanAbsolute()) for signal in Signal)
    return C.RoundRollout(4, transitions, C.EpisodeTracker(num_slots=2, statistics=statistics).complete(transitions))


def test_rollout_has_the_transitions_per_slot_and_the_episodes_that_ended():
    data = _synthetic_round()
    assert data.round_index == 4
    transitions, episodes = data.transitions, data.completed_episodes
    assert (transitions.num_slots, transitions.steps_per_slot, transitions.num_transitions) == (2, 5, 10)
    np.testing.assert_allclose(transitions.signals[REWARD], [[-1.0] * 5, [2.0] * 5])
    np.testing.assert_allclose(transitions.signals[LAGRANGIAN], [[-3.0] * 5, [6.0] * 5])
    assert episodes.count == 2
    np.testing.assert_allclose(episodes.contexts[:, 0], [1.28, 1.72])
    np.testing.assert_allclose(episodes.returns, [2.0, 7.0])  # the recorded totals at the done transitions
    np.testing.assert_allclose(episodes.costs, [3.0, 3.0])
    np.testing.assert_allclose(episodes.lengths, [25.0, 25.0])
    mean_absolute = lambda signal: episodes.statistics[C.EpisodeStatistic(signal, C.MeanAbsolute())]
    np.testing.assert_allclose(mean_absolute(REWARD), [1.0, 2.0])  # mean |advantage| over the episode's three steps
    np.testing.assert_allclose(mean_absolute(COST), [1.0, 2.0])
    np.testing.assert_allclose(mean_absolute(LAGRANGIAN), [3.0, 6.0])
    with pytest.raises(ValueError, match="signals"):
        C.Transitions(transitions.contexts, transitions.episode_done, transitions.episode_return, transitions.episode_cost,
                      transitions.episode_length, {REWARD: transitions.signals[REWARD][:, :3]})


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
    # value loss: mean |advantage| of the transitions in each bin (|−1| below 1.5, 2 above), 0 in empty bins
    value_loss = _histogram_masses(metrics[CURRICULUM + "reward_value_loss/velocity_threshold"])
    np.testing.assert_allclose(_histogram_masses(metrics[CURRICULUM + "cost_value_loss/velocity_threshold"]), value_loss)
    np.testing.assert_allclose(_histogram_masses(metrics[CURRICULUM + "lagrangian_value_loss/velocity_threshold"]), 3 * value_loss)
    edges = np.linspace(float(suite.space.low[0]), float(suite.space.high[0]), NUM_BINS + 1)
    flat_contexts, flat_advantage = data.transitions.flat_contexts[:, 0], data.transitions.flat_signals[REWARD]
    bins = np.digitize(flat_contexts, edges) - 1
    expected = [np.abs(flat_advantage[bins == b]).mean() if np.any(bins == b) else 0.0 for b in range(NUM_BINS)]
    np.testing.assert_allclose(value_loss, expected)
    assert 0.0 in value_loss and 1.0 in value_loss and 2.0 in value_loss
    assert not any("bin_" in key or "completed_episodes/" in key for key in metrics)


def test_trainer_refuses_a_teacher_whose_signals_the_agent_does_not_produce_before_compiling():
    """`plr:cost` needs the cost advantage; an agent that only produces the reward advantage
    is refused in argument validation — no environment is reset, nothing is compiled."""
    suite = C.suite_contexts(ENV_NAME)
    hook = C.ContextRoundHook(C.parse_distribution("plr:cost", suite, total_rounds=1), num_slots=4)
    assert hook.required_signals == {COST}
    never_called = lambda *args, **kwargs: pytest.fail("compute ran: validation did not happen first")
    reward_only = LearningSignalsFunction(produces=frozenset({REWARD}), compute=never_called)

    from jax._src import monitoring
    from jax._src.dispatch import BACKEND_COMPILE_EVENT
    compiles: List[float] = []
    monitoring.register_event_duration_secs_listener(
        lambda event, seconds, **_: compiles.append(seconds) if event == BACKEND_COMPILE_EVENT else None
    )
    with pytest.raises(ValueError, match="COST_ADVANTAGE.*produces.*REWARD_ADVANTAGE"):
        ppo_train.train(environment=envs.get_environment(ENV_NAME, level=1), num_timesteps=80, episode_length=20,
                        num_envs=4, batch_size=4, num_minibatches=1, unroll_length=5,
                        round_hook=hook, learning_signals_fn=reward_only)
    assert compiles == [], f"the refusal came after compiling: {compiles}"
    with pytest.raises(ValueError, match="learning_signals_fn"):
        ppo_train.train(environment=envs.get_environment(ENV_NAME, level=1), num_timesteps=80, episode_length=20,
                        num_envs=4, batch_size=4, num_minibatches=1, unroll_length=5, round_hook=hook)


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


def _run(
    training_spec: str,
    train_fn: Callable[..., Any] = train_ppo_lag,
    learner: C.Learner = C.DEFAULT_LEARNER,
    env_name: str = ENV_NAME,
    **train_overrides: Any,
) -> Tuple[C.ContextTrainingSetup, List[Tuple[int, Dict[str, Any]]], List[float], _RecordingHook]:
    setup = C.context_training_setup(
        env_name,
        training_spec,
        "level:3",
        learner,
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
    env = envs.get_environment(env_name, level=1)
    train_kwargs = setup.train_kwargs()
    hook = _RecordingHook(setup.distribution, num_slots=setup.num_envs)
    train_kwargs["round_hook"] = hook
    train_fn(
        environment=env,
        num_timesteps=NUM_TIMESTEPS,
        seed=0,
        progress_fn=lambda step, metrics: history.append((step, dict(metrics))),
        **{"safety_bound": 25.0, **train_overrides},
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


def test_end_to_end_ppo_pid_runs_the_curriculum_and_logs_only_registered_keys():
    """PPO-PID forwards the two curriculum hooks like PPO-Lagrange; every `training/*` key
    it emits (its controller's `pid/violation` included) is known to the dashboard."""
    _, history, _, _ = _run("uniform", train_fn=train_ppo_pid)
    round_entries = _round_entries(history)
    assert len(round_entries) == TOTAL_ROUNDS
    assert any(f"evaluation/{C.UNIFORM_EVALUATION}/episode_reward" in m for _, m in history)
    training_keys = {key for _, m in history for key in m if key.startswith("training/")}
    assert {"training/lambda_lagr", "training/pid/violation"} <= training_keys
    for key in training_keys:
        dashboard.registered(key)  # raises UnregisteredMetric otherwise


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
    shipped over that whole episode; the intended distribution is then the replay mixture.
    Scored by the Lagrangian objective, so the λ the agent keeps in its state reaches the score
    (the reward-scored variant runs in the parametrised test above)."""
    setup, history, _, hook = _run("plr:lagrangian")
    plr = setup.distribution
    assert isinstance(plr, C.PrioritizedLevelReplay) and plr.score == "lagrangian"
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
    # *_value_loss/<d>: mean |advantage| per bin -- only bins the round had transitions in can be
    # non-zero (a bin whose transitions all had advantage exactly 0 is legitimately 0)
    for _, m in round_entries:
        experienced = _histogram_masses(m[EXPERIENCED])
        for signal in Signal:  # one heatmap per signal the agent produces, whatever PLR ranks by
            value_loss = _histogram_masses(m[f"{CURRICULUM}{signal.value}_value_loss/velocity_threshold"])
            assert np.all(experienced[value_loss > 0] > 0), signal
            assert np.any(value_loss > 0), signal
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


# --------------------------------------------------------------------------- #
# The contextual learner on the real trainer
# --------------------------------------------------------------------------- #


def _per_round_training_metrics(history) -> List[Dict[str, Any]]:
    """The one entry per round in which the trainer's `training/*` arrive together with the
    hook's `training_curriculum/*` (the stock logger re-reports `training/*` on its own cadence)."""
    return [m for _, m in history if CURRICULUM + "num_transitions" in m and "training/cost_violation" in m]


def test_end_to_end_scalar_constraint_reads_the_batch_cost_from_the_rounds_data():
    """The scalar rule now reads the cost from the round's transitions instead of the loss's
    minibatch metrics; the two agree, so `cost_violation` is the batch's `mean_cost` minus the
    budget per step to float precision, and λ moves by rate × that."""
    _, history, _, _ = _run("uniform")
    rounds = _per_round_training_metrics(history)
    assert len(rounds) == TOTAL_ROUNDS
    budget_per_step = 25.0 / TINY["episode_length"]
    for metrics in rounds:
        assert float(np.mean(metrics["training/cost_violation"])) == pytest.approx(
            float(np.mean(metrics["training/mean_cost"])) - budget_per_step, abs=1e-6
        )
    assert not any(key.startswith("training/lambda_lagr/") for m in rounds for key in m), "a scalar arm logs no spread"


@pytest.mark.parametrize("form", ["network", "additive"])
def test_end_to_end_per_context_constraint_with_a_told_policy(form):
    """λ(ω) in either function class with the policy told its context, on lift ant (Ω = {0,1}^4:
    which feet are restricted). One compiled program; the multiplier function's mean, spread and
    fit error over the batch are logged under registered keys; the policy is evaluated on w and r
    through the {sensors, context} observation."""
    told = C.Learner(constraint=form, policy_context="told")
    # budget 0: a context whose restricted feet touch the ground is over budget and its λ rises; the
    # context that restricts no foot has cost 0 and its λ falls, so the function must spread
    setup, history, compile_seconds, _ = _run(
        "uniform", learner=told, env_name="safe_lift_ant", safety_bound=0.0, lagrangian_coef_rate=1.0, initial_lambda_lagr=0.1
    )
    assert setup.arm == f"uniform {form} told"
    assert setup.wandb_config()["arm"] == setup.arm and setup.wandb_config()["constraint"] == form

    rounds = _per_round_training_metrics(history)
    assert len(rounds) == TOTAL_ROUNDS
    for metrics in rounds:
        mean, std = float(np.mean(metrics["training/lambda_lagr"])), float(np.mean(metrics["training/lambda_lagr/std"]))
        low, high = float(np.mean(metrics["training/lambda_lagr/min"])), float(np.mean(metrics["training/lambda_lagr/max"]))
        assert 0.0 <= low <= mean <= high and std >= 0.0
        assert np.isfinite(float(np.mean(metrics["training/lambda_lagr/fit_error"])))
        assert float(np.mean(metrics["training/lambda_lagr/target_spread"])) >= 0.0
        for key in metrics:
            if key.startswith("training/"):
                dashboard.registered(key)
    # contexts whose cost differs end with different multipliers: the spread is not identically zero
    assert float(np.mean(rounds[-1]["training/lambda_lagr/std"])) > 0
    assert float(np.mean(rounds[-1]["training/lambda_lagr/max"])) > float(np.mean(rounds[-1]["training/lambda_lagr/min"]))

    dimension = setup.suite.space.names[0]
    assert all(_histogram_masses(m[f"{CURRICULUM}experienced/{dimension}"]).sum() == pytest.approx(1.0) for _, m in history if f"{CURRICULUM}experienced/{dimension}" in m)
    evaluations = [m for _, m in history if f"evaluation/{C.DEPLOYMENT_EVALUATION}/episode_reward" in m]
    assert len(evaluations) == 3 and all(f"evaluation/{C.UNIFORM_EVALUATION}/episode_cost" in m for m in evaluations)
    substantial = [s for s in compile_seconds if s > 1.0]
    assert len(substantial) <= 4, f"unexpected recompiles: {substantial}"


def test_per_context_constraint_needs_ppo_lagrange():
    """PPO-PID has no `constraint` argument: the setup's kwargs are refused by the trainer
    (the CLI turns this into its 'does not accept' message before anything runs)."""
    told = C.Learner(constraint="additive", policy_context="told")
    setup = C.context_training_setup("safe_lift_ant", "uniform", "level:3", told, num_timesteps=NUM_TIMESTEPS,
                                     num_envs=TINY["num_envs"], batch_size=TINY["batch_size"],
                                     unroll_length=TINY["unroll_length"], num_minibatches=TINY["num_minibatches"])
    with pytest.raises(TypeError, match="constraint"):
        train_ppo_pid(environment=envs.get_environment("safe_lift_ant", level=1), num_timesteps=NUM_TIMESTEPS, **TINY, **setup.train_kwargs())
