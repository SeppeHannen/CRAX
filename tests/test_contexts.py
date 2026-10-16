"""Tests for training.contexts: context spaces, distributions, and per-episode
context sampling through ContextualAutoResetWrapper.

Run with: JAX_PLATFORMS=cpu pytest tests/test_contexts.py -v
"""
from __future__ import annotations

import dataclasses
from typing import Dict

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from crax import envs
from training import contexts as C
from training.contexts.constraints.additive import CONTINUOUS_BINS
from training.agents.ppo.train import _strip_weak_type
from training.rounds import Signal

ENV_NAME = "safe_velocity_ant"
NUM_SLOTS = 8
EPISODE_LENGTH = 25

REWARD, COST, LAGRANGIAN = Signal.REWARD_ADVANTAGE, Signal.COST_ADVANTAGE, Signal.LAGRANGIAN_ADVANTAGE
MEAN_ABSOLUTE = {signal: C.EpisodeStatistic(signal, C.MeanAbsolute()) for signal in Signal}


def signals(reward, cost=None, lagrangian=None) -> Dict[Signal, np.ndarray]:
    """The three learning signals, ``reward`` standing in for any not given."""
    reward = np.asarray(reward, np.float32)
    return {
        REWARD: reward,
        COST: reward if cost is None else np.asarray(cost, np.float32),
        LAGRANGIAN: reward if lagrangian is None else np.asarray(lagrangian, np.float32),
    }


def round_rollout(
    contexts: np.ndarray, per_slot_signals: Dict[Signal, np.ndarray], distribution: C.ContextDistribution,
    round_index: int, steps: int = 5,
) -> C.RoundRollout:
    """A round in which slot ``i`` ran ``steps`` transitions in ``contexts[i]`` with
    ``per_slot_signals[s][i]`` at every step, and its episode ended on the last one — so
    every context has exactly one completed episode whose mean |signal| is
    ``|per_slot_signals[s][i]|``. The episodes carry the statistics ``distribution`` asks for."""
    num_slots = contexts.shape[0]
    done = np.zeros((num_slots, steps), bool)
    done[:, -1] = True
    per_step = lambda per_slot: np.repeat(np.asarray(per_slot, np.float32)[:, None], steps, axis=1)
    transitions = C.Transitions(
        contexts=np.repeat(np.asarray(contexts, np.float32)[:, None, :], steps, axis=1),
        episode_done=done,
        episode_return=np.zeros((num_slots, steps), np.float32),
        episode_cost=np.zeros((num_slots, steps), np.float32),
        episode_length=np.full((num_slots, steps), float(steps), np.float32),
        signals={signal: per_step(values) for signal, values in per_slot_signals.items()},
    )
    tracker = C.EpisodeTracker(num_slots, distribution.required_statistics)
    return C.RoundRollout(round_index, transitions, tracker.complete(transitions))


@pytest.fixture(scope="module")
def suite():
    return C.suite_contexts(ENV_NAME)


@pytest.fixture(scope="module")
def env():
    return envs.get_environment(ENV_NAME, level=1)


def build(env, distribution):
    wrapped = C.wrap_for_context_training(env, distribution, episode_length=EPISODE_LENGTH)
    return jax.jit(wrapped.reset), jax.jit(wrapped.step)


def zero_actions(env):
    return jnp.zeros((NUM_SLOTS, env.action_size))


# --------------------------------------------------------------------------- #
# Space and registry
# --------------------------------------------------------------------------- #


def test_space_round_trip(suite):
    ctx = suite.space.encode(velocity_threshold=1.5)
    assert suite.space.decode(ctx) == {"velocity_threshold": pytest.approx(1.5)}
    assert bool(suite.space.contains(ctx[None])[0])
    assert not bool(suite.space.contains(jnp.asarray([[100.0]]))[0])


def test_levels_match_difficulty_module(suite):
    from crax.envs.safe_velocity import get_threshold_for_level

    assert sorted(suite.level_contexts) == [1, 2, 3]
    for level in (1, 2, 3):
        assert float(suite.level(level)[0]) == pytest.approx(get_threshold_for_level("ant", level))
    with pytest.raises(KeyError, match="no level 4"):
        suite.level(4)


def test_uniform_samples_inside_space(suite):
    samples = suite.space.sample_uniform(jax.random.PRNGKey(0), 1000)
    assert samples.shape == (1000, 1)
    assert bool(jnp.all(suite.space.contains(samples)))


def test_integer_total_cap_cuts_the_box_and_uniform_is_uniform_over_what_remains():
    """A capped Ω: integer tuples over the cap are outside Ω, the sampler never draws
    them, every admissible tuple is equally likely (no endpoint bias from rounding),
    and the log-density is one constant over Ω."""
    space = C.ContextSpace(
        (C.Dimension("a", 0, 3, "integer"), C.Dimension("b", 0, 2, "integer"), C.Dimension("x", 0.0, 1.0)),
        integer_total_cap=3,
    )
    admissible = {(a, b) for a in range(4) for b in range(3) if a + b <= 3}
    assert {tuple(point) for point in space.integer_points} == admissible
    assert not bool(space.contains(space.encode(a=3, b=1, x=0.5)[None])[0])
    assert bool(space.contains(space.encode(a=3, b=0, x=0.5)[None])[0])

    samples = np.asarray(space.sample_uniform(jax.random.PRNGKey(0), 20_000))
    assert bool(jnp.all(space.contains(jnp.asarray(samples))))
    tuples, counts = np.unique(samples[:, :2].astype(int), axis=0, return_counts=True)
    assert {tuple(point) for point in tuples} == admissible
    expected = 20_000 / len(admissible)
    assert np.all(np.abs(counts - expected) < 4 * np.sqrt(expected)), counts  # endpoints get full weight


def test_a_cap_below_every_integer_total_is_refused():
    with pytest.raises(ValueError, match="integer_total_cap"):
        C.ContextSpace((C.Dimension("a", 2, 5, "integer"),), integer_total_cap=1)
    with pytest.raises(ValueError, match="no integer dimension"):
        C.ContextSpace((C.Dimension("x", 0.0, 1.0),), integer_total_cap=1)


def test_integer_total_floor_mirrors_the_cap():
    """A floor keeps only tuples whose total is at least the floor; sampling respects it; an
    impossible floor, or one above the cap, is refused."""
    dims = (C.Dimension("a", 0, 3, "integer"), C.Dimension("b", 0, 2, "integer"), C.Dimension("x", 0.0, 1.0))
    space = C.ContextSpace(dims, integer_total_floor=4)
    assert {tuple(p) for p in space.integer_points} == {(a, b) for a in range(4) for b in range(3) if a + b >= 4}
    assert not bool(space.contains(space.encode(a=3, b=0, x=0.5)[None])[0])
    assert bool(space.contains(space.encode(a=3, b=1, x=0.5)[None])[0])
    samples = space.sample_uniform(jax.random.PRNGKey(0), 2000)
    assert bool(jnp.all(space.contains(samples))) and bool(jnp.all(samples[:, 0] + samples[:, 1] >= 4))
    assert "≥ 4" in space.describe()
    with pytest.raises(ValueError, match="integer_total_floor"):
        C.ContextSpace(dims, integer_total_floor=6)
    with pytest.raises(ValueError, match="exceeds"):
        C.ContextSpace(dims, integer_total_cap=2, integer_total_floor=3)


def test_goal_point_out_of_distribution_region_is_disjoint_from_omega():
    """The region shares Ω's dimensions and no point: denser than the cap and a smaller goal."""
    suite = C.suite_contexts("safe_goal_point")
    region = suite.region("out_of_distribution")
    assert region.relation == "disjoint_from" and region.space.names == suite.space.names
    draws = region.space.sample_uniform(jax.random.PRNGKey(0), 2000)
    assert not bool(suite.space.contains(draws).any()) and bool(region.space.contains(draws).all())
    assert bool(jnp.all(draws[:, :4].sum(axis=1) > suite.space.integer_total_cap))
    assert float(draws[:, 4].max()) < float(suite.space.low[4])
    assert C.suite_contexts("safe_velocity_ant").evaluation_regions == ()
    with pytest.raises(KeyError, match="no evaluation region"):
        suite.region("skill/navigate")


def test_evaluation_regions_state_their_relation_to_omega_and_it_is_checked():
    """A region declared disjoint must share no point with Ω; one declared within must lie
    inside it; names must be unique."""
    suite = C.suite_contexts("safe_goal_point")
    overlapping = C.ContextSpace(suite.space.dimensions, integer_total_floor=10)  # shares tuples 10..20 with Ω
    outside = suite.region("out_of_distribution").space
    make = lambda **fields: C.EvaluationRegion(**{"name": "probe", "space": outside, "label": "", "relation": "disjoint_from", **fields})
    with pytest.raises(ValueError, match="overlaps"):
        dataclasses.replace(suite, evaluation_regions=(make(space=overlapping),))
    with pytest.raises(ValueError, match="not within"):
        dataclasses.replace(suite, evaluation_regions=(make(space=outside, relation="within"),))
    dense_corner = C.ContextSpace(suite.space.dimensions, integer_total_cap=20, integer_total_floor=18)  # a slice of Ω
    within = dataclasses.replace(suite, evaluation_regions=(make(space=dense_corner, relation="within"),))
    assert within.region("probe").space is dense_corner
    with pytest.raises(ValueError, match="share a name"):
        dataclasses.replace(suite, evaluation_regions=(make(), make()))
    with pytest.raises(ValueError, match="dimensions"):
        dataclasses.replace(suite, evaluation_regions=(make(space=C.box(x=(0.0, 1.0))),))


# --------------------------------------------------------------------------- #
# Wrapper: per-slot contexts, reset on done
# --------------------------------------------------------------------------- #


def test_uniform_contexts_differ_per_slot_and_env_reads_them(suite, env):
    reset, _ = build(env, C.UniformDistribution(suite.space))
    state = reset(jax.random.split(jax.random.PRNGKey(0), NUM_SLOTS))
    contexts = np.asarray(C.current_contexts(state))[:, 0]
    assert len(set(np.round(contexts, 4))) > 1
    np.testing.assert_allclose(np.asarray(state.info["velocity_threshold"]), contexts)
    np.testing.assert_allclose(np.asarray(state.metrics["velocity_threshold"]), contexts)


def test_context_changes_exactly_at_episode_end(suite, env):
    """A slot's context changes on exactly the steps its episode ends, and the slots
    are spread over the episode: the first reset gives each a random head start, so
    their first episodes end at different steps and they never all end together."""
    reset, step = build(env, C.UniformDistribution(suite.space))
    state = reset(jax.random.split(jax.random.PRNGKey(0), NUM_SLOTS))
    initial_steps = np.asarray(state.info["steps"]).astype(int)
    assert len(set(initial_steps.tolist())) > 1, initial_steps
    assert np.all((0 <= initial_steps) & (initial_steps < EPISODE_LENGTH))
    history = [np.asarray(C.current_contexts(state))[:, 0]]
    done_history = []
    for _ in range(2 * EPISODE_LENGTH + 1):
        state = step(state, zero_actions(env))
        history.append(np.asarray(C.current_contexts(state))[:, 0])
        done_history.append(np.asarray(state.done).astype(bool))
        # the env's view must always agree with the wrapper's
        np.testing.assert_allclose(np.asarray(state.info["velocity_threshold"]), history[-1])
    changed = np.abs(np.diff(np.stack(history), axis=0)) > 1e-6  # [T, N]
    np.testing.assert_array_equal(changed, np.stack(done_history))  # change iff that slot's episode ended
    first_end = np.argmax(np.stack(done_history), axis=0)
    np.testing.assert_array_equal(first_end + 1, EPISODE_LENGTH - initial_steps)  # the head start shortens the first episode
    assert not np.any(np.stack(done_history).all(axis=1)), "every slot ended on the same step: slots are synchronised"


def test_evaluation_episodes_start_at_step_zero_and_are_summed_whole(suite, env):
    """The Evaluator sums an episode's metrics until its first `done`. On the evaluation
    stack every episode starts at step 0, so what it reports is the whole episode: the
    cost summed from the first step to the episode's own end (a fall, or the step limit).
    The training stack's phase spread would cut that sum short by a random head start
    (the 2026-10-08 bug, which halved every evaluation metric)."""
    from training.acting import Evaluator

    distribution = C.UniformDistribution(suite.space)
    evaluation_env = C.wrap_for_context_evaluation(env, distribution, episode_length=EPISODE_LENGTH)

    def random_policy(_params):
        return lambda observation, key: (
            jax.random.uniform(key, (observation[C.SENSORS_KEY].shape[0], env.action_size), minval=-1.0, maxval=1.0), {}
        )

    from crax.envs.wrappers.training import EvalWrapper
    from training.acting import generate_unroll

    key = jax.random.PRNGKey(4)
    reported = Evaluator(evaluation_env, random_policy, num_eval_envs=NUM_SLOTS, episode_length=EPISODE_LENGTH, action_repeat=1, key=key).run_evaluation(None, training_metrics={})
    assert reported["eval/episode_cost"] > 0, "the random policy incurred no cost; the test would pass vacuously"
    assert reported["eval/avg_episode_length"] < EPISODE_LENGTH, "the ant falls under random torques; if not, the length check below is vacuous"

    # The Evaluator's unroll on the evaluation stack, with every step's done recorded: the
    # episode length EvalWrapper accumulates (what avg_episode_length reports) must be the
    # number of steps each episode ran from step 0 to its own first done -- nothing before
    # the unroll is missing. (A separate unroll rather than the Evaluator's own, whose
    # jitted reset+unroll is not bit-reproducible under random torques on CPU.)
    _, unroll_key = jax.random.split(key)
    eval_env = EvalWrapper(evaluation_env)
    first = eval_env.reset(jax.random.split(unroll_key, NUM_SLOTS))
    assert not np.any(np.asarray(first.info["steps"])), "an evaluation episode must start at step 0"
    final, data = jax.jit(lambda s, k: generate_unroll(eval_env, s, random_policy(None), k, EPISODE_LENGTH))(first, unroll_key)
    steps_run = np.argmax(np.asarray(data.discount) == 0, axis=0) + 1  # every slot ends by the step limit at the latest
    np.testing.assert_array_equal(np.asarray(final.info["eval_metrics"].episode_steps), steps_run)

    # The training stack through the same Evaluator is the bug: its head start makes the
    # reported length larger than the steps actually run, by the head start.
    training_eval = EvalWrapper(C.wrap_for_context_training(env, distribution, episode_length=EPISODE_LENGTH))
    first = training_eval.reset(jax.random.split(unroll_key, NUM_SLOTS))
    head_start = np.asarray(first.info["steps"])
    assert np.any(head_start > 0)
    final, data = jax.jit(lambda s, k: generate_unroll(training_eval, s, random_policy(None), k, EPISODE_LENGTH))(first, unroll_key)
    steps_run = np.argmax(np.asarray(data.discount) == 0, axis=0) + 1
    reported_length = np.asarray(final.info["eval_metrics"].episode_steps)
    assert np.all(reported_length >= steps_run) and np.any(reported_length > steps_run)


def test_fixed_context_holds_across_resets(suite, env):
    reset, step = build(env, C.FixedContext(suite.space, suite.level(3)))
    state = reset(jax.random.split(jax.random.PRNGKey(1), NUM_SLOTS))
    for _ in range(EPISODE_LENGTH + 2):
        state = step(state, zero_actions(env))
    np.testing.assert_allclose(np.asarray(C.current_contexts(state))[:, 0], float(suite.level(3)[0]))


def test_staged_switches_without_recompiling(suite, env):
    from jax._src import monitoring
    from jax._src.dispatch import BACKEND_COMPILE_EVENT

    staged = C.parse_distribution("staged:1,2,3", suite, total_rounds=3)
    reset, step = build(env, staged)
    state = _strip_weak_type(reset(jax.random.split(jax.random.PRNGKey(2), NUM_SLOTS)))
    parameters = C.current_parameters(state)
    actions = zero_actions(env)
    # Warm up: first call compiles; train() strips weak types on every call.
    state = _strip_weak_type(step(state, actions))
    state = _strip_weak_type(step(state, actions))

    substantial = []
    monitoring.register_event_duration_secs_listener(
        lambda event, seconds, **_: substantial.append(seconds) if event == BACKEND_COMPILE_EVENT and seconds > 0.2 else None
    )
    trained_at = []
    for round_index in range(3):
        for _ in range(EPISODE_LENGTH + 1):
            state = _strip_weak_type(step(state, actions))
        trained_at.append(float(np.asarray(C.current_contexts(state))[0, 0]))
        rollout = round_rollout(np.asarray(C.current_contexts(state)), signals(np.zeros(NUM_SLOTS)), staged, round_index)
        parameters = staged.update(parameters, rollout)
        state = _strip_weak_type(C.attach_parameters(state, parameters))

    assert trained_at == pytest.approx([float(suite.level(l)[0]) for l in (1, 2, 3)])
    assert substantial == [], f"stage switching recompiled the step program: {substantial}"


# --------------------------------------------------------------------------- #
# Rounds: the per-slot layout, and episodes completed across round boundaries
# --------------------------------------------------------------------------- #


def test_transitions_recover_each_slots_time_series_from_the_recorded_slot_labels():
    """The trainer records one row per (unroll, slot), each labelled with its slot by the
    wrapper. A slot's transitions must come back in time order whatever order the trainer
    put the rows in; labels that do not describe ``num_slots`` equal slots are refused."""
    num_unrolls, num_slots, unroll_length = 3, 2, 4
    time_index = np.arange(num_unrolls * unroll_length, dtype=np.float32)  # what every slot saw, in order
    # one row per (unroll, slot); slot s's rows carry the label s and its unrolls in time order
    rows = np.stack([time_index.reshape(num_unrolls, unroll_length)] * num_slots, axis=1).reshape(-1, unroll_length)  # [U*N, T]
    labels = np.tile(np.arange(num_slots), num_unrolls)[:, None].repeat(unroll_length, axis=1)
    done = np.zeros_like(rows)
    done[:, -1] = 1.0  # every unroll ends an episode -> contexts may change between unrolls
    contexts = (rows // unroll_length)[..., None]  # one context per episode

    def recorded_in(order: np.ndarray):
        fields = {C.TRANSITION_CONTEXT_KEY: contexts, C.SLOT_INDEX_KEY: labels, "episode_done": done,
                  "episode_metrics": {name: rows for name in ("sum_reward", "cost", "length")}}
        permute = lambda x: x[order]
        return jax.tree_util.tree_map(permute, fields), signals(rows[order])

    # the trainer's actual order, and a shuffled one that keeps each slot's unrolls in time order
    trainer_order = np.arange(num_unrolls * num_slots)
    shuffled = np.array([1, 0, 3, 5, 2, 4])
    for order in (trainer_order, shuffled):
        transitions = C.Transitions.from_recorded_fields(*recorded_in(order), num_slots)
        assert (transitions.num_slots, transitions.steps_per_slot) == (num_slots, num_unrolls * unroll_length)
        for slot in range(num_slots):
            np.testing.assert_array_equal(transitions.signals[REWARD][slot], time_index)
            np.testing.assert_array_equal(transitions.episode_return[slot], time_index)

    fields, recorded_signals = recorded_in(trainer_order)
    with pytest.raises(ValueError, match="equally many rows"):
        C.Transitions.from_recorded_fields(fields, recorded_signals, num_slots=4)
    with pytest.raises(KeyError, match=C.SLOT_INDEX_KEY):
        C.Transitions.from_recorded_fields({k: v for k, v in fields.items() if k != C.SLOT_INDEX_KEY}, recorded_signals, num_slots)
    # a context changing mid-episode means the layout is wrong
    with pytest.raises(ValueError, match="layout"):
        C.Transitions(contexts=rows[..., None], episode_done=np.zeros_like(rows, dtype=bool), episode_return=rows,
                      episode_cost=rows, episode_length=rows, signals=signals(rows))
    with pytest.raises(ValueError, match="shape"):
        C.Transitions(contexts=rows[..., None], episode_done=np.zeros_like(rows, dtype=bool), episode_return=rows,
                      episode_cost=rows, episode_length=rows, signals={REWARD: rows[:, :2]})


@dataclasses.dataclass(frozen=True)
class _Maximum:
    """A test-only fold with a non-additive monoid: the episode's largest signal."""

    combine = np.maximum
    identity = -np.inf

    def map(self, signal):
        return signal

    def finalise(self, combined, count):
        return combined


@dataclasses.dataclass(frozen=True)
class _WrongIdentityMaximum(_Maximum):
    """0 is not neutral for max: max(0, -7) = 0."""

    identity = 0.0


def test_episode_statistic_is_a_signal_and_a_fold():
    statistic = C.EpisodeStatistic(COST, C.MeanAbsolute())
    assert statistic == C.EpisodeStatistic(COST, C.MeanAbsolute()) and hash(statistic) == hash(C.EpisodeStatistic(COST, C.MeanAbsolute()))
    assert statistic != C.EpisodeStatistic(COST, _Maximum())
    assert statistic.name == "cost/MeanAbsolute"
    with pytest.raises(TypeError, match="Signal"):
        C.EpisodeStatistic("cost", C.MeanAbsolute())
    with pytest.raises(TypeError, match="Fold"):
        C.EpisodeStatistic(COST, np.abs)
    with pytest.raises(ValueError, match="not neutral"):
        C.EpisodeStatistic(COST, _WrongIdentityMaximum())
    assert Signal("cost") is COST  # the CLI spelling resolves to the member


def test_episode_tracker_computes_every_requested_statistic_over_the_whole_episode_across_rounds():
    """Slot 0's episode spans two rounds and ends in the second; slot 1 ends two episodes in
    the first round. Each requested statistic is taken over *all* of the episode's steps:
    mean |x| for two signals (the cost one twice the reward one) and, through a fold with a
    non-additive combine, the episode's maximum reward signal."""
    maximum_reward = C.EpisodeStatistic(REWARD, _Maximum())
    tracker = C.EpisodeTracker(num_slots=2, statistics=frozenset({MEAN_ABSOLUTE[REWARD], MEAN_ABSOLUTE[COST], maximum_reward}))

    def transitions(advantage, done, contexts):
        advantage = np.asarray(advantage, np.float32)
        return C.Transitions(contexts=np.asarray(contexts, np.float32)[..., None], episode_done=np.asarray(done, bool),
                             episode_return=advantage, episode_cost=advantage, episode_length=advantage,
                             signals=signals(advantage, cost=2 * advantage))

    first = tracker.complete(transitions(
        advantage=[[1, 1, 1, 1], [2, -4, 6, 6]],
        done=[[0, 0, 0, 0], [0, 1, 0, 1]],
        contexts=[[10, 10, 10, 10], [20, 20, 21, 21]],
    ))
    assert first.count == 2
    np.testing.assert_allclose(first.contexts[:, 0], [20, 21])
    np.testing.assert_allclose(first.statistics[MEAN_ABSOLUTE[REWARD]], [3.0, 6.0])  # (2+4)/2, (6+6)/2
    np.testing.assert_allclose(first.statistics[MEAN_ABSOLUTE[COST]], [6.0, 12.0])
    np.testing.assert_allclose(first.statistics[maximum_reward], [2.0, 6.0])  # max(2, -4), max(6, 6)
    assert set(first.statistics) == tracker.statistics  # exactly what was asked for
    np.testing.assert_allclose(first.returns, [-4, 6])  # the recorded totals at the done transition

    second = tracker.complete(transitions(
        advantage=[[3, 3, 5, 5], [0, 0, 0, 0]],
        done=[[0, 1, 0, 0], [0, 0, 0, 0]],
        contexts=[[10, 10, 11, 11], [22, 22, 22, 22]],
    ))
    assert second.count == 1
    assert float(second.contexts[0, 0]) == 10
    assert float(second.statistics[MEAN_ABSOLUTE[REWARD]][0]) == pytest.approx((4 * 1 + 3 + 3) / 6)  # four steps from round 1, two from round 2
    assert float(second.statistics[MEAN_ABSOLUTE[COST]][0]) == pytest.approx(2 * (4 * 1 + 3 + 3) / 6)
    assert float(second.statistics[maximum_reward][0]) == 3.0  # the carry's 1s and this round's 3s

    with pytest.raises(ValueError, match="slots"):
        tracker.complete(transitions(advantage=[[1]], done=[[0]], contexts=[[0]]))
    # a round whose signals cannot supply a requested statistic is refused
    only_reward = C.Transitions(contexts=np.zeros((2, 1, 1), np.float32), episode_done=np.ones((2, 1), bool),
                                episode_return=np.zeros((2, 1)), episode_cost=np.zeros((2, 1)), episode_length=np.zeros((2, 1)),
                                signals={REWARD: np.zeros((2, 1))})
    with pytest.raises(KeyError, match="COST_ADVANTAGE"):
        tracker.complete(only_reward)
    # a distribution that asks for nothing gets episodes with no statistics, from any signals
    nothing = C.EpisodeTracker(num_slots=2, statistics=frozenset()).complete(only_reward)
    assert nothing.count == 2 and nothing.statistics == {}


# --------------------------------------------------------------------------- #
# Prioritized Level Replay: the buffer, the replay distribution, the sampler
# --------------------------------------------------------------------------- #


def _buffer_rows(parameters) -> set:
    return {row.tobytes() for row in np.asarray(parameters.contexts, np.float32)[np.asarray(parameters.occupied)]}


def test_plr_first_round_is_uniform_then_keeps_the_highest_scoring_contexts(suite):
    plr = C.PrioritizedLevelReplay(suite.space, buffer_size=8, replay_probability=0.5, temperature=1.0, staleness_coefficient=0.3)
    parameters = plr.initialise()
    assert float(parameters.replay_probability) == 0.0 and not bool(parameters.occupied.any())
    drawn = np.asarray(plr.sample(parameters, jax.random.PRNGKey(1), 10), np.float32)
    assert bool(jnp.all(suite.space.contains(jnp.asarray(drawn)))) and len({row.tobytes() for row in drawn}) == 10

    # ten distinct contexts, slot i with |advantage| i: the buffer (8 rows) keeps the top 8
    parameters = plr.update(parameters, round_rollout(drawn, signals(np.arange(10.0)), plr, round_index=0))
    assert int(parameters.occupied.sum()) == 8
    assert sorted(np.asarray(parameters.scores).tolist()) == [2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
    assert _buffer_rows(parameters) == {row.tobytes() for row in drawn[2:]}
    assert float(parameters.replay_probability) == pytest.approx(0.5)  # the method's p, as soon as there is anything to replay
    assert int(parameters.round_index) == 1 and np.all(np.asarray(parameters.last_round) == 0)


def test_plr_scores_by_the_chosen_signal(suite):
    """`plr` ranks by the reward critic's error, `plr:cost` by the cost critic's,
    `plr:lagrangian` by the Lagrangian objective's; the other signals are ignored. Here the
    three orderings keep three different sets of contexts. PLR asks the tracker for exactly
    one statistic: the mean |score signal|."""
    drawn = np.asarray(C.UniformDistribution(suite.space).sample(None, jax.random.PRNGKey(3), 6), np.float32)
    per_slot = signals(np.arange(6.0), cost=np.arange(6.0)[::-1], lagrangian=[0, 9, 1, 8, 2, 7])
    for score, expected_kept, expected_scores in (
        (REWARD, drawn[2:], [2.0, 3.0, 4.0, 5.0]),
        (COST, drawn[:4], [2.0, 3.0, 4.0, 5.0]),
        (LAGRANGIAN, drawn[[1, 3, 5, 4]], [2.0, 7.0, 8.0, 9.0]),
    ):
        plr = C.PrioritizedLevelReplay(suite.space, score=score, buffer_size=4)
        assert plr.required_statistics == {C.EpisodeStatistic(score, C.MeanAbsolute())}
        parameters = plr.update(plr.initialise(), round_rollout(drawn, per_slot, plr, round_index=0))
        assert _buffer_rows(parameters) == {row.tobytes() for row in expected_kept}, score
        assert sorted(np.asarray(parameters.scores).tolist()) == expected_scores, score
    with pytest.raises(TypeError, match="Signal"):
        C.PrioritizedLevelReplay(suite.space, score="cost")
    assert C.parse_distribution("plr:cost", suite, total_rounds=1).score is COST
    with pytest.raises(ValueError, match="plr:<signal>"):
        C.parse_distribution("plr:return", suite, total_rounds=1)


def test_plr_replay_weights_are_rank_mixed_with_staleness():
    space = C.box(x=(0.0, 1.0))
    plr = C.PrioritizedLevelReplay(space, buffer_size=4, temperature=1.0, staleness_coefficient=0.25)
    scores = np.asarray([0.1, 0.9, 0.5, 0.0])
    last_round = np.asarray([3, 1, 3, -1])
    occupied = np.asarray([True, True, True, False])
    weights = plr.replay_weights(scores, last_round, occupied, round_index=4)
    by_rank = np.asarray([1 / 3, 1 / 1, 1 / 2])  # ranks 3, 1, 2 -> 1/rank
    by_staleness = np.asarray([1.0, 3.0, 1.0])  # rounds old in round 4
    expected = 0.75 * by_rank / by_rank.sum() + 0.25 * by_staleness / by_staleness.sum()
    np.testing.assert_allclose(weights[:3], expected)
    assert weights[3] == 0.0 and weights.sum() == pytest.approx(1.0)
    # a cold temperature concentrates on the top rank
    cold = C.PrioritizedLevelReplay(space, buffer_size=4, temperature=0.1, staleness_coefficient=0.0)
    assert cold.replay_weights(scores, last_round, occupied, round_index=4)[1] > 0.999
    with pytest.raises(ValueError, match="future"):
        plr.replay_weights(scores, last_round, occupied, round_index=3)


def test_plr_samples_buffer_rows_exactly_with_the_replay_weights(suite):
    plr = C.PrioritizedLevelReplay(suite.space, buffer_size=8, replay_probability=0.5, temperature=1.0, staleness_coefficient=0.3)
    parameters = plr.initialise()
    drawn = np.asarray(plr.sample(parameters, jax.random.PRNGKey(1), 8), np.float32)
    parameters = plr.update(parameters, round_rollout(drawn, signals(np.arange(8.0)), plr, round_index=0))

    samples = np.asarray(plr.sample(parameters, jax.random.PRNGKey(2), 20_000), np.float32)
    rows = np.asarray(parameters.contexts, np.float32)
    in_buffer = _buffer_rows(parameters)
    replayed = np.asarray([row.tobytes() in in_buffer for row in samples])
    assert replayed.mean() == pytest.approx(0.5, abs=0.02)
    shares = np.asarray([(samples[replayed] == rows[i]).all(axis=1).sum() for i in range(8)]) / replayed.sum()
    np.testing.assert_allclose(shares, np.asarray(parameters.replay_weights), atol=0.02)
    assert bool(jnp.all(suite.space.contains(jnp.asarray(samples[~replayed]))))


def test_plr_rescores_replayed_contexts_and_admits_only_better_newcomers(suite):
    plr = C.PrioritizedLevelReplay(suite.space, buffer_size=4, replay_probability=1.0, temperature=1.0, staleness_coefficient=0.3)
    parameters = plr.initialise()
    first = np.asarray(plr.sample(parameters, jax.random.PRNGKey(1), 4), np.float32)
    parameters = plr.update(parameters, round_rollout(first, signals([4.0, 3.0, 2.0, 1.0]), plr, round_index=0))
    rows = np.asarray(parameters.contexts, np.float32)

    # round 1: rows 0 and 1 are trained on again (new, lower scores), plus two newcomers: one
    # that beats the row with the least replay mass, one that beats nothing
    newcomers = np.asarray(suite.space.sample_uniform(jax.random.PRNGKey(7), 2), np.float32)
    contexts = np.concatenate([rows[:2], newcomers])
    parameters = plr.update(parameters, round_rollout(contexts, signals([0.5, 0.25, 10.0, 0.0]), plr, round_index=1))

    scores = dict(zip((row.tobytes() for row in np.asarray(parameters.contexts, np.float32)), np.asarray(parameters.scores).tolist()))
    assert scores[rows[0].tobytes()] == 0.5  # overwritten, not averaged
    assert scores[newcomers[0].tobytes()] == 10.0
    assert newcomers[1].tobytes() not in scores  # scored 0: beats nothing
    # the displaced row is the one with the least replay mass: the lowest score (row 1, just
    # rescored to 0.25) and the freshest score; rows 2 and 3 are stale, which protects them
    assert rows[1].tobytes() not in scores
    assert rows[2].tobytes() in scores and rows[3].tobytes() in scores
    last_round = dict(zip((row.tobytes() for row in np.asarray(parameters.contexts, np.float32)), np.asarray(parameters.last_round).tolist()))
    assert last_round[rows[0].tobytes()] == 1 and last_round[newcomers[0].tobytes()] == 1 and last_round[rows[2].tobytes()] == 0
    summary = plr.summary(parameters)
    assert summary["buffer_occupancy"] == 1.0 and summary["score/max"] == 10.0
    assert summary["staleness/mean"] == pytest.approx((1 + 1 + 2 + 2) / 4)  # round_index 2 minus last_round
    assert summary["replay_mass/top_10"] == pytest.approx(1.0) == summary["replay_mass/top_100"]  # 4 rows hold everything
    top_row = float(np.asarray(parameters.replay_weights).max())
    assert top_row < 1.0  # with ρ > 0 the top row never holds all of it


def test_plr_rejects_hyperparameters_outside_their_range(suite):
    for bad in (dict(buffer_size=0), dict(replay_probability=1.5), dict(temperature=0.0), dict(staleness_coefficient=-0.1)):
        with pytest.raises(ValueError):
            C.PrioritizedLevelReplay(suite.space, **bad)


# --------------------------------------------------------------------------- #
# Contextual learner: the observation, the networks, λ(ω)
# --------------------------------------------------------------------------- #

LIFT = "safe_lift_ant"  # an Ω with integer dimensions (one 0/1 per foot) and no continuous one


def test_observation_is_the_sensors_and_the_running_episodes_context(env, suite):
    """Both stacks observe {sensors, context}; the context part is always the context of
    the episode the sensors describe, including on the step a slot's episode ends."""
    for wrap in (C.wrap_for_context_training, C.wrap_for_context_evaluation):
        wrapped = wrap(env, C.UniformDistribution(suite.space), episode_length=EPISODE_LENGTH)
        step = jax.jit(wrapped.step)
        state = jax.jit(wrapped.reset)(jax.random.split(jax.random.PRNGKey(0), NUM_SLOTS))
        ended = 0
        for _ in range(EPISODE_LENGTH + 1):
            assert set(state.obs) == {C.SENSORS_KEY, C.CONTEXT_OBSERVATION_KEY}
            assert state.obs[C.SENSORS_KEY].shape == (NUM_SLOTS, env.observation_size)
            np.testing.assert_array_equal(np.asarray(state.obs[C.CONTEXT_OBSERVATION_KEY]), np.asarray(C.current_contexts(state)))
            ended += int(np.asarray(state.done).sum())
            state = step(state, zero_actions(env))
        assert ended > 0, "no episode ended; the at-done case was not exercised"


def test_networks_read_the_context_where_the_learner_allows_it():
    """Blind policy: a function of the sensors only. Told policy and both critics: the
    context changes their output."""
    from training.acme import running_statistics, specs

    observation_size = {C.SENSORS_KEY: (6,), C.CONTEXT_OBSERVATION_KEY: (2,)}
    normalizer = running_statistics.init_state({key: specs.Array(shape, jnp.float32) for key, shape in observation_size.items()})
    sensors = jax.random.normal(jax.random.PRNGKey(1), (3, 6))
    observation = {C.SENSORS_KEY: sensors, C.CONTEXT_OBSERVATION_KEY: jnp.ones((3, 2))}
    other_context = {C.SENSORS_KEY: sensors, C.CONTEXT_OBSERVATION_KEY: -jnp.ones((3, 2))}
    for policy_observes_context in (False, True):
        networks = C.make_contextual_ppo_networks(
            observation_size, action_size=2, preprocess_observations_fn=running_statistics.normalize,
            policy_observes_context=policy_observes_context, context_bounds={"low": (-2.0, -2.0), "high": (2.0, 2.0)},
        )
        keys = jax.random.split(jax.random.PRNGKey(2), 3)
        policy_params = networks.policy_network.init(keys[0])
        logits = lambda obs: networks.policy_network.apply(normalizer, policy_params, obs)
        assert logits(observation).shape == (3, 4)  # mean and std per action
        policy_sees_it = not np.allclose(logits(observation), logits(other_context))
        assert policy_sees_it == policy_observes_context
        for critic, key in ((networks.value_network, keys[1]), (networks.cost_value_network, keys[2])):
            params = critic.init(key)
            value = lambda obs: critic.apply(normalizer, params, obs)
            assert value(observation).shape == (3,)
            assert not np.allclose(value(observation), value(other_context)), "a critic must see the context"


def test_scale_to_bounds_maps_omega_to_the_unit_box_and_a_degenerate_dimension_to_zero():
    space = C.suite_contexts("safe_goal_point").space
    bounds = space.bounds
    assert bounds.low == tuple(float(d.low) for d in space.dimensions) and len(bounds.low) == space.size
    corners = jnp.stack([space.low, space.high])
    np.testing.assert_allclose(np.asarray(space.scale(corners)), np.stack([-np.ones(space.size), np.ones(space.size)]), atol=1e-6)
    np.testing.assert_allclose(np.asarray(space.scale((space.low + space.high) / 2)), 0.0, atol=1e-6)
    degenerate = C.scale_to_bounds(jnp.asarray([[3.0, 7.0]]), C.Bounds(low=(3.0, 0.0), high=(3.0, 10.0)))
    np.testing.assert_allclose(np.asarray(degenerate), [[0.0, 0.4]], atol=1e-6)
    with pytest.raises(ValueError, match="exceeds"):
        C.Bounds(low=(1.0,), high=(0.0,))
    with pytest.raises(ValueError, match="lower bounds"):
        C.Bounds(low=(0.0, 0.0), high=(1.0,))


def test_a_told_policy_trained_on_one_context_still_sees_every_other_context_in_scale():
    """The defect of 2026-10-15: under `staged` the running normaliser sees one context per
    stage, its std collapses to the floor, and any other context normalises to ~1e6. The
    context part must not go through the running normaliser at all: whatever the statistics
    say, the policy's output on another context stays finite and the context still matters."""
    from training.acme import running_statistics, specs

    suite = C.suite_contexts("safe_goal_point")
    space = suite.space
    level_1, level_3 = suite.level_contexts[1], suite.level_contexts[3]
    observation_size = {C.SENSORS_KEY: (4,), C.CONTEXT_OBSERVATION_KEY: (space.size,)}
    statistics = running_statistics.init_state({key: specs.Array(shape, jnp.float32) for key, shape in observation_size.items()})
    one_context_batch = {C.SENSORS_KEY: jax.random.normal(jax.random.PRNGKey(0), (512, 4)), C.CONTEXT_OBSERVATION_KEY: jnp.tile(level_1, (512, 1))}
    for _ in range(20):
        statistics = running_statistics.update(statistics, one_context_batch)
    # the premise of the defect holds for the raw statistics: level 3 would normalise to ~1e6
    raw = running_statistics.normalize({C.SENSORS_KEY: jnp.zeros((1, 4)), C.CONTEXT_OBSERVATION_KEY: level_3[None]}, statistics)
    assert float(jnp.abs(raw[C.CONTEXT_OBSERVATION_KEY]).max()) > 1e5

    networks = C.Learner(constraint="scalar", policy_context="told").train_kwargs(space)["network_factory"](
        observation_size, 2, preprocess_observations_fn=running_statistics.normalize
    )
    params = networks.policy_network.init(jax.random.PRNGKey(1))
    sensors = jnp.zeros((1, 4))
    on_level_1 = networks.policy_network.apply(statistics, params, {C.SENSORS_KEY: sensors, C.CONTEXT_OBSERVATION_KEY: level_1[None]})
    on_level_3 = networks.policy_network.apply(statistics, params, {C.SENSORS_KEY: sensors, C.CONTEXT_OBSERVATION_KEY: level_3[None]})
    assert np.all(np.isfinite(np.asarray(on_level_3)))
    assert float(jnp.abs(on_level_3).max()) < 10.0, "a fresh policy on an unseen context must not be saturated"
    assert not np.allclose(np.asarray(on_level_1), np.asarray(on_level_3)), "the told policy must still see the context"


PER_CONTEXT_FORMS = ("network", "additive")


def _per_context_constraint(form: str, space: C.ContextSpace):
    return C.Learner(constraint=form, policy_context="told").per_context_constraint(space)


@pytest.mark.parametrize("form", PER_CONTEXT_FORMS)
def test_per_context_constraint_starts_as_the_scalar_and_moves_by_the_integrator(form):
    """Either function class: the constant `initial_lambda` everywhere at the start; one update
    with every transition in one context moves λ there by rate·(mean cost − budget), the
    scalar's step; λ never drops below 0; the metrics describe the batch."""
    space = C.suite_contexts(LIFT).space
    constraint = _per_context_constraint(form, space)
    assert constraint.extra_fields == (C.TRANSITION_CONTEXT_KEY,)
    multipliers = constraint.initial_multipliers(0.5)
    everywhere = jnp.asarray(space.integer_points, jnp.float32)  # the 16 contexts of {0,1}^4
    np.testing.assert_allclose(np.asarray(constraint.multipliers_at(multipliers, everywhere)), 0.5, atol=1e-6)
    # one context over budget by 1 per step at rate 0.25: its λ should land near 0.5 + 0.25 = 0.75
    context = jnp.ones((64, 8, 4))
    state_extras = {C.TRANSITION_CONTEXT_KEY: context}
    costs = jnp.full((64, 8), 2.0)
    updated, metrics = constraint.update(multipliers, costs, state_extras, budget_per_step=1.0, rate=0.25)
    per_transition = np.asarray(constraint.per_transition(updated, state_extras))
    assert per_transition.shape == (64, 8)
    assert per_transition.mean() == pytest.approx(0.75, abs=0.05)
    assert metrics["lambda_lagr"] == pytest.approx(per_transition.mean()) and metrics["lambda_lagr/std"] == pytest.approx(0.0, abs=1e-4)
    assert metrics["lambda_lagr/fit_error"] < 0.05 and metrics["lambda_lagr/target_spread"] == pytest.approx(0.0)  # one cost: no scatter
    # under budget for many rounds: λ falls towards 0 and never below
    for _ in range(30):
        updated, metrics = constraint.update(updated, jnp.zeros((64, 8)), state_extras, budget_per_step=1.0, rate=0.25)
    assert 0.0 <= float(metrics["lambda_lagr/min"]) and float(metrics["lambda_lagr"]) < 0.05


@pytest.mark.parametrize("form", PER_CONTEXT_FORMS)
def test_per_context_constraint_tracks_a_scalar_and_separates_contexts(form):
    """Forty rounds of one context over budget: λ there follows the scalar integrator to within
    a tenth; then two contexts with opposite violations in one round move in opposite directions."""
    from training.agents.ppo_lag.constraint import ScalarConstraint

    space = C.suite_contexts(LIFT).space
    constraint, scalar = _per_context_constraint(form, space), ScalarConstraint()
    function, value = constraint.initial_multipliers(0.1), scalar.initial_multipliers(0.1)
    first, second = jnp.zeros((32, 8, 4)), jnp.ones((32, 8, 4))
    costs = jnp.full((32, 8), 0.085)  # over a budget of 0.025 by 0.06 per step
    update = jax.jit(lambda f, c, ctx: constraint.update(f, c, {C.TRANSITION_CONTEXT_KEY: ctx}, 0.025, 1.0))
    for _ in range(40):
        function, metrics = update(function, costs, first)
        value, _ = scalar.update(value, costs, {}, 0.025, 1.0)
    assert float(metrics["lambda_lagr"]) == pytest.approx(float(value[0]), abs=0.1)
    at_first, at_second = (float(constraint.multipliers_at(function, ctx[0, :1])[0]) for ctx in (first, second))
    both = jnp.concatenate([first, second])
    opposite = jnp.concatenate([jnp.full((32, 8), 0.01), jnp.full((32, 8), 0.065)])  # first under, second over
    function, metrics = update(function, opposite, both)
    now_first, now_second = (float(constraint.multipliers_at(function, ctx[0, :1])[0]) for ctx in (first, second))
    assert now_first < at_first and now_second > at_second
    assert float(metrics["lambda_lagr/std"]) > 0.0


def test_network_constraint_needs_a_positive_start_and_keeps_unseen_contexts_between():
    """softplus never reaches 0, so the network form refuses λ₀ = 0; an unvisited context's λ
    after one round lies between its start and the visited context's new value."""
    space = C.suite_contexts(LIFT).space
    constraint = C.NetworkConstraint(space)
    with pytest.raises(ValueError, match="cannot start at 0"):
        constraint.initial_multipliers(0.0)
    multipliers = constraint.initial_multipliers(0.5)
    state_extras = {C.TRANSITION_CONTEXT_KEY: jnp.ones((64, 8, 4))}
    updated, _ = constraint.update(multipliers, jnp.full((64, 8), 2.0), state_extras, budget_per_step=1.0, rate=0.25)
    visited = float(constraint.multipliers_at(updated, jnp.ones((1, 4)))[0])
    unvisited = float(constraint.multipliers_at(updated, jnp.zeros((1, 4)))[0])
    assert 0.0 < unvisited < visited


def test_additive_constraint_is_monotone_in_difficulty_and_transfers_across_factors():
    """λ = f(Σ_i h_i(ω_i)), h_i nondecreasing in the factor's difficulty direction, f nondecreasing.

    On goal-point: training on contexts that differ only in the cylinder count, with cost rising
    with it, makes λ nondecreasing in cylinders and 0 on the slack contexts; the difficulty learnt
    then carries to contexts with other factors set (no interactions), monotonically; goal_size is
    harder when *lower*, so cost at small goals raises λ at small goals, not large ones; λ₀ = 0 is
    allowed; and the fit reports which factors carry the difficulty."""
    suite = C.suite_contexts("safe_goal_point")
    space = suite.space
    constraint = C.AdditiveConstraint(space)
    steps_per_count = sum(int(d.high) - int(d.low) for d in space.integer_dimensions)  # one increment per value past the easiest
    assert constraint.num_difficulty_features == steps_per_count + (CONTINUOUS_BINS - 1)  # counts + goal-size inner bin edges
    multipliers = constraint.initial_multipliers(0.0)
    at = lambda m, **values: float(constraint.multipliers_at(m, space.encode(**values)[None])[0])
    base = dict(active_cylinders=0, active_collidable_cylinders=0, active_cubes=0, active_collidable_cubes=0, goal_size=0.2)
    assert at(multipliers, **base) == 0.0
    easiest = space.encode(**{**base, "goal_size": 0.3})
    hardest = space.encode(active_cylinders=12, active_collidable_cylinders=12, active_cubes=12, active_collidable_cubes=12, goal_size=0.1)
    np.testing.assert_allclose(np.asarray(constraint.difficulty_at(multipliers, jnp.stack([easiest, hardest]))), [0.0, 1.0], atol=1e-6)

    # cost rises with cylinders, everything else fixed: 0, 4, 8 cylinders at per-step cost 0.02, 0.05, 0.08 vs budget 0.025
    contexts = jnp.stack([space.encode(**{**base, "active_cylinders": d}) for d in (0.0, 4.0, 8.0)])  # [3, 5]
    contexts = jnp.repeat(contexts[:, None, :], 16, axis=1)  # [3, 16, 5]
    costs = jnp.repeat(jnp.asarray([[0.02], [0.05], [0.08]]), 16, axis=1)
    update = jax.jit(lambda m, c, ctx: constraint.update(m, c, {C.TRANSITION_CONTEXT_KEY: ctx}, 0.025, 1.0))
    for _ in range(20):
        multipliers, metrics = update(multipliers, costs, contexts)
    at_0, at_4, at_8 = (at(multipliers, **{**base, "active_cylinders": d}) for d in (0, 4, 8))
    assert at_0 == pytest.approx(0.0, abs=0.02) and at_0 <= at_4 < at_8  # slack context at 0; monotone
    assert at_8 == pytest.approx(20 * (0.08 - 0.025), rel=0.1)  # the integrator's value after 20 rounds
    assert metrics["lambda_lagr/fit_error"] < 0.05
    # monotone across the whole factor, visited values or not
    along = [at(multipliers, **{**base, "active_cylinders": d}) for d in range(13)]
    assert all(a <= b + 1e-5 for a, b in zip(along, along[1:]))
    # the cylinder difficulty carries into contexts with other factors set, never lowering λ
    assert at(multipliers, **{**base, "active_cylinders": 8, "active_collidable_cylinders": 3}) >= at_8 - 1e-5
    # the fit says the cylinders carry the difficulty it has seen
    shares = constraint.factor_shares(multipliers)
    assert max(shares, key=lambda name: float(shares[name])) == "active_cylinders"
    # goal_size is harder when lower: cost at a small goal raises λ at small goals, not at large ones
    small, large = space.encode(**{**base, "goal_size": 0.11}), space.encode(**{**base, "goal_size": 0.29})
    both = jnp.stack([jnp.repeat(small[None], 16, 0), jnp.repeat(large[None], 16, 0)])
    for _ in range(5):
        multipliers, metrics = update(multipliers, jnp.asarray([[0.1] * 16, [0.0] * 16]), both)
    assert at(multipliers, **{**base, "goal_size": 0.11}) > at(multipliers, **{**base, "goal_size": 0.29}) + 0.1
    assert float(metrics["lambda_lagr/baseline"]) == pytest.approx(0.0, abs=0.02)


def test_scalar_constraint_is_the_published_rule_and_broadcasts():
    from training.agents.ppo_lag.constraint import ScalarConstraint

    constraint = ScalarConstraint()
    multipliers = constraint.initial_multipliers(0.5)
    costs = jnp.asarray([[0.0, 0.0, 0.0], [1.0, 2.0, 3.0]])  # batch mean 1.0
    updated, metrics = constraint.update(multipliers, costs, {}, budget_per_step=0.5, rate=2.0)
    np.testing.assert_allclose(np.asarray(updated), [1.5])
    assert metrics == {"lambda_lagr": updated}
    assert constraint.per_transition(updated, {}).shape == (1,)


def test_lagrangian_advantage_takes_lambda_per_transition_in_the_datas_layout():
    """`_time_first` lays λ out [T, B] like the swapped data; a [1] λ broadcasts to the same."""
    from training.agents.ppo_lag.losses import _time_first

    batch, time = 2, 3
    per_transition = jnp.arange(batch * time, dtype=jnp.float32).reshape(batch, time)
    np.testing.assert_array_equal(np.asarray(_time_first(per_transition, (batch, time))), np.asarray(per_transition).T)
    np.testing.assert_array_equal(np.asarray(_time_first(jnp.asarray([2.0]), (batch, time))), np.full((time, batch), 2.0))


def test_learner_names_the_arm_and_hands_the_trainer_what_realises_it():
    told = C.Learner(constraint="additive", policy_context="told")
    assert C.arm_label("staged:1,2,3", told) == "staged:1,2,3 additive told"
    assert C.run_name_fragment("staged:1,2,3", told) == "staged123_additive_told"
    assert C.arm_label("uniform", C.DEFAULT_LEARNER) == "uniform scalar blind"
    kwargs = told.train_kwargs(C.suite_contexts(LIFT).space)
    assert isinstance(kwargs["lagrange_constraint"], C.AdditiveConstraint)
    assert kwargs["network_factory"].keywords == {
        "policy_observes_context": True, "context_bounds": {"low": (0.0,) * 4, "high": (1.0,) * 4},
    }
    assert set(C.DEFAULT_LEARNER.train_kwargs(C.suite_contexts(LIFT).space)) == {"network_factory"}
    network = C.Learner(constraint="network", policy_context="blind")
    assert isinstance(network.train_kwargs(C.suite_contexts(ENV_NAME).space)["lagrange_constraint"], C.NetworkConstraint)  # any Ω
    with pytest.raises(ValueError, match="holds no per-context constraint"):
        C.DEFAULT_LEARNER.per_context_constraint(C.suite_contexts(LIFT).space)
    with pytest.raises(ValueError):
        C.Learner(constraint="per_episode", policy_context="told")
    with pytest.raises(ValueError, match="need a context distribution"):
        C.context_training_setup(ENV_NAME, C.NO_DISTRIBUTION, "level:3", told, num_timesteps=10, num_envs=1, batch_size=1, unroll_length=1, num_minibatches=1)
    assert C.context_training_setup(ENV_NAME, C.NO_DISTRIBUTION, "level:3", C.DEFAULT_LEARNER, num_timesteps=10, num_envs=1, batch_size=1, unroll_length=1, num_minibatches=1) is None
