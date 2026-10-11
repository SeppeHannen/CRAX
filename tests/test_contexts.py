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
        return lambda observation, key: (jax.random.uniform(key, (observation.shape[0], env.action_size), minval=-1.0, maxval=1.0), {})

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
