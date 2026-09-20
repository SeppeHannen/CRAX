"""Tests for training.contexts: context spaces, distributions, and per-episode
context sampling through ContextualAutoResetWrapper.

Run with: JAX_PLATFORMS=cpu pytest tests/test_contexts.py -v
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from crax import envs
from training import contexts as C
from training.agents.ppo.train import _strip_weak_type

ENV_NAME = "safe_velocity_ant"
NUM_SLOTS = 8
EPISODE_LENGTH = 25


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

    for level in (1, 2, 3):
        assert float(suite.level(level)[0]) == pytest.approx(get_threshold_for_level("ant", level))
    assert suite.levels().shape == (3, 1)


def test_uniform_samples_inside_space(suite):
    samples = suite.space.sample_uniform(jax.random.PRNGKey(0), 1000)
    assert samples.shape == (1000, 1)
    assert bool(jnp.all(suite.space.contains(samples)))


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
    reset, step = build(env, C.UniformDistribution(suite.space))
    state = reset(jax.random.split(jax.random.PRNGKey(0), NUM_SLOTS))
    history = [np.asarray(C.current_contexts(state))[:, 0]]
    for _ in range(2 * EPISODE_LENGTH + 1):
        state = step(state, zero_actions(env))
        history.append(np.asarray(C.current_contexts(state))[:, 0])
        # the env's view must always agree with the wrapper's
        np.testing.assert_allclose(np.asarray(state.info["velocity_threshold"]), history[-1])
    changes = (np.abs(np.diff(np.stack(history), axis=0)) > 1e-6).sum(axis=0)
    assert np.all(changes == 2), changes  # episodes end at t=25 and t=50
    assert int(state.info["steps"][0]) == (2 * EPISODE_LENGTH + 1) % EPISODE_LENGTH


def test_fixed_context_holds_across_resets(suite, env):
    reset, step = build(env, C.FixedContext(suite.space, suite.level(3)))
    state = reset(jax.random.split(jax.random.PRNGKey(1), NUM_SLOTS))
    for _ in range(EPISODE_LENGTH + 2):
        state = step(state, zero_actions(env))
    np.testing.assert_allclose(np.asarray(C.current_contexts(state))[:, 0], float(suite.level(3)[0]))


def test_staged_switches_without_recompiling(suite, env):
    from jax._src import monitoring
    from jax._src.dispatch import BACKEND_COMPILE_EVENT

    staged = C.StagedContexts.equal_split(suite.space, suite.levels(), total_rounds=3)
    reset, step = build(env, staged)
    state = _strip_weak_type(reset(jax.random.split(jax.random.PRNGKey(2), NUM_SLOTS)))
    params = C.current_parameters(state)
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
        feedback = C.EpisodeFeedback(
            contexts=C.current_contexts(state),
            returns=jnp.zeros(NUM_SLOTS),
            costs=jnp.zeros(NUM_SLOTS),
            lengths=jnp.full(NUM_SLOTS, EPISODE_LENGTH),
            round_index=round_index,
        )
        params = staged.update(params, feedback)
        state = _strip_weak_type(C.attach_parameters(state, params))

    assert trained_at == pytest.approx([float(suite.level(l)[0]) for l in (1, 2, 3)])
    assert substantial == [], f"stage switching recompiled the step program: {substantial}"


def test_log_probability(suite):
    uniform = C.UniformDistribution(suite.space)
    params = uniform.initialise(jax.random.PRNGKey(0))
    inside = suite.levels()
    outside = jnp.asarray([[0.0]])
    assert bool(jnp.all(jnp.isfinite(uniform.log_probability(params, inside))))
    assert bool(jnp.isneginf(uniform.log_probability(params, outside)[0]))

    fixed = C.FixedContext(suite.space, suite.level(2))
    fparams = fixed.initialise(jax.random.PRNGKey(0))
    lp = fixed.log_probability(fparams, inside)
    assert bool(jnp.isneginf(lp[0])) and float(lp[1]) == 0.0 and bool(jnp.isneginf(lp[2]))
