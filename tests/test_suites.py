"""Every suite with a registered context space works with the whole stack.

The invariant behind the dashboard is "every key a run logs is registered"; the
invariant behind the contexts package is "levels reproduce the difficulty
module and the environment reads its context". Both were built on
``safe_velocity_ant`` and are asserted here for every registered suite, on CPU,
without a training run: the keys come from the same two producers the trainer
uses — ``episode_metrics`` of the training wrapper stack (→ ``episodic/*``) and
``acting.Evaluator`` on the evaluation stack (→ ``evaluation/<distribution>/*``).

Run with: JAX_PLATFORMS=cpu pytest tests/test_suites.py -v
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import jax
import jax.numpy as jnp
import mujoco
import numpy as np
import pytest

from crax import envs
from crax.envs import context
from crax.envs.base import Env, State
from crax.envs.context import CONTEXT_KEY
from training import contexts as C
from training import dashboard
from training.acting import Evaluator
from training.agents.ppo.train import _evaluation_metric_name
from training.contexts.registry import LEVELS

SUITES = C.registered_environments()
NUM_SLOTS = 2
EPISODE_LENGTH = 5

# Button's Ω contains layouts its arena cannot hold (docs/acl/README.md, Known defects):
# Uniform(Ω) reaches them and the reset raises. Strict, so the day Ω is fixed this flips.
_UNIFORM_BUTTON_IS_INFEASIBLE = pytest.mark.xfail(
    strict=True, raises=Exception,
    reason="safe_button_point: Uniform(Ω) samples layouts that do not fit the 2 m square (README, Known defects)",
)
SUITES_UNDER_UNIFORM = [
    pytest.param(name, marks=_UNIFORM_BUTTON_IS_INFEASIBLE) if name == "safe_button_point" else name for name in SUITES
]


def _uniform_training_stack(env_name: str) -> Tuple[Env, Env, State]:
    """The suite's env, the training wrapper stack with Uniform(Ω), and its reset state."""
    suite = C.suite_contexts(env_name)
    env = envs.get_environment(env_name, level=1)
    wrapped = C.wrap_for_context_training(env, C.UniformDistribution(suite.space), episode_length=EPISODE_LENGTH)
    state = jax.jit(wrapped.reset)(jax.random.split(jax.random.PRNGKey(0), NUM_SLOTS))
    return env, wrapped, state


def _logged_keys(env_name: str) -> List[str]:
    """The keys a context run of this suite logs, produced the way the trainer produces them."""
    env, wrapped, state = _uniform_training_stack(env_name)
    state = jax.jit(wrapped.step)(state, jnp.zeros((NUM_SLOTS, env.action_size)))
    keys = [f"episodic/{name}" for name in state.info["episode_metrics"]]

    def zero_policy(_params):
        return lambda observation, key: (jnp.zeros((observation.shape[0], env.action_size)), {})

    evaluator = Evaluator(wrapped, zero_policy, num_eval_envs=NUM_SLOTS, episode_length=EPISODE_LENGTH,
                          action_repeat=1, key=jax.random.PRNGKey(1))
    for distribution in (C.DEPLOYMENT_EVALUATION, C.UNIFORM_EVALUATION):
        keys += [_evaluation_metric_name(key, distribution) for key in evaluator.run_evaluation(None, training_metrics={})]
    return keys


@pytest.mark.parametrize("env_name", SUITES_UNDER_UNIFORM)
def test_every_key_the_suite_logs_is_registered(env_name):
    keys = _logged_keys(env_name)
    assert "episodic/cost" in keys and f"evaluation/{C.DEPLOYMENT_EVALUATION}/episode_reward" in keys
    unregistered = []
    for key in keys:
        try:
            dashboard.registered(key)
        except dashboard.UnregisteredMetric:
            unregistered.append(key)
    assert not unregistered, f"{env_name} logs keys the dashboard does not know: {unregistered}"


@pytest.mark.parametrize("env_name", SUITES)
def test_omega_names_the_environments_context_parameters(env_name):
    suite = C.suite_contexts(env_name)
    env = envs.get_environment(env_name, level=1).unwrapped
    assert suite.space.names == env.CONTEXT_PARAMETERS


@pytest.mark.parametrize("env_name", SUITES)
def test_levels_differ_and_a_plain_reset_lands_in_the_levels_context(env_name):
    """The three levels are three distinct points of Ω, and a level-``n`` environment's plain
    ``reset`` writes exactly ``level(n)`` into the state — the stock run is a context run."""
    suite = C.suite_contexts(env_name)
    points = [np.asarray(suite.level(level)) for level in LEVELS]
    assert len({tuple(np.round(point, 6)) for point in points}) == len(LEVELS), points
    for level in LEVELS:
        env = envs.get_environment(env_name, level=level).unwrapped
        state = env.reset(jax.random.PRNGKey(0))
        np.testing.assert_allclose(np.asarray(state.info[CONTEXT_KEY]), np.asarray(suite.level(level)), err_msg=f"{env_name} level {level}")


def _roll(env: Env, state: State, num_steps: int) -> Dict[str, np.ndarray]:
    """Fixed random actions from ``state``; the per-step metrics stacked."""
    step = jax.jit(env.step)
    key = jax.random.PRNGKey(7)
    history = []
    for _ in range(num_steps):
        key, action_key = jax.random.split(key)
        state = step(state, jax.random.uniform(action_key, (env.action_size,), minval=-1.0, maxval=1.0))
        history.append({name: np.asarray(value) for name, value in state.metrics.items()})
    return {name: np.stack([h[name] for h in history]) for name in history[0]}


@pytest.mark.parametrize("env_name", SUITES)
def test_the_context_is_the_only_place_the_knob_is_read(env_name):
    """The invariant of crax/envs/context.py: a level-1 environment reset into the
    level-3 context is indistinguishable, step by step, from a level-3 environment.
    If any code path still read the constructor's constant, the two would differ."""
    suite = C.suite_contexts(env_name)
    easy, hard = (envs.get_environment(env_name, level=level) for level in (1, 3))
    assert not np.allclose(np.asarray(suite.level(1)), np.asarray(suite.level(3))), "levels 1 and 3 must differ for this test to mean anything"
    reset_key = jax.random.PRNGKey(0)
    from_easy_env = _roll(easy, easy.reset_with_context(reset_key, suite.level(3)), num_steps=15)
    from_hard_env = _roll(hard, hard.reset(reset_key), num_steps=15)
    for name in from_hard_env:
        np.testing.assert_allclose(from_easy_env[name], from_hard_env[name], rtol=1e-5, atol=1e-6, err_msg=f"{env_name}: {name}")


@pytest.mark.parametrize("env_name", SUITES)
def test_uniform_contexts_differ_per_slot(env_name):
    """Uniform(Ω) gives slots different contexts (for a discrete Ω, at least two distinct ones among a few slots)."""
    suite = C.suite_contexts(env_name)
    env = envs.get_environment(env_name, level=1)
    wrapped = C.wrap_for_context_training(env, C.UniformDistribution(suite.space), episode_length=EPISODE_LENGTH)
    state = jax.jit(wrapped.reset)(jax.random.split(jax.random.PRNGKey(0), 8))
    contexts = np.asarray(C.current_contexts(state))
    assert len({tuple(np.round(row, 4)) for row in contexts}) >= 2


@pytest.mark.parametrize("env_name", SUITES)
def test_task_description_states_whether_episodes_can_end_early(env_name):
    """The dashboard reads episode length through ``task.episode_ends_early``.

    The probe — random actions for 250 steps — is one-sided: one early ``done``
    refutes a claim of "never ends early", so that claim is checked fully; a
    claim of "can end early" is confirmed when the probe sees it and cannot be
    refuted when it does not (a wheeled robot rarely flips under random torques).
    The asymmetry is the right one: the dashboard sentence that would mislead is
    "this panel is a constant" said of a suite where it is not.
    """
    suite = C.suite_contexts(env_name)
    env = envs.get_environment(env_name, level=1)
    step = jax.jit(env.step)
    key = jax.random.PRNGKey(0)
    state = env.reset(key)
    ended_early = False
    for _ in range(250):
        key, action_key = jax.random.split(key)
        state = step(state, jax.random.uniform(action_key, (env.action_size,), minval=-1.0, maxval=1.0))
        if bool(state.done):
            ended_early = True
            break
    assert suite.task.episode_ends_early or not ended_early, f"{env_name} ended an episode early but its task says episodes never do"


def _wall_geoms(env: Env) -> List[str]:
    """The arena fence's geom names (``crax/envs/arena.py``); empty for an open arena."""
    model = env.sys.mj_model
    names = (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, index) or "" for index in range(model.ngeom))
    return [name for name in names if name.startswith("wall_")]


FENCED_SUITES = [name for name in SUITES if _wall_geoms(envs.get_environment(name, level=1).unwrapped)]


@pytest.mark.parametrize("env_name", FENCED_SUITES)
def test_the_fence_is_not_among_the_hazards(env_name):
    """The invariant of crax/envs/arena.py, structurally: the four walls are world-body
    geoms, and every hazard the environment knows is one of its hazard specs' — none
    is a wall, so nothing that iterates over hazards (cost, lidar, compass, placement)
    can reach the fence."""
    env = envs.get_environment(env_name, level=1).unwrapped
    assert len(_wall_geoms(env)) == 4
    assert all(hazard.hazard_type != "rect" for hazard in env._hazard_manager.hazards)
    assert env._num_hazards == sum(group.count for group in env._hazard_manager.groups)


def test_driving_into_the_fence_is_free_and_invisible():
    """The invariant of crax/envs/arena.py, behaviourally, on the one fenced suite whose
    arena can be emptied (goal: activate no hazard). Full thrust at a wall for 300
    steps: the robot stays in, pays nothing, and its hazard lidar stays dark."""
    env = envs.get_environment("safe_goal_point", level=1).unwrapped
    suite = C.suite_contexts("safe_goal_point")
    counts = {name: 0.0 for name in suite.space.names if name.startswith("active_")}
    state = jax.jit(env.reset_with_context)(jax.random.PRNGKey(0), context.encode(env, **counts, goal_size=0.2))
    step = jax.jit(env.step)
    thrust = jnp.zeros((env.action_size,)).at[0].set(1.0)
    hazard_lidar = slice(12 + env._lidar_num_bins, 12 + 2 * env._lidar_num_bins)
    arena_half_width = env._placement_extents[2] + 0.5  # the fence stands 0.5 m outside the placement square
    for _ in range(300):
        state = step(state, thrust)
        position = np.asarray(state.pipeline_state.xpos[env._agent_body][:2])
        assert np.all(np.abs(position) < arena_half_width), f"the robot left the arena at {position}"
        assert float(state.info["cost"]) == 0.0
        assert float(jnp.max(state.obs[hazard_lidar])) == 0.0
    assert abs(position[0]) > env._placement_extents[2], "the robot never reached the fence"
