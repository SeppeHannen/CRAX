"""The episode review (docs/acl/design/episode_review.md): recording the
evaluation episodes a run already simulates, with the learner's view of every
step, and serving them to the review page.

Run with: JAX_PLATFORMS=cpu MUJOCO_GL=egl pytest tests/test_episode_review.py -v
(the frame tests need a GL context; without MUJOCO_GL they are skipped)
"""
from __future__ import annotations

import io
import json
import os
import pathlib
import tempfile
import threading
import urllib.request
from typing import Any, Dict, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from PIL import Image

from crax import envs
from training import contexts as C
from training import types
from training.acme import running_statistics, specs
from training.agents.ppo import networks as ppo_networks
from training.agents.ppo_lag import losses as ppo_lag_losses
from training.agents.ppo_lag.train import train as train_ppo_lag
from training.evaluation import TRANSITION_SCENE_KEY, EpisodeRecord, EpisodeRecorder, LearnerView, SceneWrapper, records_under
from training.evaluation.scene import Scene
from training.evaluation.recorder import EvaluationRecording, _episode_length
from training.review.library import Library
from training.review.policy_view import SCALE
from training.review.server import ReviewServer

ENV_NAME = "safe_velocity_ant"  # on the MJX backend: the scene is its pipeline state's pose fields
RENDERING = pytest.mark.skipif("MUJOCO_GL" not in os.environ, reason="rendering needs a GL context (MUJOCO_GL=egl)")


def _record(length: int = 5, **overrides: Any) -> EpisodeRecord:
    per_step = lambda *shape: np.arange(np.prod((length, *shape)), dtype=np.float32).reshape(length, *shape)
    fields: Dict[str, Any] = dict(
        env_name=ENV_NAME, run_name="run", arm="uniform", evaluation_name="deployment", environment_steps=640, episode_index=1,
        context=np.asarray([1.5], np.float32), qpos=per_step(15), qvel=per_step(14), mocap_pos=per_step(0, 3), mocap_quat=per_step(0, 4),
        observation=per_step(27), images={}, action=per_step(8), reward=per_step(), cost=per_step(), value=per_step(), cost_value=per_step(),
        lagrange_multiplier=per_step(), reward_advantage=per_step(), cost_advantage=per_step(), lagrangian_advantage=per_step(),
    )
    return EpisodeRecord(**{**fields, **overrides})


# --------------------------------------------------------------------------- #
# The record
# --------------------------------------------------------------------------- #


def test_record_round_trips_through_its_file_and_names_itself(tmp_path):
    record = _record()
    path = record.save(tmp_path)
    assert path.name == "000000000640_deployment_01.episode.npz"
    loaded = EpisodeRecord.load(path)
    assert loaded.key == record.key == ("uniform", 640, "deployment", 1)
    for name in ("qpos", "observation", "reward", "lagrangian_advantage", "context"):
        np.testing.assert_array_equal(getattr(loaded, name), getattr(record, name))
    assert loaded.length == 5 and loaded.episode_return == pytest.approx(10.0) and loaded.episode_cost == pytest.approx(10.0)
    assert [r.key for r in records_under(tmp_path)] == [record.key]


def test_record_keeps_the_frames_the_policy_saw_by_part_name():
    """A vision run's record holds each image part as [T, H, W, C] uint8 under the observation
    part's name (which has a '/'); a state-based run's holds none."""
    frames = np.arange(5 * 4 * 4 * 6, dtype=np.uint8).reshape(5, 4, 4, 6)  # two stacked RGB frames
    record = _record(images={"pixels/vision": frames})
    with tempfile.TemporaryDirectory() as directory:
        loaded = EpisodeRecord.load(record.save(pathlib.Path(directory)))
    assert tuple(loaded.images) == ("pixels/vision",)
    np.testing.assert_array_equal(loaded.images["pixels/vision"], frames)
    assert loaded.images["pixels/vision"].dtype == np.uint8
    with pytest.raises(ValueError, match="uint8"):
        _record(images={"pixels/vision": frames.astype(np.float32)})
    with pytest.raises(ValueError, match=r"T=5"):
        _record(images={"pixels/vision": frames[:4]})


def test_record_refuses_arrays_of_the_wrong_length_or_shape():
    with pytest.raises(ValueError, match="leading axis"):
        _record(cost=np.zeros(4, np.float32))
    with pytest.raises(ValueError, match="one value per step"):
        _record(value=np.zeros((5, 1), np.float32))
    with pytest.raises(ValueError, match="one vector"):
        _record(context=np.zeros((1, 1), np.float32))
    with pytest.raises(ValueError, match="at least one step"):
        _record(length=0)


def test_episode_length_is_up_to_and_including_the_first_done():
    assert _episode_length(np.asarray([1.0, 1.0, 0.0, 1.0, 0.0])) == 3
    assert _episode_length(np.asarray([0.0, 1.0])) == 1
    assert _episode_length(np.ones(7)) == 7


# --------------------------------------------------------------------------- #
# The scene
# --------------------------------------------------------------------------- #


def test_scene_wrapper_writes_the_scene_a_transition_started_from_and_refuses_other_backends():
    """After reset: the reset state's scene. After a step: the scene of the state the step was
    taken from, so a record's row t shows the state the policy saw and acted in."""
    env = envs.get_environment(ENV_NAME, level=1, backend="mjx")
    wrapped = SceneWrapper(env)
    initial = jax.jit(wrapped.reset)(jax.random.PRNGKey(0))
    np.testing.assert_array_equal(np.asarray(initial.info[TRANSITION_SCENE_KEY].qpos), np.asarray(initial.pipeline_state.qpos))
    stepped = jax.jit(wrapped.step)(initial, jnp.ones((env.action_size,)))
    assert not np.allclose(np.asarray(stepped.pipeline_state.qvel), np.asarray(initial.pipeline_state.qvel)), "the step must move the state"
    np.testing.assert_array_equal(np.asarray(stepped.info[TRANSITION_SCENE_KEY].qvel), np.asarray(initial.pipeline_state.qvel))
    assert stepped.info[TRANSITION_SCENE_KEY].mocap_pos.shape == (env.sys.mj_model.nmocap, 3)
    with pytest.raises(ValueError, match="mjx"):
        SceneWrapper(envs.get_environment(ENV_NAME, level=1))  # the suite's default backend is `generalized`


# --------------------------------------------------------------------------- #
# The learner's view is the loss's arithmetic
# --------------------------------------------------------------------------- #


def _batch(key: jax.Array, batch: int, steps: int, obs: int, actions: int) -> types.Transition:
    keys = jax.random.split(key, 6)
    return types.Transition(
        observation=jax.random.normal(keys[0], (batch, steps, obs)),
        action=jax.random.normal(keys[1], (batch, steps, actions)),
        reward=jax.random.normal(keys[2], (batch, steps)),
        discount=jnp.ones((batch, steps)).at[0, 2].set(0.0),  # one episode ends early
        next_observation=jax.random.normal(keys[3], (batch, steps, obs)),
        extras={
            "policy_extras": {"raw_action": jax.random.normal(keys[4], (batch, steps, actions)), "log_prob": jnp.zeros((batch, steps))},
            "state_extras": {"truncation": jnp.zeros((batch, steps)).at[:, -1].set(1.0), "cost": jax.random.uniform(keys[5], (batch, steps))},
        },
    )


def test_learner_view_is_the_learning_signals_values_and_unstandardised_lagrangian():
    """`compute_learner_view` and `compute_learning_signals` share `critic_estimates`: the same
    advantages reach the record and the round hook; the Lagrangian one is raw in the record."""
    obs, actions = 6, 2
    network = ppo_networks.make_ppo_networks(obs, actions, running_statistics.normalize, cost_value_hidden_layer_sizes=(16,))
    keys = jax.random.split(jax.random.PRNGKey(3), 4)
    params = ppo_lag_losses.PPONetworkParams(
        policy=network.policy_network.init(keys[0]), value=network.value_network.init(keys[1]),
        cost_value=network.cost_value_network.init(keys[2]), encoder=None,
    )
    normalizer = running_statistics.init_state(specs.Array((obs,), jnp.float32))
    data = _batch(keys[3], batch=3, steps=7, obs=obs, actions=actions)
    lam = jnp.asarray([0.7])
    constants = dict(ppo_network=network, discounting=0.9, reward_scaling=0.5, gae_lambda=0.95)
    view = ppo_lag_losses.compute_learner_view(params, normalizer, data, lam, **constants)
    signals = ppo_lag_losses.compute_learning_signals(params, normalizer, data, lam, normalize_advantage=False, **constants)
    for field in (view.value, view.cost_value, view.reward_advantage, view.cost_advantage, view.lagrangian_advantage, view.lagrange_multiplier):
        assert field.shape == (3, 7)
    np.testing.assert_allclose(view.reward_advantage, signals[ppo_lag_losses.Signal.REWARD_ADVANTAGE], rtol=1e-6)
    np.testing.assert_allclose(view.cost_advantage, signals[ppo_lag_losses.Signal.COST_ADVANTAGE], rtol=1e-6)
    np.testing.assert_allclose(view.lagrangian_advantage, view.reward_advantage - 0.7 * view.cost_advantage, rtol=1e-6)
    np.testing.assert_allclose(view.lagrange_multiplier, 0.7)
    # the values are the critics applied to the observations
    expected_value = jax.vmap(lambda o: network.value_network.apply(normalizer, params.value, o))(data.observation)
    np.testing.assert_allclose(view.value, expected_value, rtol=1e-6)
    # an episode that ended at step 2 does not see what came after: its advantages before the end are
    # those of the same data with everything after the end replaced
    altered = data._replace(reward=data.reward.at[0, 3:].set(100.0))
    other = ppo_lag_losses.compute_learner_view(params, normalizer, altered, lam, **constants)
    np.testing.assert_allclose(other.reward_advantage[0, :3], view.reward_advantage[0, :3], rtol=1e-5)


# --------------------------------------------------------------------------- #
# End to end: a tiny run records, the library indexes, the server serves
# --------------------------------------------------------------------------- #

TINY = dict(num_envs=8, batch_size=8, num_minibatches=2, unroll_length=5, num_updates_per_batch=1, episode_length=20,
            num_eval_envs=4, num_evals=3, learning_rate=1e-3, normalize_observations=True)
STEPS_PER_ROUND = 8 * 5 * 2
NUM_TIMESTEPS = STEPS_PER_ROUND * 6


LEARNERS = {
    "scalar_blind": C.Learner(constraint="scalar", policy_context="blind"),
    "additive_told": C.Learner(constraint="additive", policy_context="told"),
}


def _record_a_run(directory: pathlib.Path, learner: C.Learner) -> C.ContextTrainingSetup:
    setup = C.context_training_setup(ENV_NAME, "uniform", "level:3", learner, num_timesteps=NUM_TIMESTEPS, num_envs=8,
                                     batch_size=8, unroll_length=5, num_minibatches=2)
    recorder = EpisodeRecorder(directory, ENV_NAME, "tiny_run", setup.arm, num_episodes=2, sensors_key=C.SENSORS_KEY, context_key=C.TRANSITION_CONTEXT_KEY,
                               image_keys=())
    assert recorder.recording == EvaluationRecording(2, ("cost", "truncation", TRANSITION_SCENE_KEY, C.TRANSITION_CONTEXT_KEY))
    train_ppo_lag(environment=envs.get_environment(ENV_NAME, level=1, backend="mjx"), num_timesteps=NUM_TIMESTEPS, seed=0,
                  progress_fn=lambda *_: None, safety_bound=25.0, initial_lambda_lagr=0.1, **TINY, **setup.train_kwargs(),
                  episode_recorder=recorder)
    return setup


@pytest.fixture(scope="module")
def recorded_run(tmp_path_factory) -> Tuple[pathlib.Path, C.ContextTrainingSetup]:
    """A run under the published learner (scalar λ, blind policy); what the review page is tested on."""
    directory = tmp_path_factory.mktemp("episodes")
    return directory, _record_a_run(directory, LEARNERS["scalar_blind"])


def test_records_under_a_per_context_constraint_carry_the_multiplier_of_each_episodes_context(tmp_path):
    """Under λ(ω) with a told policy the recorded λ is the constraint's lookup at the episode's
    context (so it differs between contexts), and the observation recorded is the sensors part
    of the {sensors, context} dict — the context is the record's own field."""
    setup = _record_a_run(tmp_path, LEARNERS["additive_told"])
    records = list(records_under(tmp_path))
    assert len(records) == 12
    for record in records:
        assert record.observation.shape[1] == 27, "the sensors, not the {sensors, context} dict"
        assert np.isfinite(record.lagrange_multiplier).all() and (record.lagrange_multiplier >= 0).all()
        np.testing.assert_allclose(record.lagrangian_advantage, record.reward_advantage - record.lagrange_multiplier * record.cost_advantage, atol=1e-5)
        # one episode, one context, one λ: the lookup is per transition but the context does not change within an episode
        np.testing.assert_allclose(record.lagrange_multiplier, record.lagrange_multiplier[0], atol=1e-6)
    # at step 0 the function is the constant `initial_lambda_lagr` everywhere (design doc §1): every record agrees
    initial = [float(r.lagrange_multiplier[0]) for r in records if r.environment_steps == 0]
    np.testing.assert_allclose(initial, 0.1, atol=1e-5)
    assert len({tuple(np.round(r.context, 3)) for r in records}) > 1, "uniform draws more than one context"


def test_every_evaluation_of_a_run_leaves_its_first_episodes_as_records(recorded_run):
    directory, setup = recorded_run
    records = list(records_under(directory))
    expected_keys = {(setup.arm, step, target, k) for step in (0, 3 * STEPS_PER_ROUND, 6 * STEPS_PER_ROUND)
                     for target in (C.DEPLOYMENT_EVALUATION, C.UNIFORM_EVALUATION) for k in (0, 1)}
    assert setup.arm == "uniform scalar blind"  # the arm names the learner too; records carry it
    assert {r.key for r in records} == expected_keys
    for record in records:
        assert record.env_name == ENV_NAME and record.run_name == "tiny_run"
        assert 1 <= record.length <= TINY["episode_length"]
        assert record.observation.shape[1] == 27 and record.action.shape[1] == 8 and record.qpos.shape[1] == 15
        assert bool(setup.suite.space.contains(jnp.asarray(record.context)[None])[0])
        np.testing.assert_allclose(record.lagrangian_advantage, record.reward_advantage - record.lagrange_multiplier * record.cost_advantage, atol=1e-5)
        assert all(np.isfinite(getattr(record, name)).all() for name in ("value", "cost_value", "reward_advantage", "cost_advantage"))
    deployment = [r for r in records if r.evaluation_name == C.DEPLOYMENT_EVALUATION]
    assert all(np.allclose(r.context, setup.suite.level(3)) for r in deployment)  # w is the point mass on level 3


def test_recorder_refuses_an_unroll_of_the_wrong_width(tmp_path):
    recorder = EpisodeRecorder(tmp_path, ENV_NAME, "run", "uniform", num_episodes=2, sensors_key=C.SENSORS_KEY, context_key=C.TRANSITION_CONTEXT_KEY, image_keys=())
    with pytest.raises(ValueError, match="recorded 1 episodes"):
        recorder.record("deployment", 0, _unroll(width=1), _view(width=1))


def test_recorder_keeps_the_named_image_parts_of_the_observation(tmp_path):
    """With image keys, each record holds that part's frames cut to the episode's length, as the
    observation dict had them; the sensors part stays the record's `observation`."""
    recorder = EpisodeRecorder(tmp_path, ENV_NAME, "run", "uniform", num_episodes=2, sensors_key=C.SENSORS_KEY,
                               context_key=C.TRANSITION_CONTEXT_KEY, image_keys=("pixels/vision",))
    frames = jnp.arange(3 * 2 * 4 * 4 * 3, dtype=jnp.uint8).reshape(3, 2, 4, 4, 3)
    unroll = _unroll(width=2, observation={C.SENSORS_KEY: jnp.zeros((3, 2, 4)), "pixels/vision": frames})
    paths = recorder.record("deployment", 0, unroll, _view(width=2))
    records = [EpisodeRecord.load(path) for path in paths]
    for k, record in enumerate(records):
        np.testing.assert_array_equal(record.images["pixels/vision"], np.asarray(frames[:, k]))
        assert record.observation.shape == (3, 4)


def _unroll(width: int, steps: int = 3, observation: Any = None) -> types.Transition:
    z = lambda *shape: jnp.zeros((steps, width, *shape))
    return types.Transition(observation=z(4) if observation is None else observation, action=z(2), reward=z(),
                            discount=jnp.ones((steps, width)), next_observation=z(4),
                            extras={"state_extras": {"cost": z(), "truncation": z(), C.TRANSITION_CONTEXT_KEY: z(1),
                                                     TRANSITION_SCENE_KEY: Scene(qpos=z(3), qvel=z(3), mocap_pos=z(0, 3), mocap_quat=z(0, 4))}})


def _view(width: int, steps: int = 3) -> LearnerView:
    z = jnp.zeros((width, steps))
    return LearnerView(value=z, cost_value=z, lagrange_multiplier=z, reward_advantage=z, cost_advantage=z, lagrangian_advantage=z)


def test_trainer_refuses_a_recorder_without_the_agents_learner_view():
    from training.agents.ppo import train as ppo_train

    recorder = EpisodeRecorder(pathlib.Path("/nonexistent"), ENV_NAME, "run", "uniform", num_episodes=1, sensors_key=C.SENSORS_KEY, context_key=C.TRANSITION_CONTEXT_KEY, image_keys=())
    with pytest.raises(ValueError, match="learner_view_fn"):
        ppo_train.train(environment=envs.get_environment(ENV_NAME, level=1, backend="mjx"), num_timesteps=80, episode_length=20,
                        num_envs=4, batch_size=4, num_minibatches=1, unroll_length=5, episode_recorder=recorder)


@pytest.fixture(scope="module")
def review_server(recorded_run):
    directory, _ = recorded_run
    library = Library.from_directories("tiny_group", [directory])
    server = ReviewServer(("127.0.0.1", 0), library)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield server, f"http://127.0.0.1:{server.server_address[1]}"
    server.shutdown()
    server.renderers.close()


def _get(url: str) -> Tuple[int, str, bytes]:
    with urllib.request.urlopen(url) as response:
        return response.status, response.headers["Content-Type"], response.read()


def test_library_summarises_without_loading_the_scene(recorded_run):
    directory, _ = recorded_run
    library = Library.from_directories("tiny_group", [directory])
    assert len(library.episodes) == 12 and library.env_names == (ENV_NAME,)
    summary = library.episodes[0]
    record = library.record(0)
    assert (summary.arm, summary.environment_steps, summary.evaluation_name, summary.episode_index) == record.key
    assert summary.length == record.length and summary.episode_return == pytest.approx(record.episode_return)
    assert summary.context == tuple(float(v) for v in record.context)
    with pytest.raises(IndexError):
        library.summary(12)
    with pytest.raises(FileNotFoundError):
        Library.from_directories("empty", [directory / "nowhere"])


def test_server_serves_the_page_the_library_and_an_episodes_traces(review_server):
    _, url = review_server
    status, content_type, body = _get(url + "/")
    assert status == 200 and content_type.startswith("text/html") and b"Episode review" in body
    status, _, body = _get(url + "/static/app.js")
    assert status == 200 and b"openEpisode" in body
    library = json.loads(_get(url + "/api/library")[2])
    assert library["group"] == "tiny_group" and len(library["episodes"]) == 12
    assert library["default_camera"] in library["environments"][ENV_NAME]["cameras"]
    assert library["policy_views"] == [], "a state-based run's records hold no policy view"
    assert library["environments"][ENV_NAME]["control_dt"] > 0
    episode = json.loads(_get(url + "/api/episode/0")[2])
    assert episode["length"] == len(episode["reward"]) == len(episode["value"]) == len(episode["action"])
    assert set(("reward", "cost", "value", "cost_value", "lagrange_multiplier", "reward_advantage", "cost_advantage",
                "lagrangian_advantage", "action", "context", "arm", "evaluation_name", "environment_steps")) <= set(episode)
    with pytest.raises(urllib.error.HTTPError) as error:
        _get(url + "/api/episode/99")
    assert error.value.code == 400
    with pytest.raises(urllib.error.HTTPError) as error:
        _get(url + "/static/../server.py")
    assert error.value.code in (400, 404)


def test_server_serves_what_the_policy_saw_as_a_frame_of_its_own_camera(tmp_path):
    """A record with an image part lists it as a policy view, and the frame route serves that
    part's stacked frames as one PNG strip, oldest frame first."""
    frames = np.zeros((5, 4, 4, 6), np.uint8)
    frames[..., 0:3] = 10  # the older frame
    frames[..., 3:6] = 200  # the newest
    _record(images={"pixels/vision": frames}).save(tmp_path)
    server = ReviewServer(("127.0.0.1", 0), Library.from_directories("tiny_group", [tmp_path]))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        url = f"http://127.0.0.1:{server.server_address[1]}"
        assert json.loads(_get(url + "/api/library")[2])["policy_views"] == ["pixels/vision"]
        status, content_type, body = _get(url + "/frame/0/2?camera=pixels/vision")
        assert status == 200 and content_type == "image/png"
        strip = np.asarray(Image.open(io.BytesIO(body)))
        assert strip.shape == (4 * SCALE, 2 * 4 * SCALE, 3)
        assert (strip[:, : 4 * SCALE] == 10).all() and (strip[:, 4 * SCALE :] == 200).all()
    finally:
        server.shutdown()


@RENDERING
def test_server_renders_a_frame_of_a_record_from_its_scene(review_server):
    _, url = review_server
    status, content_type, body = _get(url + "/frame/0/0?camera=fixedfar")
    assert status == 200 and content_type == "image/jpeg" and body[:3] == b"\xff\xd8\xff"
    with pytest.raises(urllib.error.HTTPError) as error:
        _get(url + "/frame/0/9999")
    assert error.value.code == 400
    with pytest.raises(urllib.error.HTTPError) as error:
        _get(url + "/frame/0/0?camera=nonexistent")
    assert error.value.code == 400
