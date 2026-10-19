# Copyright 2024 The Brax Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Brax training acting functions."""

import time
from typing import Callable, Optional, Sequence, Tuple

from crax import envs
from training.evaluation.cost_tail import CostTail
from training.evaluation.recorder import EvaluationRecording
from training.evaluation.scene import SceneWrapper
from training.types import Metrics
from training.types import Policy
from training.types import PolicyParams
from training.types import PRNGKey
from training.types import Transition
import jax
import numpy as np

State = envs.State
Env = envs.Env


def actor_step(
    env: Env,
    env_state: State,
    policy: Policy,
    key: PRNGKey,
    extra_fields: Sequence[str] = (),
) -> Tuple[State, Transition]:
  """Collect data."""
  # named_scope only labels ops in profiler traces; it does not change the program.
  with jax.named_scope('policy_forward'):
    actions, policy_extras = policy(env_state.obs, key)
  with jax.named_scope('environment_step'):
    nstate = env.step(env_state, actions)
  state_extras = {x: nstate.info[x] for x in extra_fields}
  return nstate, Transition(  # pytype: disable=wrong-arg-types  # jax-ndarray
      observation=env_state.obs,
      action=actions,
      reward=nstate.reward,
      discount=1 - nstate.done,
      next_observation=nstate.obs,
      extras={'policy_extras': policy_extras, 'state_extras': state_extras},
  )


def generate_unroll(
    env: Env,
    env_state: State,
    policy: Policy,
    key: PRNGKey,
    unroll_length: int,
    extra_fields: Sequence[str] = (),
) -> Tuple[State, Transition]:
  """Collect trajectories of given unroll_length."""

  @jax.jit
  def f(carry, unused_t):
    state, current_key = carry
    current_key, next_key = jax.random.split(current_key)
    nstate, transition = actor_step(
        env, state, policy, current_key, extra_fields=extra_fields
    )
    return (nstate, next_key), transition

  (final_state, _), data = jax.lax.scan(
      f, (env_state, key), (), length=unroll_length
  )
  return final_state, data


# TODO: Consider moving this to its own file.
class Evaluator:
  """Class to run evaluations."""

  def __init__(
      self,
      eval_env: envs.Env,
      eval_policy_fn: Callable[[PolicyParams], Policy],
      num_eval_envs: int,
      episode_length: int,
      action_repeat: int,
      key: PRNGKey,
      recording: Optional[EvaluationRecording] = None,
      cost_tail: Optional[CostTail] = None,
  ):
    """Init.

    Args:
      eval_env: Batched environment to run evals on.
      eval_policy_fn: Function returning the policy from the policy parameters.
      num_eval_envs: Each env will run 1 episode in parallel for each eval.
      episode_length: Maximum length of an episode.
      action_repeat: Number of physics steps per env step.
      key: RNG key.
      recording: if given, the unroll of the first `recording.num_episodes`
        environments is kept whole, with the `state.info` fields it names
        recorded per step, and returned by `run_recorded_evaluation`
        (docs/acl/design/episode_review.md). The scene the renderer needs is
        written into the state by `SceneWrapper`, so it can be one of them.
      cost_tail: if given, each evaluation also reports the shape of its
        episode-cost distribution (`eval/episode_cost_median`,
        `eval/episode_cost_over_budget`, `eval/episode_cost_over_4x_budget`;
        training/evaluation/cost_tail.py). Needs the environment to report a
        `cost` episode metric.
    """
    self._key = key
    self._cost_tail = cost_tail
    self._eval_walltime = 0.0
    if recording is not None and recording.num_episodes > num_eval_envs:
      raise ValueError(f'cannot record {recording.num_episodes} episodes from {num_eval_envs} evaluation environments')
    num_recorded = recording.num_episodes if recording is not None else 0
    extra_fields = recording.extra_fields if recording is not None else ()

    eval_env = envs.training.EvalWrapper(eval_env)
    if recording is not None:
      # Writes the scene into the state for the record; only an MJX suite has one.
      eval_env = SceneWrapper(eval_env)

    def generate_eval_unroll(
        policy_params: PolicyParams, key: PRNGKey
    ) -> Tuple[State, Transition]:
      reset_keys = jax.random.split(key, num_eval_envs)
      eval_first_state = eval_env.reset(reset_keys)
      final_state, data = generate_unroll(
          eval_env,
          eval_first_state,
          eval_policy_fn(policy_params),
          key,
          unroll_length=episode_length // action_repeat,
          extra_fields=extra_fields,
      )
      # The first `num_recorded` environments' transitions, [T, K, ...]; nothing
      # of the rest leaves the device.
      recorded = jax.tree_util.tree_map(lambda x: x[:, :num_recorded], data)
      return final_state, recorded

    self._generate_eval_unroll = jax.jit(generate_eval_unroll)
    self._steps_per_unroll = episode_length * num_eval_envs

  def run_evaluation(
      self,
      policy_params: PolicyParams,
      training_metrics: Metrics,
      aggregate_episodes: bool = True,
  ) -> Metrics:
    """Run one epoch of evaluation."""
    return self.run_recorded_evaluation(policy_params, training_metrics, aggregate_episodes)[0]

  def run_recorded_evaluation(
      self,
      policy_params: PolicyParams,
      training_metrics: Metrics,
      aggregate_episodes: bool = True,
  ) -> Tuple[Metrics, Transition]:
    """Run one epoch of evaluation; also return the recorded environments' unroll
    (`[T, K, ...]`, `K` = `recording.num_episodes`, 0 without a recording)."""
    self._key, unroll_key = jax.random.split(self._key)

    t = time.time()
    eval_state, recorded = self._generate_eval_unroll(policy_params, unroll_key)
    eval_metrics = eval_state.info['eval_metrics']
    eval_metrics.active_episodes.block_until_ready()
    epoch_eval_time = time.time() - t
    metrics = {}
    for fn in [np.mean, np.std]:
      suffix = '_std' if fn == np.std else ''
      metrics.update({
          f'eval/episode_{name}{suffix}': (
              fn(value) if aggregate_episodes else value
          )
          for name, value in eval_metrics.episode_metrics.items()
      })
    if self._cost_tail is not None:
      if 'cost' not in eval_metrics.episode_metrics:
        raise KeyError("a cost tail needs the environment's `cost` episode metric; this environment reports none")
      metrics.update({
          f'eval/{name}': value
          for name, value in self._cost_tail.metrics(np.asarray(eval_metrics.episode_metrics['cost'])).items()
      })
    metrics['eval/avg_episode_length'] = np.mean(eval_metrics.episode_steps)
    metrics['eval/std_episode_length'] = np.std(eval_metrics.episode_steps)
    metrics['eval/epoch_eval_time'] = epoch_eval_time
    metrics['eval/sps'] = self._steps_per_unroll / epoch_eval_time
    self._eval_walltime = self._eval_walltime + epoch_eval_time
    metrics = {
        'eval/walltime': self._eval_walltime,
        **training_metrics,
        **metrics,
    }

    return metrics, recorded  # pytype: disable=bad-return-type  # jax-ndarray
