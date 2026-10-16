# Copyright 2024 The Brax Authors.
# Modifications Copyright 2026 CRAX Authors.
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
#
# This file is derived from Brax 0.12.3 (training/checkpoint.py) and has been modified
# by the CRAX Authors: relative checkpoint paths are resolved to absolute paths before saving and
# restoring, which orbax requires.

"""Checkpointing functions."""

import inspect
import json
import logging
from typing import Any, Dict, Tuple, Union

import jax
import numpy as np
from etils import epath
from flax.training import orbax_utils
from ml_collections import config_dict
from orbax import checkpoint as ocp

from training import types
from training.acme import running_statistics
from training.agents.ppo import networks as ppo_networks
from training.agents.sac import networks as sac_networks


def _get_function_kwargs(func: Any) -> Dict[str, Any]:
    """Gets kwargs of a function."""
    return {
        p.name: p.default
        for p in inspect.signature(func).parameters.values()
        if p.default is not inspect.Parameter.empty
    }


def _get_function_defaults(func: Any) -> Dict[str, Any]:
    """Gets default kwargs of a function potentially wrapped in partials."""
    kwargs = _get_function_kwargs(func)
    if hasattr(func, 'func'):
        kwargs.update(_get_function_defaults(func.func))
    return kwargs


def network_config(
        observation_size: types.ObservationSize,
        action_size: int,
        normalize_observations: bool,
        network_factory: types.NetworkFactory[
            Union[
                ppo_networks.PPONetworks,
                sac_networks.SACNetworks,
            ]
        ],
) -> config_dict.ConfigDict:
    """Returns a config dict for re-creating a network from a checkpoint."""
    config = config_dict.ConfigDict()
    kwargs = _get_function_kwargs(network_factory)
    defaults = _get_function_defaults(network_factory)

    if 'preprocess_observations_fn' in kwargs:
        if (
                kwargs['preprocess_observations_fn']
                != defaults['preprocess_observations_fn']
        ):
            raise ValueError(
                'checkpointing only supports identity_observation_preprocessor as the'
                ' preprocess_observations_fn'
            )
        del kwargs['preprocess_observations_fn']
    if 'activation' in kwargs:
        # TODO: Add other activations.
        if kwargs['activation'] != defaults['activation']:
            raise ValueError('checkpointing only supports default activation')
        del kwargs['activation']

    config.network_factory_kwargs = kwargs
    config.normalize_observations = normalize_observations
    config.observation_size = observation_size
    config.action_size = action_size
    return config


def get_network(
        config: config_dict.ConfigDict,
        network_factory: types.NetworkFactory[
            Union[
                ppo_networks.PPONetworks,
                sac_networks.SACNetworks,
            ]
        ],
) -> Union[ppo_networks.PPONetworks, sac_networks.SACNetworks]:
    """Generates a network given config."""
    normalize = lambda x, y: x
    if config.normalize_observations:
        normalize = running_statistics.normalize
    network = network_factory(
        config.to_dict()['observation_size'],
        config.action_size,
        preprocess_observations_fn=normalize,
        **config.network_factory_kwargs,
    )
    return network


def save(
        path: Union[str, epath.Path],
        step: int,
        params: Tuple[Any, ...],
        config: config_dict.ConfigDict,
        config_fname: str = 'config.json',
):
    """Saves a checkpoint."""
    # Ensure absolute path (orbax requires absolute paths)
    path = epath.Path(path)
    if not path.is_absolute():
        from pathlib import Path as StdPath
        path = epath.Path(StdPath(path).resolve())
    ckpt_path = path / f'{step:012d}'
    logging.info('saving checkpoint to %s', ckpt_path.as_posix())

    if not ckpt_path.exists():
        ckpt_path.mkdir(parents=True)

    orbax_checkpointer = ocp.PyTreeCheckpointer()
    save_args = orbax_utils.save_args_from_target(params)
    orbax_checkpointer.save(ckpt_path, params, force=True, save_args=save_args)

    config_path = ckpt_path / config_fname
    config_path.write_text(config.to_json())


LEARNER_STATE_DIRECTORY = 'learner'


def save_learner_state(path: Union[str, epath.Path], step: int, learner_state: Any) -> None:
    """Save what the learner knows beyond the policy, next to :func:`save`'s policy tuple.

    The policy tuple (normaliser, policy, value[, encoder]) is what acting needs and what
    every existing reader unpacks positionally. A constrained agent also holds a cost
    critic and its constraint's multipliers; a replay that wants the learner's view of an
    episode (``docs/acl/design/episode_review.md``) needs them too. They go into
    ``<step>/learner`` as their own tree, so the policy tuple keeps its shape and
    :func:`load` keeps its meaning. ``learner_state`` is the agent's own structure
    (PPO-Lagrange: ``{'cost_value': params, 'aux_state': multipliers}``); the reader
    rebuilds that structure and hands it to :func:`load_learner_state`.
    """
    directory = _step_directory(path, step) / LEARNER_STATE_DIRECTORY
    if not directory.exists():
        directory.mkdir(parents=True)
    orbax_checkpointer = ocp.PyTreeCheckpointer()
    orbax_checkpointer.save(directory, learner_state, force=True, save_args=orbax_utils.save_args_from_target(learner_state))


def load_learner_state(path: Union[str, epath.Path], like: Any) -> Any:
    """The tree :func:`save_learner_state` wrote under ``<step>/learner``, restored into the
    structure of ``like`` (the agent's own containers: a multiplier state is a dataclass, an
    optimizer state a NamedTuple; orbax alone would return dicts), every array NumPy. Raises
    if the checkpoint has none (trained before learner states were saved)."""
    directory = _absolute(path) / LEARNER_STATE_DIRECTORY
    if not directory.exists():
        raise ValueError(f'checkpoint {path} has no learner state ({LEARNER_STATE_DIRECTORY}/): it holds the policy only')
    restore_args = jax.tree.map(lambda _: ocp.RestoreArgs(restore_type=np.ndarray), like)
    return ocp.PyTreeCheckpointer().restore(directory, ocp.args.PyTreeRestore(item=like, restore_args=restore_args))


def _step_directory(path: Union[str, epath.Path], step: int) -> epath.Path:
    return _absolute(path) / f'{step:012d}'


def _absolute(path: Union[str, epath.Path]) -> epath.Path:
    """orbax requires absolute paths."""
    path = epath.Path(path)
    if not path.is_absolute():
        from pathlib import Path as StdPath
        path = epath.Path(StdPath(path).resolve())
    return path


def _restore_as_numpy(path: epath.Path) -> Any:
    """Restore every array as NumPy, whatever device it was saved from: a checkpoint trained
    on a GPU (Snellius) must load on a machine without one. The saved tree is
    ``metadata.item_metadata``; the step metadata around it is one leaf, and mapping over
    that (Brax 0.12.3) produced restore args that never matched the tree, so restoring fell
    back to the saved ``cuda:0`` sharding."""
    orbax_checkpointer = ocp.PyTreeCheckpointer()
    saved_tree = orbax_checkpointer.metadata(path).item_metadata
    restore_args = jax.tree.map(lambda _: ocp.RestoreArgs(restore_type=np.ndarray), saved_tree)
    return orbax_checkpointer.restore(path, ocp.args.PyTreeRestore(restore_args=restore_args))


def load(
        path: Union[str, epath.Path],
):
    """Loads checkpoint."""
    # Ensure absolute path (orbax requires absolute paths)
    path = epath.Path(path)
    if not path.is_absolute():
        from pathlib import Path as StdPath
        path = epath.Path(StdPath(path).resolve())
    if not path.exists():
        raise ValueError(f'checkpoint path does not exist: {path.as_posix()}')

    logging.info('restoring from checkpoint %s', path.as_posix())

    target = _restore_as_numpy(path)
    target[0] = running_statistics.RunningStatisticsState(**target[0])

    return target


def load_config(
        config_path: Union[str, epath.Path],
) -> config_dict.ConfigDict:
    """Loads config from config path."""
    # Ensure absolute path (orbax requires absolute paths)
    config_path = epath.Path(config_path)
    if not config_path.is_absolute():
        from pathlib import Path as StdPath
        config_path = epath.Path(StdPath(config_path).resolve())
    if not config_path.exists():
        raise ValueError(f'Config file not found at {config_path.as_posix()}')
    return config_dict.create(**json.loads(config_path.read_text()))
