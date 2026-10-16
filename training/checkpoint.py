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

import wandb

from training import types
from training.acme import running_statistics
from training.agents.ppo import networks as ppo_networks
from training.agents.sac import networks as sac_networks

RUN_CHECKPOINTS_DIRECTORY = 'checkpoints'


def run_checkpoint_directory() -> epath.Path:
    """Where the active W&B run saves its checkpoints.

    Inside the run's directory: :func:`save` uploads every checkpoint it writes
    there with the run's other files, so any machine can fetch a run's policy
    through the W&B API — the one door for every artefact of a run, wherever it
    trained. Requires an initialised run.
    """
    if wandb.run is None:
        raise RuntimeError('checkpoints are saved under the W&B run directory; initialise the run first')
    path = epath.Path(wandb.run.dir) / RUN_CHECKPOINTS_DIRECTORY
    path.mkdir(exist_ok=True)
    return path


def _upload_with_run(checkpoint_path: epath.Path) -> None:
    """Hand a written checkpoint to the active W&B run when it lies in the run's directory.

    W&B uploads only the files it is told about, after they exist (a glob registered
    before the files appear uploads nothing; verified 2026-10-15), so this is called
    once per written checkpoint. A checkpoint written elsewhere — the transfer
    benchmark's working directory, which its own safe phases restore from — stays local.
    """
    if wandb.run is None:
        return
    run_directory = epath.Path(wandb.run.dir).resolve()
    if run_directory not in checkpoint_path.resolve().parents:
        return
    wandb.run.save(str(checkpoint_path / '**'), base_path=str(run_directory), policy='now')


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
    _upload_with_run(ckpt_path)


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

    # Restore every array as NumPy, whatever device it was saved from: a checkpoint
    # trained on a GPU (Snellius) must load on a machine without one. The saved
    # tree is `metadata.item_metadata`; the step metadata around it is one leaf, and
    # mapping over that (Brax 0.12.3) produced restore args that never matched the
    # tree, so restoring fell back to the saved `cuda:0` sharding.
    orbax_checkpointer = ocp.PyTreeCheckpointer()
    saved_tree = orbax_checkpointer.metadata(path).item_metadata
    restore_args = jax.tree.map(lambda _: ocp.RestoreArgs(restore_type=np.ndarray), saved_tree)
    target = orbax_checkpointer.restore(path, ocp.args.PyTreeRestore(restore_args=restore_args))
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
