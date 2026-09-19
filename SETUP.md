# CRAX Setup Instructions

Getting a working install and your first training run. For the algorithm and
environment catalogue, see the [README](README.md); for pixel observations, see
[docs/VISION_TRAINING.md](docs/VISION_TRAINING.md).

## 1. Create Virtual Environment

Python **3.10+** is required (`requires-python = ">=3.10"`).

```bash
python3 -m venv env
source env/bin/activate  # On Windows: env\Scripts\activate
```

Conda works too:

```bash
conda create -n crax python=3.11 && conda activate crax
```

## 2. Install CRAX

The base install (`pip install -e .`) contains only the environment and
simulation dependencies. Training dependencies live in the `train` extra; see
the extras table in the [README](README.md#installation).

**Option A: training + CUDA 12 (recommended)**
```bash
pip install -e ".[train,cuda]"
```

**Option B: CPU-only training**
```bash
pip install -e ".[train]"
```

**Option B2: environments only (no RL stack)**
```bash
pip install -e .
```

**Option C: pinned versions (for reproducibility)**
```bash
pip install -r requirements-pinned.txt
pip install -e . --no-deps
```

`requirements.txt` holds the same stack with looser bounds and is kept for
environments that install dependencies separately from the package.

> **Do not mix MuJoCo versions.** `mujoco`, `mujoco-mjx` and `mujoco-warp` must
> all be `3.11.0`, paired with `warp-lang==1.16.0`. MJWarp is what the GPU
> vision renderer runs on.

## 3. Verify Installation

```bash
python -c "from crax import envs; print(sorted(envs._envs))"
python -c "from training.agents.ppo_lag import train; print('PPO-Lagrange available!')"
python -c "import jax; print(f'Devices: {jax.devices()}')"
```

The `cuda` extra pulls `jax[cuda12]`, which needs an NVIDIA GPU, CUDA Toolkit
12.x and cuDNN. Without the extra you get CPU JAX: training runs, but far too
slowly for the default `--num_envs 2048`, so drop it to a few dozen envs.
Vision mode (`--vision`) needs a GPU, since MJWarp renders on CUDA.

## 4. First Training Run

```bash
python -m training.train_env \
  --env_name safe_goal_point \
  --alg ppo_lag \
  --difficulty 1 \
  --num_envs 32 \
  --seeds 0
```

`--seeds` takes a list, so `--seeds 0 1 2` runs three sequential experiments.
`--difficulty` accepts `1`, `2` or `3`.

Other entry points:

```bash
# Curriculum training (progressive difficulty)
python -m training.train_curriculum --env_name safe_goal_point --alg ppo_lag

# Safety transfer (pre-train with PPO, then fine-tune with a safe algorithm)
python -m training.train_transfer --env_name safe_velocity_ant --alg ppo_lag

# Pixel observations, rendered on GPU via MJWarp
python -m training.train_env --env_name safe_goal_point --alg ppo_lag --vision
```

All three scripts share the same argument set, defined in
`training/config.py` (`build_base_parser`). Run with `--help` for the
full list, including the per-algorithm sections (`--safety_bound`, `--pid_kp`,
`--nu_lr`, `--tau`, ...).

Boolean flags (`--store_model`, `--normalize_observations`,
`--deterministic_eval`) accept `true/false`, `1/0`, `yes/no` in either case,
and default to `True` when passed bare (`--store_model` == `--store_model true`).
Anything else is a parse error.


## 5. Weights & Biases (wandb)

Every run is tracked in W&B; there is no switch to turn it off. All runs from
all machines go into one project (`training/config.py`: `WANDB_ENTITY` /
`WANDB_PROJECT`, currently `g-g-hannen-eindhoven-university-of-technology/crax`);
use `--wandb_group` and `--wandb_tags` to organise them.

Training aborts before compiling anything if no credentials are found. Get
your API key from https://wandb.ai/authorize and make it available once per
machine, in whichever way suits it:

```bash
# Laptop / workstation / cluster login node: put it in your shell profile
echo 'export WANDB_API_KEY=<key>' >> ~/.bashrc

# Alternative: `wandb login` writes the same key to ~/.netrc

# Batch job on a node without internet: also set
export WANDB_MODE=offline        # then `wandb sync <run dir>` from a node that has
```

```bash
# Group and tag runs
python -m training.train_env --env_name safe_goal_point --alg ppo_lag \
  --wandb_group safe_goal_point \
  --wandb_tags baseline seed-sweep
```

## Troubleshooting

### MuJoCo import errors

If you see `ModuleNotFoundError: No module named 'mujoco.introspect'`:
```bash
pip install --upgrade mujoco mujoco-mjx
```
Keep `mujoco`, `mujoco-mjx` and `mujoco-warp` on the same version (3.11.0).

### GPU out of memory

Reduce the number of parallel environments:
```bash
python -m training.train_env --env_name safe_goal_point --alg ppo_lag --difficulty 1 \
  --num_envs 32 --num_eval_envs 32
```

(Defaults are `--num_envs 2048`, `--num_eval_envs 128`.)

In vision mode, JAX and MJWarp share the GPU. `setup_gpu_environment` caps
`XLA_PYTHON_CLIENT_MEM_FRACTION=0.5` automatically so the renderer has
headroom. Set the variable yourself before launching to override it.

### Rendering issues (headless server)

`setup_gpu_environment` sets `MUJOCO_GL=egl` at startup, which needs a GPU with
EGL. On a machine without one, use software rendering:
```bash
export MUJOCO_GL=osmesa   # also edit training.run_utils.setup_gpu_environment, which
                          # currently overwrites MUJOCO_GL unconditionally
```

## Updating Dependencies

To update to latest compatible versions:
```bash
pip install --upgrade -r requirements.txt
pip freeze > requirements-pinned.txt  # Save new versions
```
