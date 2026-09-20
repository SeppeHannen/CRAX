# ACL on CRAX — start here

Giuseppe Hannen's graduation project: extend CRAX with automated curriculum
learning (ACL). This file is the **index by task**. Each task lists its status,
the documents that hold the decisions it depends on, and the code it touches.
Read only what the task you are picking up needs.

Last updated: 2026-09-20 (T1–T3 landed in commit `0cc05ed`).

## Vocabulary (used everywhere; do not drift)

| term | meaning |
|---|---|
| context $\omega \in \Omega$ | per-episode task parameters of one suite (e.g. velocity threshold). A **value** on the `num_envs` axis, never a shape. |
| distribution (not "teacher") | object holding $\Phi$ that samples $\omega$ per slot in-program and updates on the host between compiled calls |
| $r$ | `Uniform(Ω)` — the domain-randomisation baseline |
| $w$ | deployment distribution, usually level 3 as a `FixedContext` |
| round $k$ | one compiled call = one PPO training step (`training_round.md`) |
| $q_{m,k}$ / $\hat q$ | intended (per episode start) vs realised (per transition) curriculum — always report both |
| slot | one of the `num_envs` parallel environments |

## Task board

### Done
| # | task | where |
|---|---|---|
| D1 | W&B mandatory, single project `crax`, `--use_wandb` removed | `training/config.py`, `training/run_utils.py` |
| D2 | Performance tracker + XProf traces → W&B | `training/performance/`, `performance_measurement.md` |
| D3 | Baseline measurement + GPU trace: launch-bound, physics 83%, task code <1% | `measurements/2026-09-19_*`, `measurements/2026-09-20_gpu_trace_walkthrough.md` |
| D4 | `num_envs` sweep (Point): knee at 8192 → **use 8192** | `measurements/2026-09-20_num_envs_sweep_and_launch_bound.md`, `scripts/sweep_num_envs.py` |
| D5 | Design: per-slot rule, round definition, $q$ vs $\hat q$ | `design/per_slot_constraints.md`, `design/training_round.md`, `design/intended_vs_realised_curriculum.md` |
| D6 | `training/contexts` package, SafeVelocity reads context, 8 CPU tests | `design/contexts_package.md`, `training/contexts/`, `tests/test_contexts.py` |
| D7 | Ω audit for all 9 suites (value / count / structural) | `design/context_spaces_by_suite.md` |
| D8 (=T1) | Trainer wiring: generic `RoundHook` (epoch := 1 round), `ContextRoundHook` updates φ between calls, $q$ and $\hat q$ per round to W&B | `design/contexts_package.md` § how it is wired, `training/rounds.py`, `training/contexts/{round_data,realised_curriculum,round_hook}.py` |
| D9 (=T2) | Eval on $w$ and $r$: `evaluation_wrap_env_fns` → `eval/deployment/*`, `eval/uniform/*` | `training/agents/ppo/train.py`, `training/contexts/setup.py` |
| D10 (=T3) | CLI `--context_distribution none\|uniform\|level:<n>\|staged:<n>,..` and `--deployment_distribution` (same grammar); `train_env.py` wired; 9 CPU tests + GPU smoke run | `training/config.py`, `training/train_env.py`, `tests/test_context_training.py` |

### Next — in order
| # | task | status | read first | touches |
|---|---|---|---|---|
| **T5** | **First experiment** on `safe_velocity_ant`, 4 arms: direct L3 / staged L1→L2→L3 / uniform $r$ / direct L1; eval on $w$ and $r$; report $q$, $\hat q$ | **ready — GPU, ask first**; exact commands in the doc | `experiments/2026-09-20_uniform_vs_staged_velocity_ant.md`, `design/intended_vs_realised_curriculum.md` | none (CLI only); results section of the experiment doc |
| **T4** | **Reset-cost measurement** (GPU — ask first): `--context_distribution level --difficulty 3` vs `none` with `--measure_performance`, `safe_velocity_ant` @8192; falls out of the T5 pilot | not started | `performance_measurement.md`, `design/per_slot_constraints.md` (§ AutoReset discovery) | `training/contexts/wrapper.py` if it turns out expensive |
| T4b | Forward `round_hook` / `evaluation_wrap_env_fns` in the other PPO-family trainers (focops, p3o, crpo, ppo_pid, ppo_saute, ppo_cost); route `train_curriculum.py` through `staged` for registered suites | not started | `design/contexts_package.md` § not yet done | `training/agents/*/train.py`, `training/train_curriculum.py` |
| T6 | Extend value-typed suites: height, push (step-side reads), lift (feet mask), pathway (`reset_with_context`) | not started | `design/context_spaces_by_suite.md` | `crax/envs/safe_{height,push,lift,pathway}.py`, `training/contexts/registry.py` |
| T7 | Pad-and-mask suites: reach, circle | not started | `design/context_spaces_by_suite.md`, `design/per_slot_constraints.md` | env XML builders + lidar/cost code |
| T8 | Union-model suites: goal, button (tell Tristan first) | not started | same as T7 | same as T7 |
| T9 | GPU side quests (when GPU free): Ant/Humanoid sweep; CUDA-graph flag `--xla_gpu_enable_command_buffer=...` at Point@8192; trace 32768 vs 8192 | not started | `measurements/2026-09-20_num_envs_sweep_and_launch_bound.md` (exact commands) | `scripts/sweep_num_envs.py` |
| T10 | Actual ACL methods (after uniform-vs-manual baseline exists) | not started | thesis preliminaries; `distribution.py` protocol | `training/contexts/distributions/` (one file per method) |

## Standing rules
- **No GPU runs without asking.** CPU tests: `JAX_PLATFORMS=cpu .venv/bin/python -m pytest tests/test_contexts.py -q -p no:cacheprovider`.
- W&B is mandatory; `WANDB_API_KEY` lives in `~/.bashrc` (below the interactive guard — sbatch must export it).
- Values not shapes. A new compiled program costs ~50 s; a per-slot value costs nothing.
- One file per distribution; clear types, protocols, encapsulation; no abbreviations in code.
- `epoch` in the trainer means one training step (= one round) whenever a `round_hook` is set, i.e. for every `--context_distribution` other than `none`.
- Anything that changes Tristan's baselines (AutoReset replay, union models) gets flagged to him.

## Document map
```
docs/acl/
  README.md                        ← this file (task index)
  profiling_options.md             tool survey behind D2 (historical)
  performance_measurement.md       how to use training/performance (user guide)
  measurements/                    dated logs; newest wins
  experiments/                     one file per experiment: arms, commands, results
    2026-09-20_uniform_vs_staged_velocity_ant.md   T5 (ready to launch)
  design/
    per_slot_constraints.md        what may differ between slots; AutoReset discovery
    training_round.md              epoch := 1 training step
    intended_vs_realised_curriculum.md   q vs q̂
    contexts_package.md            training/contexts handoff, trainer wiring, logged keys, CLI
    context_spaces_by_suite.md     Ω per suite, value/count/structural
```
Memory files for the assistant mirror this at `/memories/repo/project.md` and
`/memories/repo/profiling.md`; the README is the source of truth when they
disagree.
