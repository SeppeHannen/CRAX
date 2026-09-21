# A W&B dashboard one can read

Status: step 1 (first principles, §1–5) and step 2 (inventory, §6) written
2026-09-21, for discussion. Nothing implemented yet; step 3 (how) waits for
the discussion.

## 1. What a run is for

Every run is one arm of a comparison between training distributions, judged by
how well the resulting policy does on the deployment task $w$ under the cost
budget $d$. Anything on the dashboard must serve one of four purposes, in this
order of importance:

| purpose | question it answers | what it is made of |
|---|---|---|
| **verdict** | Did this arm produce a policy that is good *and* safe on $w$, and when did it get there? | frozen-policy evaluation on the deployment distribution: return and cost, cost against the budget |
| **mechanism** | What did the student train on over time, and how did the constraint machinery react? | sampled vs experienced contexts per round; λ; the per-episode cost the student was being pushed by |
| **trust** | Is the run healthy enough that verdict and mechanism can be believed? | throughput, compiles, training episode length, evaluation noise (std over eval episodes), number of completed episodes per round |
| **detail** | Everything else, wanted only when the above raise a question. | reward components, losses, per-bin masses, walltime |

A panel that serves none of these should not be logged.

## 2. Review situations

| situation | who / when | default view | what must be true of it |
|---|---|---|---|
| **A. one run, "is it behaving"** | during training, glance every hour | run page, primary section on top | verdict + trust visible without scrolling; a freeze (episode length → 10, λ → 400) is obvious at a glance |
| **B. one run, "what happened"** | after training, writing the experiment note | run page, all sections | mechanism panels line up with verdict panels on the same x-axis |
| **C. a group, "which arm won and why"** | the real question, after each experiment | a group workspace / report: arms overlaid on the same axes | same x (environment steps), cost panel with the budget line, seeds aggregated (mean ± range), one line per arm not per run |
| **D. across experiments, "did this change help"** | paper-style, in `results/` | offline figures from `results/` | needs the same keys in every run of the project → the registry below is shared with `results/common.py` |

The run page is not the group page. Today we only have run pages and read C
off two run pages side by side; that is what makes the comparison slow.

## 3. What a panel must say for itself

A reviewer should be able to read any panel without opening the code:

1. **Population** — over what was this averaged? Three populations appear today
   under near-identical names:
   - training episodes that *ended in the logging window* (`episodic/*`, `MetricsLogger`, window = `training_metrics_steps` = 1 M steps, which with 655 360-step rounds means **every second round**: 390 rows against 780 rounds in the 500 M runs);
   - the 128 frozen-policy episodes of one evaluation pass (`evaluation/<name>/*`, `Evaluator`, every `num_timesteps / num_evals` steps);
   - transitions (`training/mean_cost`, and the whole `training_curriculum/experienced/*`).
2. **Unit** — per episode (sum over ≤ 1000 steps) or per step; Ant's reward is
   scaled by `reward_scaler = 0.01` so "reward 20" is 2000 unscaled.
3. **Reference** — the budget $d = 25$ per episode on every cost-per-episode
   panel; $d/T = 0.025$ on per-step cost panels; the level thresholds on
   context panels.
4. **x-axis** — environment steps, everywhere, including `performance/`
   (today `performance/*` has its own step metric, `performance/environment_steps`, with the same values; harmless but a second name for the same axis).

The core problem is (1). `episode_reward` under `evaluation/deployment/`,
`evaluation/uniform/` and `episodic/sum_reward` are three different
populations of the same quantity; the panel title says which section only if
you already know what the sections mean. Section names and panel titles
should state the population.

## 4. What our own confusion says

Each key we did not understand, with what it is, where it comes from, and the
verdict: **concept** (a reviewer needs it), **diagnostic** (keep, secondary),
**noise** (stop logging).

### `episodic/*` — training episodes ended in the window (`EpisodeWrapper` sums → `MetricsLogger`)

| key | what it is | verdict |
|---|---|---|
| `episodic/sum_reward` | sum of `state.reward` over the episode = scaled Ant reward | **concept**: training return |
| `episodic/cost` | sum of the binary velocity cost over the episode = number of steps over the threshold | **concept**: training cost per episode; the quantity λ is pushing on |
| `episodic/length` | steps per episode | **concept** (trust): the freeze shows up here first |
| `episodic/forward_reward`, `episodic/reward_forward`, `episodic/x_velocity` | *the same number three times* (verified identical). Ant logs `reward_forward` (Brax's name), `forward_reward` (added for `results/common.py`, which reads `episodic/forward_reward` for velocity suites) and `x_velocity`; all three are the forward velocity, summed over the episode = distance travelled / dt | **concept** for results/ (one of them, `forward_reward` since `results/` reads it); the other two are **noise** |
| `episodic/reward_survive` | 1 per step alive → identical to `length` (`terminate_when_unhealthy`) | **noise** (duplicate of length) |
| `episodic/reward_ctrl`, `episodic/reward_contact` | −0.5 Σa²; contact is identically 0 (contact forces disabled; verified) | ctrl: **diagnostic**; contact: **noise** |
| `episodic/reward_unscaled` | `velocity_constraints.add_velocity_cost_metrics` stores `reward` *after* `Ant.step` but *before* `× reward_scaler`; `reward_unscaled = 100 × sum_reward` to 5 digits | **noise** (a constant factor) |
| `episodic/velocity_cost` | identical to `cost` (cost weight = 1) | **noise** |
| `episodic/velocity_violation` | identical to `cost` in binary mode | **noise** |
| `episodic/velocity_value`, `episodic/velocity_magnitude` | the same number twice (verified): sum over the episode of the constrained speed | **diagnostic**: keep one, as *mean* speed would be more useful than the sum; today `mean = value / length` |
| `episodic/velocity_threshold` | sum over the episode of the threshold = threshold × length (verified: ratio 1.835 = mean of Ω for uniform) | **noise** as a sum; the context is already reported in `training_curriculum` |
| `episodic/x_position`, `y_position`, `distance_from_origin`, `y_velocity` | sums of per-step positions — meaningless as sums | **noise** |

One more thing about this population, verified against the data: the
`jax.debug.callback` hands `MetricsLogger` one training step (= one round) at
a time, so each buffered value is the mean over *all* episodes completed in
that round, and the logged value is the equal-weight mean of the rounds in the
window. `episodic/cost` equals the mean of the two rounds'
`training_curriculum/completed_episodes/mean_cost` to 1e-7. So the two
training-episode populations are the same quantity at two resolutions (every
round vs every second round); there is no reason to keep both (decision 1).

### `training/*` — optimiser and λ (PPO loss metrics, `post_step_fn`, `training_epoch_with_timing`)

| key | what it is | verdict |
|---|---|---|
| `training/lambda_lagr` | the Lagrange multiplier after the round | **concept** (mechanism) |
| `training/cost_violation` | mean per-transition cost in the batch − $d/T$; the signal that moves λ | **concept**: it is "training cost per step minus budget per step"; show with a zero line |
| `training/mean_cost` | mean per-transition cost in the batch | **diagnostic** (the same as `cost_violation + 0.025`) |
| `training/total_loss`, `policy_loss`, `v_loss`, `cost_v_loss`, `entropy_loss` | PPO losses | **diagnostic** |
| `training/sps`, `training/walltime` | throughput (trainer's own) | `sps` duplicates `performance/epoch_steps_per_second` (ratio 0.9998); **noise** when `--measure_performance` is on; keep the trainer's as fallback |
| `training/final_step` | only logged when there is no evaluation | **noise** |

### `evaluation/<name>/*` — 128 frozen-policy episodes (`acting.Evaluator` via `EvalWrapper`, `_std` = std over the 128)

Every `state.metrics` key gets `episode_<key>` and `episode_<key>_std`, plus `avg_episode_length`, `std_episode_length`, `epoch_eval_time`, `sps`, `walltime`. That is 2 × 19 + 5 = 43 keys per evaluation distribution, 86 for two.

| key | verdict |
|---|---|
| `episode_reward` (sum of `state.reward`, scaled) | **concept**: the verdict's return |
| `episode_cost` | **concept**: the verdict's cost, against budget 25 |
| `episode_reward_std`, `episode_cost_std` | **trust**: spread over the 128 episodes; show as band, not separate panels |
| `avg_episode_length`, `std_episode_length` | **trust** (does the policy survive?) |
| `episode_forward_reward` / `episode_reward_forward` | **diagnostic**, one of them |
| `episode_velocity_value` (÷ length = mean speed) | **diagnostic**: the one number that explains cost on every level at once, given the thresholds |
| `episode_reward_unscaled`, `velocity_cost`, `velocity_violation`, `velocity_magnitude`, `velocity_threshold`, `reward_survive`, `reward_contact`, `x/y_position`, `distance_from_origin`, `x/y_velocity` and all their `_std` | **noise**, same reasons as under `episodic/` |
| `epoch_eval_time`, `sps`, `walltime` | **noise** on the dashboard (evaluation walltime is not a research quantity); keep one in `performance/` if ever needed |

The population mismatch is real: `evaluation/deployment/episode_cost` is
*"cost per episode, 128 frozen-policy episodes, level 3"*, `episodic/cost` is
*"cost per episode, training policy with exploration noise, whatever contexts
the distribution sampled, episodes ended in the last 1 M steps"*. Both are
concepts, both are needed, and the name must say which is which.

### `training_curriculum/*` — the round hook (`ContextRoundHook`, `curriculum_metrics`)

| key | verdict |
|---|---|
| `sampled/velocity_threshold`, `experienced/velocity_threshold` (histograms → heatmap over rounds) | **concept**: *the* mechanism panel; q̂ next to q |
| `sampled/…/mean`, `experienced/…/mean` | **concept** for the group view: one line per arm; the gap between them is the exposure-vs-selection lag |
| `sampled/…/std`, `experienced/…/std` | **diagnostic** |
| `sampled/…/bin_00..11`, `experienced/…/bin_00..11` (24 scalars) | logged so masses are comparable across runs/seeds (the heatmap is per run and cannot be averaged). Nobody has looked at a `bin_NN` panel. **Diagnostic at best; candidate to drop** unless we plan to aggregate heatmaps across seeds — for D (paper figures) we would recompute from artifacts anyway. Decision needed. Note the histogram itself is stored as `…/bins` and `…/values` (nested keys, visible in the inventory), so the 12 bin masses are *already* in the history once; the scalars are a second copy. |
| `episode_length/velocity_threshold/bin_00..11` (12 scalars) | the explanation for sampled ≠ experienced. Same objection: 12 lines nobody reads. Better logged as *one* histogram per round (mean episode length per bin → heatmap). **Replace 12 scalars by 1 histogram.** |
| `intended/stage`, `intended/context/velocity_threshold` | **concept** for staged; makes the switch rounds visible |
| `completed_episodes/mean_return`, `mean_cost`, `mean_length` | duplicates `episodic/sum_reward`, `episodic/cost`, `episodic/length` on a per-round instead of per-1 M-step window. **Noise** — or the replacement: they are per round (the natural unit) whereas `episodic/` is per arbitrary window. Keep one population. |
| `num_transitions`, `num_completed_episodes` | **trust**: completed episodes per round is the sample size behind `sampled/*` |
| `round` | **noise** (rounds ↔ steps is a constant) |

### `performance/*` (`training/performance`)

In the history: `epoch_steps_per_second`, `epoch_wall_seconds`,
`epoch_compiles`, `epoch_compile_seconds` (+ its own `environment_steps`):
**trust**, all four, in one small panel group. The other 24 `performance/*`
keys are run-summary columns (steady-state medians, program sizes, git sha),
not time series, and do not clutter the run page. Fine as is.

## 5. The dashboard

### Primary panels (the top of every run page, and the whole of the group page)

x-axis is environment steps on all of them. "Deployment" = the
`--deployment_distribution` (level 3 today); its name appears in the title.

| # | title | population | unit | reference | serves |
|---|---|---|---|---|---|
| 1 | **Deployment return** — frozen policy, 128 episodes on level 3, mean ± std | evaluation/deployment | scaled reward per episode | — | verdict |
| 2 | **Deployment cost** — frozen policy, 128 episodes on level 3, mean ± std | evaluation/deployment | cost steps per episode | budget 25 | verdict |
| 3 | **Uniform-Ω return and cost** — frozen policy, 128 episodes, ω ~ Uniform(Ω) | evaluation/uniform | as 1–2 | budget 25 | verdict (generalisation), one panel with two lines or two small panels |
| 4 | **Contexts trained on** — experienced (per transition) heatmap over rounds, with sampled (per episode) beside it | training_curriculum | fraction of transitions per bin of Ω | level thresholds 2.62 / 1.97 / 1.31 as horizontal lines | mechanism |
| 5 | **Mean context, sampled vs experienced** — two lines | training_curriculum | velocity threshold | level thresholds | mechanism (this is the version that overlays across arms and seeds; #4 does not) |
| 6 | **Lagrange multiplier λ** | training | — | — | mechanism |
| 7 | **Training cost per episode** — training policy, episodes ended in the round, all contexts | episodic (or completed_episodes) | cost steps per episode | budget 25 | mechanism: what λ reacts to |
| 8 | **Training episode length** and **steps per second** | episodic + performance | steps; steps/s | 1000 (max length) | trust |

Eight panels, four rows of two. Situation A reads rows 1–2 and panel 8;
situation B all eight; situation C the same eight with arms overlaid and seeds
aggregated (heatmaps: one column per arm).

### Secondary groups (collapsed by default)

- `evaluation/*`: `episode_forward_reward`, `episode_velocity_value`, `avg_episode_length`, per distribution.
- `training/*`: `cost_violation` (zero line), `mean_cost`, the five losses.
- `training_curriculum/*`: `intended/*`, `num_completed_episodes`, `…/std`, the episode-length-per-bin heatmap.
- `episodic/*`: `sum_reward`, `forward_reward`, `reward_ctrl`, `velocity_value`.
- `performance/*`: as today.

### Drop (stop logging or stop forwarding to W&B)

From the environment side (flag to Tristan — it changes what every upstream run logs):
`reward_unscaled`, `velocity_cost`, `velocity_violation`, `velocity_magnitude`,
`velocity_threshold` as episode sums; `reward_contact` (always 0),
`reward_survive` (= length); the per-step position/velocity metrics as episode
sums (`x_position`, `y_position`, `distance_from_origin`, `x_velocity`,
`y_velocity`). Alternative that does not touch upstream: filter at the
`custom_progress_fn` funnel with the registry — the environment keeps logging,
W&B stops receiving. The second is the one I would do first.

From our side: all `bin_NN` scalars (36 keys) → one histogram for episode
length per bin, none for masses unless we need cross-seed aggregation;
`completed_episodes/*` vs `episodic/*` → keep one population; `round`;
`eval/*_std` as separate panels → bands on the mean panels.

Expected key count after this: ~15 primary + ~25 secondary + performance,
down from ~200, and each has a title that states population, unit and
reference.

### Open decisions (for the discussion)

1. **One population for training-episode metrics.** `episodic/*` (Brax's,
   every second round because `training_metrics_steps` = 1 M) or
   `training_curriculum/completed_episodes/*` (ours, every round) are the same
   quantity at two resolutions. Proposal: keep `episodic/*` — `results/common.py`
   and every stock CRAX run read it — set `training_metrics_steps` to the
   round size (`num_envs × unroll_length × num_minibatches` × ... = 655 360
   here; can be derived, not typed) so it is per round, and drop
   `completed_episodes/*`. Zero new code, one fewer population.
2. **Do we ever aggregate heatmaps across seeds?** If yes, the `bin_NN` scalars
   stay (as the only aggregable form); if no, they go. My guess: for the paper
   we compute q̂ figures offline from the recorded rollouts, so they go.
3. **Renaming `evaluation/<name>/episode_reward` → `…/return`** and the
   `episodic` section to something that says "training episodes"? Renames break
   the two 500 M runs' comparability with future runs unless `results/` maps old
   names. Titles and descriptions in the workspace can fix the reading problem
   without renaming keys; that is the cheaper route.
4. **Where the budget line comes from.** It is `--safety_bound` in `wandb.config`;
   the workspace definition can read it per run, but a group panel needs one
   value — fine as long as one experiment has one budget.

## 6. Inventory of the two 500 M runs (step 2)

Group `velocity_ant_staged_vs_uniform_500M`, runs
`safe_velocity_ant_ctx_staged123_ppo_lag_seed0_20260921_000322_427485` and
`safe_velocity_ant_ctx_uniform_ppo_lag_seed0_20260921_003144_709487`, both
`finished`, 781 history rows each. Read from the local `.wandb` record files
(`wandb/run-*/run-*.wandb`, via `wandb.sdk.internal.datastore`): the online
`scan_history()` hung twice on the first page; the local file is the same
history and takes seconds. Scripts: `/tmp/crax_local_inventory.py`,
`/tmp/crax_check_duplicates.py` (to be moved into the repo in step 3 if we
keep them).

Rows are the same in both runs except `intended/*` (staged only). Rows per
key tell the population: 780 = once per round, 390 = every second round
(`episodic/`), 21 = per evaluation pass (`num_evals`).

| section | keys | rows | producer | classification |
|---|---|---|---|---|
| `episodic/` | 19 | 390 | `EpisodeWrapper` sums → `MetricsLogger.update_env_metrics` → `custom_progress_fn` | 4 concept (`sum_reward`, `cost`, `length`, `forward_reward`), 2 diagnostic (`reward_ctrl`, `velocity_value`), **13 noise** — 8 verified exact duplicates (`velocity_cost`, `velocity_violation` = `cost`; `reward_survive` = `length`; `velocity_magnitude` = `velocity_value`; `reward_forward`, `x_velocity` = `forward_reward`; `reward_unscaled` = 100 × `sum_reward`; `reward_contact` ≡ 0) and 5 meaningless sums (`velocity_threshold`, `x_position`, `y_position`, `distance_from_origin`, `y_velocity`) |
| `training/` | 10 | 780 | PPO-Lagrange loss dict + `post_step_fn` + `training_epoch_with_timing` → `progress_fn` | 2 concept (`lambda_lagr`, `cost_violation`), 6 diagnostic (`mean_cost`, five losses), 2 noise (`sps` = `performance/epoch_steps_per_second`, `walltime`) |
| `training_curriculum/` | 54 (50 + 2 × `bins`/`values` of the histograms) | 780 | `ContextRoundHook.on_round_end` → `curriculum_metrics` | 6 concept (2 histograms, 2 `mean`, `intended/stage`, `intended/context/velocity_threshold`), 4 trust/diagnostic (`num_completed_episodes`, `num_transitions`, 2 `std`), 3 duplicate-population (`completed_episodes/*`, see decision 1), **37 to fold**: 24 `bin_NN` masses (second copy of the histograms), 12 `episode_length/bin_NN` (→ one histogram), `round` |
| `evaluation/deployment/` | 41 | 21 | `acting.Evaluator` over `EvalWrapper` sums, renamed by `_evaluation_metric_name` | 2 concept (`episode_reward`, `episode_cost`), 4 trust (`episode_reward_std`, `episode_cost_std`, `avg_episode_length`, `std_episode_length`), 4 diagnostic (`episode_forward_reward`, `episode_velocity_value` and their `_std`), **31 noise** (the same duplicates as `episodic/` × 2 for `_std`, plus `epoch_eval_time`, `sps`, `walltime`; `episode_reward_survive` = `avg_episode_length` verified) |
| `evaluation/uniform/` | 41 | 21 | same | same; verified that `episode_reward` on uniform equals deployment to within eval noise (ratio 1.00, spread 0.09) — the context-blind policy, as in the experiment note |
| `performance/` | 5 in history (+24 summary-only) | 780 | `training/performance/sinks.WandbSink` | 4 trust + own step metric |
| top level | `environment_steps`, `_step`, `_runtime`, `_timestamp` | 781 | `custom_progress_fn` / W&B | x-axis |

Totals: **174 history keys** (172 for uniform). Concept 16, trust 12,
diagnostic ~20, noise or second copy ~125. The eight primary panels in §5
need 13 of the 174 keys.
