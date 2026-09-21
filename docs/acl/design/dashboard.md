# A W&B dashboard one can read

Status: steps 1–2 (§1–6) written and discussed 2026-09-21; §7 is the
design for step 3 and is **implemented and in use** (`training/dashboard/`,
tests in `tests/test_dashboard.py`). §7.3 describes the code as built, after
reviewing the first rendered dashboards with Giuseppe (2026-09-21): text
panels restructured, every panel described from the registry, the
environment defined on the page, the group invariant enforced at run start.
First real experiment on it: group `velocity_ant_staged_vs_uniform_750M`.

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
   (it used to carry its own step metric, `performance/environment_steps`, with
   the same values — a second name for one axis; removed in step 3).

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

### `training_curriculum/*` — the round hook (`ContextRoundHook`, then `curriculum_metrics`)

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
| `performance/` | 5 in history (+24 summary-only) | 780 | `training/performance` tracker (at the time via its own W&B sink; now through the progress callback) | 4 trust + own step metric (removed) |
| top level | `environment_steps`, `_step`, `_runtime`, `_timestamp` | 781 | `custom_progress_fn` / W&B | x-axis |

Totals: **174 history keys** (172 for uniform). Concept 16, trust 12,
diagnostic ~20, noise or second copy ~125. The eight primary panels in §5
need 13 of the 174 keys.

## 7. The review experience (step 3, design)

Written after discussing §1–6. Two things were still wrong with §5: it did
not say *how a reviewer learns what a number is* without opening code, and it
did not say *when* a number exists. This section fixes both and is the spec
for the implementation. It is written to hold for every suite, not just Ant.

### 7.1 Three rules

1. **Every panel group starts with a text panel** that states, for the graphs
   below it: the population (what was averaged over), the unit, the cadence
   (at which environment steps a number exists), and what to compare against.
   The reviewer never needs the code.
2. **Groups follow the review questions in order** — Verdict, Mechanism,
   Trust, Detail — not the producing module. Verdict and Mechanism are open;
   Trust and Detail are collapsed.
3. **A point appears only where a number was computed.** No smoothing
   anywhere (`smoothing_type = none`). The text panel states the cadence, so
   the straight segments W&B draws between the 21 evaluation points are not
   mistaken for data. All training-side series share one cadence: one value
   per round (decision 1 in §5 → `episodic/` window = one round).

What W&B cannot do, stated so nobody looks for it: points-only rendering is
per run-ID, not per panel, so lines stay; series longer than ~500 points are
bucketed (min/avg/max per bucket; our 780 rounds → ~1.5 rounds per bucket,
exact when zoomed); a panel has a title and axis labels but no description —
that is what rule 1's text panel is for; there is no constant reference line —
the budget is logged as a metric next to the cost so it is a line on the same
panel.

### 7.2 The view

One saved W&B view per experiment group (`--wandb_group`), generated from the
repo, runset filtered to that group and grouped by `context_distribution`
(mean, min–max band across seeds). The same generator with no group makes the
project's default view. Layout: two columns for the primary sections, three
for the rest. `{…}` are filled from the group's `wandb.config`.

**Verdict** (open, pinned) — text panel:

> Frozen policy, evaluated {num_evals} times over the run, every
> {num_timesteps / (num_evals − 1)} environment steps; {num_eval_envs}
> episodes per evaluation, one point per evaluation, segments between points
> are not data. **Deployment** = `{deployment_distribution}`; **Uniform Ω** =
> contexts drawn uniformly from Ω (`{context_space}`). Return and cost are
> **sums over one episode** (≤ {episode_length} steps), mean over the
> {num_eval_envs} episodes. Reward is the suite's scaled reward. Cost is the
> number of steps in violation; the dashed line is the budget
> {safety_bound} per episode — a policy is safe when cost is under it.
> Lines: one per training distribution, mean over seeds, band = min–max.

| panel | series | y label |
|---|---|---|
| Return per episode on deployment | `evaluation/deployment/episode_reward` | return (scaled reward) |
| Cost per episode on deployment | `evaluation/deployment/episode_cost`, `evaluation/deployment/episode_cost_budget` | violating steps per episode |
| Return per episode on uniform Ω | `evaluation/uniform/episode_reward` | return |
| Cost per episode on uniform Ω | `evaluation/uniform/episode_cost`, `…/episode_cost_budget` | violating steps per episode |

**Mechanism** (open) — text panel:

> What the student trained on and how the constraint reacted. One point per
> **round** (= one PPO training step = {environment_steps_per_round} steps;
> {num_rounds} rounds). Heatmaps: Ω cut into {NUM_BINS} bins, colour = share
> of the round's **transitions** (experienced, q̂) or **completed episodes**
> (sampled, q) in that bin; they differ when episode length depends on the
> context. Mean context: the mean of those same two distributions, the version
> that overlays across arms. λ is PPO-Lagrange's multiplier after the round.
> Training cost per episode: mean over the training episodes that **ended in
> the round**, under whatever contexts the distribution sampled, with
> exploration noise — not comparable to Verdict's cost; the budget line is the
> same {safety_bound}.

| panel | series |
|---|---|
| Contexts experienced (per transition), heatmap over rounds | `training_curriculum/experienced/<dim>` |
| Contexts sampled (per episode), heatmap over rounds | `training_curriculum/sampled/<dim>` |
| Mean context, sampled vs experienced | `…/sampled/<dim>/mean`, `…/experienced/<dim>/mean`, `…/intended/context/<dim>` when present |
| Lagrange multiplier λ | `training/lambda_lagr` |
| Training cost per episode | `episodic/cost`, `episodic/cost_budget` |
| Training return per episode | `episodic/sum_reward` |

**Trust** (collapsed) — text panel:

> Is the run healthy enough to believe the above? Episode length: mean over
> training episodes ended in the round; {episode_length} = full length, a
> collapse to a few steps means the policy falls or freezes. Evaluation
> spread: std over the {num_eval_envs} evaluation episodes — how noisy each
> Verdict point is. Throughput and compiles from the performance tracker;
> more than one compile after the first round means a shape changed.

| panel | series |
|---|---|
| Training episode length | `episodic/length` |
| Evaluation spread on deployment | `evaluation/deployment/episode_reward_std`, `…/episode_cost_std` |
| Evaluation episode length | `evaluation/deployment/avg_episode_length`, `evaluation/uniform/avg_episode_length` |
| Completed episodes per round | `training_curriculum/num_completed_episodes` |
| Steps per second | `performance/epoch_steps_per_second` |
| Compiles per round | `performance/epoch_compiles` |

**Detail** (collapsed) — text panel: "Everything else we keep. Same cadences
as above." Panels: `training/cost_violation` (text: *per-transition* cost in
the PPO batch minus budget/episode_length = {safety_bound/episode_length};
× {episode_length} gives per-episode units), `training/mean_cost`, the five
losses, `episodic/forward_reward`, `episodic/reward_ctrl`,
`episodic/velocity_value`, the suite-specific `evaluation/*/episode_<x>`
that survive the registry, `training_curriculum/*/std`,
`training_curriculum/episode_length/<dim>` (one histogram per round),
`training_curriculum/num_transitions`, `performance/epoch_wall_seconds`.

### 7.3 One source of truth (as built)

`training/dashboard/`, W&B-specific, nothing in `contexts/`:

| file | holds |
|---|---|
| `metrics.py` | the registry. `Metric(pattern, title, unit, group, description, panel, histogram)` for the 39 keys we keep — `description` is the one sentence that says where the numbers come from (population, what was done to it; may use `{evaluation}`, `{dimension}` and the `FACT_PLACEHOLDERS`) — and `Dropped(pattern, reason)` for the 47 we do not; `registered(key)` returns the metric or `None`, and **raises `UnregisteredMetric`** for anything else. `select_for_logging(metrics, safety_bound)` is what the funnel calls; it also adds `<cost>_budget` next to each per-episode cost. |
| `view.py` | `RunFacts` (what the text states: budget, evaluations, episode length, `TaskDescription` — agent, reward, cost in words — and Ω with each dimension's description; read from `wandb.config`, missing key → `KeyError`; JSON round-trip). The section text is **generated**: a preamble (Verdict carries the environment block: agent, reward, cost, Ω, budget; then measurement, evaluation distributions, legend) followed by one bullet per panel built from the registry's descriptions. `plots_for(group, facts)`: metrics sharing a `panel` become one panel; `{evaluation}` expands to *the deployment task (level:3)* and *all of Ω (uniform)*, `{dimension}` to Ω's dimensions. Runs grouped by `training_distribution` so the legend names what a line trained on. |
| `save.py` | **the group invariant.** A named W&B group is one experiment: its runs share the facts. The view is the group's record of them (the `RunFacts` JSON is stored in the view's description). `ensure_view(...)` — called by `train_env.py` for every context run with `--wandb_group`, *before* `wandb.init` — creates the view for a new group and **raises `FactsMismatch`** (listing the differing fields) for a run whose flags differ; `rebuild_view(...)` overwrites deliberately after dashboard code changed. Owns the two GraphQL calls (find by name, upsert) and prints the view URL. |
| `__main__.py` | `python -m training.dashboard --group <name>`: the rebuild path — facts from the group's runs (raises if they disagree), `rebuild_view`. |

Around it:

- `run_utils.wandb_progress_fn(safety_bound, verbose)` is the one funnel; it
  requires an active run (no buffering), passes every key through
  `select_for_logging`, and logs at `step=environment_steps`. Both entry
  points (`train_env.py`, `train_curriculum.py`) use it. The environments and
  the trainer are untouched; the dropped keys never reach W&B.
  `results/common.py` keeps reading `episodic/sum_reward`, `episodic/cost`,
  `episodic/forward_reward` — all kept.
- `contexts/setup.py` passes `training_metrics_steps = steps_per_round` to
  the trainer (decision 1) and puts `num_rounds`,
  `environment_steps_per_round`, `context_space` (bounds and description per
  dimension), `task` and `deployment_distribution` in `wandb.config` for the
  text panels. The words come from the suite: `contexts/registry.py` gives
  each `SuiteContexts` a `TaskDescription` (agent, reward, cost) so the
  dashboard never restates an environment's definition.
- `contexts/training_curriculum.py` (was `realised_curriculum.py`; it is the one
  producer of `training_curriculum/*`, so it is named after the section) logs
  three histograms per dimension
  (sampled, experienced, mean episode length per bin) plus mean/std; the
  `bin_NN` scalars, `completed_episodes/*` and `round` are gone (decisions
  1–2).
- Heatmaps: the workspaces SDK has no histogram panel type, so W&B's
  auto-generated `training_curriculum` section keeps them; the Mechanism text
  says where to look (the fallback foreseen in §7.4).
- Tests (`tests/test_dashboard.py`, CPU): every key of the 500 M runs and of
  a stock `eval/` run is registered; an unknown key raises with instructions;
  the funnel drops the duplicates and adds the budgets; the view has the four
  sections in order, text first, smoothing off, grouped by
  `training_distribution`; every panel and every series is described in its
  section's text with no placeholder left; `RunFacts` round-trips through JSON
  (the invariant's equality).
- Package: `wandb-workspaces` in `requirements.txt` / `pyproject.toml`.

### 7.4 Decisions taken here

- Decision 1: `episodic/` is the one training-episode population, per round.
  `completed_episodes/*` goes.
- Decision 2: `bin_NN` scalars go; heatmaps are per run. Cross-seed q̂ for the
  paper is computed offline.
- Decision 3: no key renames now. Titles and text carry the meaning; renames
  of upstream keys (`cost_violation`, `episode_reward`) go on the list for
  Tristan.
- Decision 4: budgets are logged as `<cost key>_budget` next to the cost, from
  `safety_bound`; the only cost in evaluation units is per episode.
- Heatmap panels: W&B auto-generates them for histogram keys; whether the
  workspaces SDK can place them in our Mechanism section is verified first
  in implementation. If not, they stay in an auto section named
  `training_curriculum` directly under Mechanism, and the text panel says so.
