# Staged vs uniform vs fixed level 3 on `safe_goal_point`

Status: done (2026-10-06). Three arms, 100 M steps each, one seed. W&B group
`goal_point_staged_vs_uniform`. Logs: `runs/logs/goal_point_baseline_20261006.log`,
`runs/performance/safe_goal_point_ctx_*_20261006_1{44522,51808,55105}*`.

Plan item 2. The first experiment on the suite Tristan recommended (low
training variance) and the first on a *layout* Ω rather than a cost-threshold
Ω.

## Question

On a context the policy can see (hazard count and layout enter the lidar), does
the manual curriculum staged 1 → 2 → 3 still collapse the way it did on
`safe_velocity_ant`, where the policy could not see the context? And does
training uniformly over Ω reach level 3 as well as training on level 3 alone?

Expected: staged does **not** collapse here. The velocity collapse had two
causes — a context-blind student for whom "easier" meant "a contradictory
task", and a Lagrange multiplier calibrated to level 1's cost regime. The first
cause is absent on goal: the hazards are in the observation, so the student can
learn one policy that behaves differently per layout. The second is still
present; if staged degrades at a switch, it is λ alone, and plan item 3
(PPO-PID) is the matched follow-up.

## Setup

Three runs, identical except for the training distribution. PPO-Lagrange,
`safe_goal_point`, 8192 envs, 1000-step episodes, cost budget 25, one seed.

The environment as of this experiment (Decisions, `docs/acl/README.md`):

- **One goal** (stock: two, of which the compass saw one).
- **The arena walls are a fence**: static geometry, not observed, not costed
  (stock: four `rect` hazards in the lidar, compass and cost).
- **Ω** = four active-hazard counts (discs ≤ 12, solid cylinders ≤ 8, squares
  ≤ 6, solid cubes ≤ 4, **total ≤ 20**) × `goal_size ∈ [0.14, 0.22]`.
  L1 = (12, 0, 0, 0, 0.20), L2 = (8, 8, 0, 0, 0.18), L3 = (6, 4, 6, 4, 0.16).
  The union model has 30 hazard bodies in every level (inactive ones parked).

| arm | `--context_distribution` |
|---|---|
| manual curriculum | `staged:1,2,3` (equal split of the rounds) |
| uniform | `uniform` |
| fixed hardest level | `level:3` |

All three are evaluated on level 3 (`evaluation/deployment/*`) and on uniform
(`evaluation/uniform/*`).

Budget: 100 M steps per arm, the trainer's default (`training/config.py`;
whether this is the paper's budget for goal-point is to be checked with
Tristan). Smoke first (10 rounds, group `smoke_goal_point`, deleted after).

```bash
cd ~/tue/graduation/CRAX
COMMON="--env_name safe_goal_point --alg ppo_lag --seeds 0 \
  --num_envs 8192 --num_timesteps 100_000_000 --num_evals 21 --episode_length 1000 \
  --unroll_length 20 --batch_size 1024 --num_minibatches 32 --num_updates_per_batch 4 \
  --safety_bound 25 --deployment_distribution level:3 \
  --store_model false --skip_rollout --skip_video --measure_performance \
  --wandb_group goal_point_staged_vs_uniform"

.venv/bin/python -m training.train_env $COMMON --context_distribution staged:1,2,3
.venv/bin/python -m training.train_env $COMMON --context_distribution uniform
.venv/bin/python -m training.train_env $COMMON --context_distribution level:3
```

## What to look at

In the group's W&B view (`.venv/bin/python -m training.dashboard --group goal_point_staged_vs_uniform`):

1. **Verdict**: `evaluation/deployment/episode_reward` and `episode_cost` over
   training, three lines, budget line at 25. Does staged dip at rounds ⅓ and ⅔?
   Does uniform match level:3 on level 3?
2. **Mechanism**: `training/lambda_lagr` per arm — the velocity collapse was λ
   1.5 → 405 at the first switch; `training_curriculum/experienced/*` next to
   `sampled/*` — staged's realised curriculum lags the switch by the episode
   length (~1000 steps ≈ 3 rounds).
3. **Trust**: `evaluation/*/avg_episode_length` (goal episodes rarely end
   early; a value far below 1000 means the robot flips), and
   `performance/epoch_steps_per_second` — the first measurement of the union
   model's per-slot cost: 30 hazard bodies, 12 collidable, in every level.

## Results

Three runs, 160 rounds of 655 360 steps each, 33 min per run. Read against the
three reasons for the experiment.

### Does staged collapse when the policy can see its context? — No, but it is never safe

Evaluation on level 3 (budget 25), selected evaluations:

| M steps | staged R / C | uniform R / C | level:3 R / C |
|---|---|---|---|
| 0 | 0.4 / 74 | 0.3 / 77 | 0.3 / 80 |
| 26 | 13.6 / 176 | 13.1 / 79 | 1.5 / 9 |
| 52 | 16.4 / 187 | 15.1 / 87 | 3.6 / 8 |
| 79 | 16.6 / 149 | 17.6 / 104 | 8.9 / 32 |
| 105 (end) | **14.5 / 135** | **16.9 / 138** | **5.1 / 22** |

Stage switches at rounds 51 (≈ 33 M) and 102 (≈ 67 M). Reward never drops at
either switch: no collapse. But the staged arm ends five times over budget with
λ still rising.

The Lagrange multiplier, per arm:

| arm | λ at rounds 0 / 40 / 80 / 120 / 160 | final per-step training cost (budget 0.025) |
|---|---|---|
| staged | 1.8 / 0.0 / 6.8 / 23.8 / **75.2**, still rising | 0.174 |
| uniform | 3.2 / 2.9 / 0.6 / 0.0 / 1.1 | 0.032 |
| level:3 | 4.2 / 4.0 / 2.2 / 6.7 / 5.5 | 0.013 |

Mechanism, from the training-side metrics: on level 1 (12 flat discs, nothing
solid) the student learns to drive *through* hazards, cheaply enough that λ
decays to exactly 0 by round 44. At the switch to level 2 the solid cylinders
make that habit expensive, and λ has to climb from zero at the Lagrange
learning rate: 0.25 by round 60, 12.8 at the end of stage 2. Level 3 raises
the cost again; λ reaches 75 and is still climbing at the end. Training cost
per third of the run: 23 → 50 → 128 per episode.

Compared with the velocity-ant result: there, a context-blind student plus λ
over-wound at the switch froze the policy; here the student sees the layout,
no task is contradictory, and reward survives the switches — **the collapse was
the blind student**. But **the dual variable is still wrong for a curriculum**:
on velocity it carried too much of the previous stage into the next, here it
carries too little (zero). Same root cause, opposite sign: λ is calibrated to
the stage that has just ended. Plan item 3 (PPO-PID, whose proportional term
reacts to the current violation rather than integrating from the past) is the
matched follow-up, with a sharper hypothesis than before.

### Are uniform and level:3 sensible baselines? — Yes, and they split the two objectives

After 100 M steps only `level:3` satisfies the constraint on level 3 (cost 22;
reward 5.1). `uniform` has the highest reward (16.9) at 5.5× the budget, equal
to staged. Level:3's λ oscillates between 0 and 9 around the constraint — the
usual Lagrange cycle — and its reward follows (1.5 → 10 → 5).

On the uniform evaluation the picture is the same: uniform 24.2 / 29, staged
21.0 / 29, level:3 6.2 / 5. Training on the deployment distribution transfers
*down* to easier contexts safely; training on easier contexts does not
transfer *up* safely.

So for this suite: *train on w* is the safe reference, *uniform over Ω* the
reward reference, and the open question for any curriculum method is whether
it can reach uniform's reward at level:3's cost.

### What does the union model cost? — 2.7× stock, the same in every arm

Steady throughput 84.7 k / 84.9 k / 85.1 k SPS (staged / uniform / level:3),
IQR ≤ 4 k, zero substantial compiles in 160 rounds for every arm. Stock
level-1 goal-point at the same batch: 228 k (`measurements/2026-09-19_baseline_goal_point.md`).
The 30-body union model with 12 collidable hazards in every slot's contact
solver costs 2.7×, and the cost is the *model*, not the layout: parked
collidable hazards cost the same as active ones (all three arms equal). At
85 k SPS a 100 M-step run is 33 min including 21 evaluations. For Tristan.

### Trust

Evaluation episode length 1000 ± 0 for every evaluation of every arm: the
robot never flips, so cost differences are not length artefacts — and the
slots stay synchronised (below). The W&B view for the group is the review
surface; `training_curriculum/experienced/*` at rounds 51 and 102 should show
a ~12-round plateau behind `sampled/*` (see the synchronisation finding).

### Two things Giuseppe saw in the W&B view (2026-10-06)

**The 12.5-round sawtooth in `training/*` is the episode boundary, and the
slots are synchronised.** Every loss curve drops sharply at rounds 76, 88,
101, 113, 126, 138, 151, and `episodic/*` is logged only at rounds 13, 25,
38, 50, 63, 75, 88, 100, 113, 125, 138, 150. Verified: the logger flushes
once per round (160 points, 655 360 steps apart; replaying the trainer's call
sequence into `MetricsLogger` gives one flush per round with the current
value), so these are per-round values. The period is the episode: each round
advances every slot by 80 env steps (4 unrolls × 20), episodes are 1000 steps,
and **goal-point episodes never end early** (length 1000 ± 0 in every
evaluation), so all 8192 slots reset at step 0 and stay in lockstep for the
whole run. Round k's batch is one 80-step *slice* of the episode, the same
slice in every slot; `done` happens in all slots at once every 12.5 rounds.

What the sawtooth then means: the cost critic's error grows with position in
the episode (robot in the clear after reset → in among the hazards later), and
in stage 3 that gradient is steep (0.4 → 9 within one episode). The trend
across episodes is real. Consequences:

1. *The PPO batch is not a sample of the episode distribution but a slice of
   it*, and the Lagrange update at round k sees the cost of that slice only.
2. *The stage switches took effect 12 and 11 rounds after they were set.* A
   context is drawn at reset; the switches at rounds 51 and 102 fall mid-episode
   and the new contexts arrive at rounds 63 and 113. The lag
   `intended_vs_realised_curriculum.md` predicts is ~4× what it assumed, because
   it assumed desynchronised slots. The q̂ heatmap should show a plateau at the
   old level for ~12 rounds, not a blur.
3. *The uniform arm redraws all 8192 contexts at once*, every 12.5 rounds.

Stock CRAX behaviour on every suite whose episodes run to the limit (velocity
ant desynchronises because the ant falls). Remedy on the board: a random
initial step offset per slot at the first reset, so slots are spread over the
episode and the batch is a sample. For Tristan.

**`level:3` is at budget on level 3 and far under it on uniform.** Cost 22 of
25 on w (λ oscillating 0–9: the constraint is active and nearly met); cost 5
of 25 on r, with 6 goals where `uniform`'s policy reaches 24. The policy
learned the margin that is safe in the densest layout and keeps it in every
layout. Two readings:

1. *An expected-cost constraint is a property of (policy, distribution), not
   of the policy.* One policy is exactly at budget under w and at a fifth of it
   under r. This is the README's first framing statement, seen from the other
   side: training under w enforces the budget under w, and only there.
2. *A constant margin is what an unobserved budget produces.* The robot sees
   the layout (lidar) but not the budget or its remaining allowance; the safe
   policy is the one for the hardest layout it trained on. So even a
   context-aware policy does not spend budget it could. This is what situation
   4 will probe, and an argument for *told* (plan item 5) — and a question:
   the hazard count *is* in the lidar, so why does the policy not adapt its
   margin to it? Is a 16-bin centre-based pseudo-lidar informative enough about
   density?

Consequence for the target: a curriculum should deliver uniform's reward at
level:3's cost on w — *and* a policy that is at budget under every context it
is deployed in, not under one. That is the per-ω constraint, not the expected
one.

### Caveats

One seed, one algorithm, 100 M steps. Level:3's reward is still rising when
the run ends; a longer run or a second seed would say whether the safe arm's
reward reaches the unsafe arms'. The smoke run's 74 k SPS was measured with
the GPU shared. Slots are synchronised (above), so per-round training metrics
are episode-phase slices; compare rounds 12.5 apart, or read the trend.
