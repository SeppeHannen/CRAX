# The cost tail on `safe_goal_point`: a lidar that reads the hazards' kind and surface removes most of the catastrophic episodes; a memory does not

Done 2026-10-19/20 on Snellius (five arms, 1 h 52 each on an A100). W&B group
`goal_point_cost_tail_500M`, one seed, commits `95d52b7` (four arms) and
`fd8a8ec` (the fifth; the difference is the experiment definition), branch
`cost-tail`. Plan task 1b.

## Question

The per-context learner meets the level-3 budget in the mean, but the mean is
carried by a tail: one episode in a hundred costs 4–40 times the budget, and
every such episode is the agent re-approaching a gap between two solid hazards
it cannot pass (`decisions/2026-10-18_vision_set_aside.md` §1–2). Measured on
384 replayed episodes before this experiment: solid contact is 0.4 % of steps
and 76 % of all cost; the worst episode has 68 touches over 700 steps and a
longest contiguous contact of 10 steps; the median interval between touches is
32 steps. Two things the observation lacks, and the question is which one
decides the episode: the fact that the gap is closed (the lidar reads centre
distances, no kind, no extent), or the fact that the agent has already tried it
(a reactive policy re-decides on every step on which the bump has passed).

## Setup

PPO-Lagrange, `additive told` learner, 8192 environments, 1000-step episodes,
budget 25, 763 rounds, 21 evaluations of 128 episodes on level 3 and on Ω; the
2026-10-15 sensors setup with the context normaliser fix. Four arms on
`uniform` — a 2 × 2 of the two candidates — and, added after they ran, the
surface lidar under `plr` (reward score), so that the lidar is read on the
distribution the best 2026-10-15 policy was trained on.

| arm | change to the observation |
|---|---|
| control | none: the benchmark's hazard lidar, 16 sectors of nearest-centre closeness over all hazards |
| geometry | `hazard_lidar="surface_by_kind"`: two 16-sector arrays, passable and solid, each sector the closeness of the nearest *surface* of that kind over the arc the footprint subtends (62 → 78 sensors) |
| memory | `--memory_time_constant 32`: an exponential trace of (sensors, action, cost) with τ = 32 steps as a third observation part |
| geometry + memory | both |
| PLR geometry | the surface lidar, training distribution `plr` |

```bash
.venv/bin/python -m training.experiments.snellius scripts/experiments/goal_point_cost_tail_500M.py --partition gpu_a100 --time 03:00:00
.venv/bin/python -m training.experiments.snellius scripts/experiments/goal_point_cost_tail_500M.py --partition gpu_a100 --time 03:00:00 --only plr_geometry
# COMMON: --env_name safe_goal_point --alg ppo_lag --deployment_distribution level:3 --constraint additive --policy_context told
#   --initial_lambda_lagr 0.1 --lagrangian_coef_rate 1 --num_envs 8192 --num_timesteps 500_000_000 --num_evals 21
#   --episode_length 1000 --unroll_length 20 --batch_size 1024 --num_minibatches 32 --num_updates_per_batch 4
#   --record_episodes 4 --store_model true --measure_performance --skip_rollout --skip_video
```

Mechanics: 0 recompiles, 92–93 k SPS on every arm (the extra inputs cost
nothing measurable). The evaluator logs, since this experiment, the median
episode cost and the share of episodes over 1× and 4× the budget
(`training/evaluation/cost_tail.py`). Final checkpoints replayed for 2048
episodes per level (`training.replay`, four seeds × 512; records in
`../crax-data/replays_2026-10-19*/`) and read by
`scripts/analysis/goal_point_cost_tail.py` (tail and trap table) and
`scripts/analysis/return_cost_mass_page.py` (the interactive figure). Training
figure: `scripts/analysis/goal_point_cost_tail_figures.py`.

## Results

### Geometry removes the trap's grip and the policy spends the budget; memory does not and slows learning

![level 3 over training](figures/2026-10-19_goal_point_cost_tail_500M/level3.png)

Means of the last five evaluations (400–510 M), level 3:

| arm | return | mean cost | median cost | over 4× budget | Ω return | Ω cost | λ max |
|---|---|---|---|---|---|---|---|
| control | 1.2 | 7.3 | 0.1 | 1.4 % | 13.9 | 22.9 | 15.8 |
| geometry | 13.6 | 24.5 | 21.8 | 0.2 % | 22.9 | 21.5 | 4.0 |
| memory | 5.2 | 20.5 | 11.5 | 2.7 % | 14.0 | 19.9 | 8.1 |
| geometry + memory | 9.1 | 18.8 | 16.9 | 0.5 % | 19.5 | 18.9 | 1.2 |
| PLR geometry | **15.3** | 25.8 | 22.6 | 0.8 % | **23.1** | 20.8 | 3.2 |

- **With the surface lidar the median cost is the mean cost.** On both
  geometry arms the median sits 2–3 under the mean (21.8 / 24.5; 22.6 / 25.8);
  on the control and the memory arm it sits at a tenth or a half of it. The
  cost is spread over the episodes, not carried by a few.
- **The geometry arms never stood still and end with the most return.** Their
  level-3 return is above 4 at every evaluation after 51 M and rises
  monotonically to 13.6 (uniform) and 15.3 (PLR) — the best level-3 policies
  so far (the 2026-10-15 best was 13.8 at a median cost of 13). The control is
  the told arm's known waypoint that this seed never left: motionless from
  230 M, median cost 0.
- **Memory alone leaves the tail** (2.7 % over 4×, the most of the five) and
  costs the first 77 M (0 goals; 65 zero-initialised inputs through the
  running normaliser). Added to the geometry it costs return (9.1 vs 13.6) for
  slightly less cost. The failure was perceptual, not a failure to notice
  unsuccessful movement.

### Replayed to 2048 episodes: a third as many trapped episodes, a worst case six times smaller, and the budget spent where it was hoarded

Replayed from each run's final checkpoint, 4 × 512 episodes per level, the
sampling policy. "Trap" = at least three solid touches on one set of solid
hazards within 100 steps of each other; "escaped" = the episode went on past
the trap without touching that set again.

| policy | level | goals | mean cost | median | p99 | worst | over budget | over 4× | episodes with a trap | touches per trap | traps escaped |
|---|---|---|---|---|---|---|---|---|---|---|---|
| PLR, centre lidar (2026-10-15) | 3 | 13.1 | 19.4 | 13.0 | 102 | 956 | 20 % | 1.1 % | 23 % | 7.7 | 50 % |
| PLR, surface lidar | 3 | **15.5** | 23.7 | 22.3 | 58 | **159** | 40 % | **0.3 %** | **7.8 %** | 6.4 | 61 % |
| uniform, surface lidar | 3 | 13.9 | 24.5 | 23.0 | 66 | 285 | 43 % | 0.5 % | 9.2 % | 6.7 | 62 % |
| PLR, centre lidar | 1 | 27.4 | 19.9 | 18.9 | 43 | 61 | 25 % | 0 | 0 | — | — |
| PLR, surface lidar | 1 | 23.8 | 23.7 | 22.8 | 51 | 73 | 42 % | 0 | 0 | — | — |
| uniform, surface lidar | 1 | 24.5 | 26.4 | 25.4 | 56 | 81 | 51 % | 0 | 0 | — | — |

Interactive: `figures/2026-10-19_goal_point_cost_tail_500M/return_cost_mass_level3.html`
(and `_level1.html`): each policy's episodes on a grid one goal wide and 2.5
cost tall, the share per cell as colour, the ≥ 50 episodes gathered in the top
row with their share of episodes and of all cost, and the budget left for the
other episodes once that row's cost is spread over all of them.

- **The centre-lidar policy's tail is larger than 384 episodes showed**: 1.1 %
  over 100 (was estimated 0.5 %), worst 956 and 705. Its ≥ 50 episodes are
  4.8 % of episodes and **29 % of all cost**; the budget left for the other
  95 % is 19.5. That is the mechanism of 2026-10-16 in numbers: the few set the
  multiplier, the many pay with caution (median 13).
- **Under the surface lidar the ≥ 50 episodes are 2.6 % of episodes and 8 % of
  the cost** (PLR), the worst episode 159, the budget left for the rest 23.2.
  The whole cloud moves up to the budget line and right to more goals.
- **The fix is avoidance, not escape.** Episodes that enter a trap fall from
  23 % to 8–9 %; the share of traps escaped moves from 50 % to 61 % only. The
  remaining catastrophic episodes are the same failure, three times rarer.
- **PLR beats uniform under the surface lidar too** (15.5 vs 13.9 goals on
  level 3 at a smaller tail; Ω 23.1 vs 22.9), as it did under the centre lidar
  on 2026-10-15 — one seed each, so an ordering to retest, not a result.
- **A mean constraint, met by a policy that spends the budget, violates it in
  40–43 % of individual episodes by small amounts**, against 20 % for the
  policy that hoards it. The constraint asks for the mean; the per-episode
  shares now logged say what that buys.

### What was verified on the way: the training-time evaluation and the replay agree, and 128 episodes cannot carry this tail

The 2026-10-15 run's final evaluation logged level-3 cost 22.9 (sd 52) for the
policy whose replay gives 17.6–19.4. Reproducing that evaluation from the
checkpoint with the trainer's own key chain gives 24.8 (sd 89): 127 episodes
with median 14.7 and one episode costing 1014. Fresh keys give 17.4–18.6. The
two stacks are the same stack; the disagreement is one 1-in-100 episode in a
128-episode mean, which moves it by 8. Every 128-episode cost mean reported
for a policy with this tail (2026-10-15, 10-17, 10-18) carries that noise; the
medians and shares do not.

## What this changes in the plan

1. **The surface lidar by kind is the observation for every later arm**, PLR
   with it the best policy so far on both targets. The benchmark's centre lidar
   stays the default of the environment; the variant is opt-in and flagged for
   Tristan (README item 25).
2. **The memory trace stays available and off.** Not the fix here; a recurrent
   policy is the next candidate for the 8 % of episodes that still enter a
   trap, since the lidar made the gap visible but not the history of trying it.
3. **Seeds before ordering.** PLR > uniform held under both lidars with one seed
   each; a second seed of the two surface-lidar arms is the first follow-up.
4. **Task 2 (the held-out hazard band) proceeds on this observation**, whose
   cost is a statement about the policy and not about one episode in a hundred.
5. **Report medians and shares next to every mean**; the dashboard does so
   from this experiment on.
