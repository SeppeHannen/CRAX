# Vision is set aside: it did not remove the failure it was adopted for, and it adds a variance the experiments cannot carry

Decision 2026-10-18. The experiments of this project run on the sensor
observation; vision is future work. This document is the chronology that led
there, one step per section, each stating what was observed and what was
concluded from it. The experiment reports hold the numbers; this holds the
argument.

## 1. The budget is met in the mean while one episode in two hundred costs ten times the budget (2026-10-16)

Replaying the best policy of the state-based learners experiment on level 3
for 384 episodes (`experiments/2026-10-15_goal_point_learners_500M.md`, last
section): the typical episode costs 13–15 against a budget of 25, and the mean
of 17.6 is carried by a tail — 0.5 % of episodes over 100, the worst 247; the
scalar-blind baseline has the same tail ten times as often (6.8 % over 100,
worst 1374). A constraint on the mean is satisfied by a policy whose failures
are few and large, and those few episodes set the multiplier: the policy pays
for them with caution everywhere else.

## 2. Every catastrophic episode is the same failure, and the observation cannot contain the fact that decides it

Watching the records: in every arm the episodes over 100 are solid-hazard
contact while the agent keeps moving — it drives into a gap closed by solid
hazards that stand between it and the goal and keeps trying the same path for
the rest of the episode. The hazard lidar is the closeness of the nearest
hazard centre per 22.5° sector (`safe_goal.py`, `sensor_val_hazard = max(0,
3 − dist)/3`, max over hazards), one array for all 30 hazards: no kind, no
extent, no occlusion. A flat disc the agent may cross and a solid cylinder it
cannot produce the same reading; verified on a recorded episode by swapping
the roles of a blocker and a nearby disc, which leaves the lidar unchanged.
The compasses add direction to the eight nearest centres, no more. So the
decision "is this gap passable" is not a function of the observation, and a
reactive policy over it cannot learn to make it.

## 3. The remedy chosen was a camera, because it carries the missing fact on every suite (2026-10-16)

A fix to the lidar (kind, extent) would be goal-point's alone; a camera shows
kind, extent and occlusion on every suite the benchmark has, and the benchmark
already renders one (`--vision`). The hypothesis: with the wall visible, the
policy learns that a gap between two cylinders is closed and goes round. The
composition of the pixel encoder with the contextual learner's `{sensors,
context}` observation was built and the fifteen arms rerun from pixels + sensors
(`experiments/2026-10-17_goal_point_learners_vision_500M.md`).

## 4. The camera did not change the failure (2026-10-17)

Reviewing the records of the best pixel arms on level 3: the catastrophic
episode of PLR additive told (501 M, episode 3, cost 630) is the failure of
step 2 exactly — the agent in a gap closed by two solid hazards, trying the
same path for the rest of the episode — with both blockers in the policy's own
frames throughout (`training.review`, camera *policy: pixels/vision*). Seeing
the wall did not make the policy go round it. The observation was necessary for
the decision and is not sufficient: the policy also has to represent that it
has tried, or to value the detour, and a feed-forward network over the current
frame does neither.

## 5. The pixel policy is a different policy at every evaluation, and that is the camera's doing, not the training schedule's (2026-10-17 → 18)

Every pixel arm's level-3 return moved 2.5–5 goals between evaluations 25 M
steps apart, against 0.8–1.4 with sensors and a sampling error of 0.2–0.4. Two
candidate causes were tested in turn, each with the same six arms:

| run | observation | round (transitions) | multiplier drift per step | swing between evaluations |
|---|---|---|---|---|
| 2026-10-15 | sensors | 655 360 | 1× | 0.8 – 1.4 |
| 2026-10-17 | pixels | 81 920 | 8× | 2.5 – 5.1 |
| 2026-10-18 (750 M) | pixels | 81 920 | 1× | 1.8 – 4.1 |
| 2026-10-18 (control) | sensors | 81 920 | 1× | 0.4 – 2.1 |

Matching the multiplier's drift left the swing (`experiments/2026-10-18_…`);
running the sensors at the pixel runs' small round gave a policy *smoother*
than at the large round on the four non-staged arms (0.4–0.6 goals; the two
staged arms' 1.8–2.1 is their level-3 switch inside the window). The small
round, which changes every per-round statistic the method computes
(`design/training_round.md`), is exonerated. What remains is the camera: the
shared encoder, its DrQ augmentation redrawn every pass, the 37 000-dimensional
input. Two pixel runs of the same arms differ at the same step by 5–10 goals,
as much as the distributions differ from each other.

## 6. Therefore: sensors for the thesis, vision for future work

The thesis measures what a context distribution does to a constrained
learner. That needs a learner whose evaluation at step *t* means something
about the distribution it was trained under, not about which side of a swing
the evaluation landed on; with one seed per arm the pixel policy cannot give
that, and three seeds per arm from pixels cost 5 h 40 each on an A100 for a
question the thesis does not ask. The reason vision was adopted — step 2's
failure — it did not address (step 4). So:

- **Experiments run on the sensor observation**, as before 2026-10-16. The
  vision composition stays in the code (`training/contexts/networks.py`,
  `--vision --vision_obs_mode pixels+state`) and is documented; nothing is
  removed.
- **Future work**: the same experiments from pixels, once the pixel policy's
  variance is understood (the encoder's learning rate and the augmentation are
  the next two arms, `--vision_augment false` first). Tristan is asked whether
  he saw evaluation-to-evaluation swings of this size in his vision training.
- **The failure of step 2 is the problem to solve next**, on sensors. It is
  the tail that sets the multiplier and makes every other episode over-cautious
  (step 1). Candidates, in the order to try: a policy with memory (an episode
  that has pressed the same wall for 200 steps should know it), an observation
  that carries hazard kind and extent (the lidar as two arrays, one per kind,
  or the nearest-hazard compasses with their sizes), and a cost signal that
  rises with sustained contact so that pressing is distinguished from brushing.
  Each changes the benchmark or the policy class and is flagged for Tristan
  before it runs. Not a cost cap and not a stuckness terminator: both end the
  episodes the measurement is about.

## What the detour produced that stays

The episode recorder and review page with the learner's view and the policy's
frames; whole-learner checkpoints and the replay tool; the Snellius launcher
and the experiment definitions; the per-round critic pass bounded by the
minibatch; `--gpu_memory_fraction`; the one door for a run's files; the
interactive trajectory page; and `design/training_round.md`'s account of what
the round's size governs. The vision composition itself is ready for the future
work.
