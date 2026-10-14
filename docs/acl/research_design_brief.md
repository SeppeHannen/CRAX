# A curriculum changes the training distribution while it trains. What does that change do to a constrained policy's generalisation?

2026-10-14, for the supervisor meeting. Full design:
`research_design.md`.

A curriculum changes the training distribution during training. That change
has two measurable properties. Its **coverage** is which contexts the
distribution puts mass on, and how much, by the end. Its **non-stationarity**
is how much the distribution changed from one round to the next on the way
there. 

We measure what each property does to generalisation on contexts the policy
never trained on, in return and in constraint satisfaction. Kirk et al.'s
survey of zero-shot generalisation in RL (JAIR 2023) names non-stationary
training data and multi-objective learning as its open questions; a
constrained curriculum is where the two meet. The core experiment trains, next
to each curriculum, two twins that keep its coverage and remove its
non-stationarity. One finding comes first: a single Lagrange multiplier learns
one reward–cost exchange rate, and a curriculum sets that rate on a moving
target; only a multiplier per context removes this. Five decisions are listed
at the end.

---

## 1. The question

| curriculum       | coverage at the end               | non-stationarity on the way |
| ---------------- | --------------------------------- | --------------------------- |
| uniform          | all of Ω                          | none                        |
| staged 1 → 2 → 3 | level 3                           | two jumps                   |
| PLR              | tilted toward the densest layouts | continuous drift            |

> How do the coverage and the non-stationarity of the training distribution
> determine how well a constrained policy generalises to contexts it never
> trained on, in return and, separately, in constraint satisfaction?

## 2. Two open questions from the survey meet in a constrained curriculum

**Non-stationary training data** is expected by the survey to interact with
generalisation (§5.3.1), but its only evidence injected non-stationarity by
hand. A curriculum produces it as a side effect of a method people use, in an
amount the table above shows can be dialled. Nobody has separated what a
curriculum covers from how it moves.

**A second objective** is named by the survey as "particularly relevant for
zero-shot generalisation and the design of new benchmarks" (§6.2). A
constrained learner is the simplest case. It optimises $r - \lambda c$, and
$\lambda$ moves during training; under one curriculum we watched it go from
0.3 to 40. So a curriculum changes both the data and the objective, and only a
constrained setting can show that.

**Cost has a line and return does not.** A Lagrangian learner ends training
with cost exactly at the budget, so after a shift any rise in cost is a
violation, while a drop in return is only a degradation. In our 500 M run
every arm trained at cost 25, the budget; the same policies cost 73 and 108 on
level 3. Part of that rise is difficulty, so every gap is read against a
reference policy trained on the test contexts.

## 3. Twins keep a curriculum's coverage and remove its non-stationarity

Train a curriculum arm and record the distribution it sampled in every round,
$\hat q_1, \dots, \hat q_K$. From that record, train two twins. Evaluate all
three on contexts none of them trained on.

| policy             | distribution in round $k$                              | coverage at the end | change between rounds               | order of the changes |
| ------------------ | ------------------------------------------------------ | ------------------- | ----------------------------------- | -------------------- |
| curriculum arm     | $\hat q_k$                                             | $\bar q$            | that of $\hat q_1, \dots, \hat q_K$ | the curriculum's     |
| time-averaged twin | $\bar q = \frac1K \sum_k \hat q_k$                     | $\bar q$            | zero                                | —                    |
| shuffled twin      | $\hat q_{\sigma(k)}$ for a random permutation $\sigma$ | $\bar q$            | that of $\hat q_1, \dots, \hat q_K$ | random               |

All three have the same coverage, $\bar q$. Arm minus time-averaged twin is
the effect of non-stationarity; arm minus shuffled twin is the effect of its
order.

**The multiplier $\lambda$ moves too**, and differently in each row, because
it is driven by the cost of whatever the round sampled. §6 shows this is a
confound and what removes it.

## 4. The benchmark follows the survey's two axes

**What differs between contexts** is the survey's first axis (§4.1).

| kind                         | factor                                           | the real-world variation it stands for                                                               | how studied, per the survey | exists today |
| ---------------------------- | ------------------------------------------------ | ---------------------------------------------------------------------------------------------------- | --------------------------- | ------------ |
| **state**, the layout        | number of hazards of each kind                   | how crowded the environment is; the benchmark's three difficulty levels differ in exactly this       | most common                 | yes          |
|                              | number of hazards between the robot and the goal | whether obstacles sit on the path, so the robot must thread through them, or beside it, so it only has to steer clear | most common                 | no           |
| **dynamics**, the physics    | ground friction                                  | terrain                                                                                              | least studied               | no           |
|                              | actuator gain                                    | motor wear                                                                                           | least studied               | no           |
| **observation**, the sensors | lidar noise                                      | sensor degradation                                                                                   | well studied                | no           |

Each factor is a value per parallel environment, so none needs a new compiled
program.

**Where the test contexts lie relative to the training contexts** is the
survey's second axis (§4.2, Fig. 3). The survey measures each gap from the
whole training set; we measure it also from the distribution the curriculum
ended on, because that is the only place the constraint was enforced.

| type of generalisation | the test contexts lie … | our test set | read against |
|---|---|---|---|
| none | in the training set, weighted uniformly | uniform over all of Ω | the uniform arm, which trained on exactly this |
| interpolation | in the training set, in a region cut out before training | $T_\text{hole}$: a band of total hazard count | a ceiling arm trained on all of Ω |
| combinatorial interpolation | at a combination of seen values that was never seen together | $T_\text{comb}$: trained on corridor-dense *or* outside-dense; tested on both | a ceiling arm trained on both |
| interpolation, dynamics | in the training set, in a region cut out before training | $T_\text{dyn}$: a band of friction | a ceiling arm per policy class |
| extrapolation | beyond the training range | $T_\text{extra}$: friction and actuator gain outside their ranges | robustness panel only: RL does not extrapolate without an inductive bias, and a curriculum adds none |

**A ceiling arm** is a policy trained on the test contexts themselves. It is
what lets a low score be told from a hard target.

**The policy sees what a deployed robot would see.** Two facts pull against
each other: a deployed robot does not know its own friction, so a policy told
its context is an oracle; but a per-context constraint (§6) needs a policy that
can hold a different trade-off per context, which a blind policy cannot. On
layout factors the lidar already carries hazard density, which is enough. On
dynamics factors the policy sees nothing unless it is recurrent and infers
them within the episode. The critic sees the full context in every case; it
exists only during training.

**Compute** is about 190 arms of 500 M steps at five seeds, roughly 400
GPU-hours on the cluster. The first read, four arms at three seeds, is one day
locally.

## 5. Each test set is a number in the context or a region cut from it

Goal-point's context today is five numbers per episode: four hazard counts,
one per kind (flat disc, solid cylinder, flat square, solid cube), with their
total capped at 20, and the goal radius. The levels sit at totals 12, 16 and
20. Every test set adds a number to that context or cuts a region from it;
none changes an array shape, so none costs a recompile.

**The hole** is a band of total hazard count, for example 17 and 18, between
level points so that staged's levels stay in the training space and every arm
has seen both sides of it. Every distribution samples from Ω with the band
removed; the test set is uniform over the band. Its ceiling is today's uniform
arm. Hazard count is visible through lidar density, so this tests adaptation.

**The corridor** is a strip of fixed half-width around the segment from the
robot's start to the goal. The new context number is how many active hazards
are placed inside it; the rest go outside. Placement already draws candidate
positions and keeps the one with most clearance, so a corridor hazard draws
its candidates in the strip and an outside hazard discards candidates that
fall in it. Training sees inside varied with outside zero, or outside varied
with inside zero; the test set has both nonzero; the ceiling trains on both.
The same number gives the challenge where reward and cost disagree and the
threading probe, so it pays three times. Two decisions are open. The goal
respawns when reached, so the corridor is defined by the first goal only:
either episodes end at the first goal, or the respawned goal defines a new
corridor. And a corridor count must be split across the four hazard kinds;
proportionally to the active counts is the natural rule.

**Friction** is the floor's sliding coefficient, 1.0 today, set per episode
through the physics model inside `step`, the mechanism already verified for
gravity. One MuJoCo detail: a contact pair uses the larger of its two geoms'
friction, and the robot's own geom is 1.0, so lowering the floor alone does
nothing; the context must set the robot's geom. A range such as [0.4, 1.6]
keeps level 3's 1.0 inside; the hole is a band away from it, for example
[1.2, 1.4]. Friction is invisible, so this tests robustness and needs a
ceiling per policy class.

**Actuator gain** multiplies the two actuators' gear, 0.3 today, by a context
number in a range such as [0.6, 1.4], standing for motor wear. It is set and
is invisible like friction. It is proposed for the extrapolation panel only.

**Lidar noise** would be a sixth test set: noise with a context-set standard
deviation on the 32 lidar values. Cheap and well studied, so last.

## 6. A single multiplier learns one reward–cost exchange rate, and a curriculum sets it on a moving target

The policy never sees $\lambda$. It enters only the loss, as the advantage
$A_r - \lambda A_c$, so $\lambda$ is the rate at which the learner trades
reward for cost when it reinforces behaviour. PPO-Lagrange updates it once per
round from the mean cost of that round's episodes,
$\lambda \leftarrow \lambda + \eta\,(\hat C_k - d)$, so it settles where the
policy is on budget for the current training distribution. The policy then
carries that one exchange rate, frozen into its weights, into every context it
meets; when the distribution shifts, the behaviour stays and the cost changes.
The 500 M numbers in §2 are this mechanism.

For the twins this is a confound: an arm and its twin sample different
contexts each round, so they have different $\lambda$ trajectories, and a
difference between them is not attributable to the contexts alone.

**Changing what drives $\lambda$ does not remove the problem; it moves it.**
Each row below holds the constraint on one distribution; the question is
whether that is the distribution the gradient comes from.

| $\lambda$ driven by cost on … | the constraint holds on | the gradient comes from | the two agree | cost |
|---|---|---|---|---|
| the round's training episodes (today) | $\hat q_k$, which moves with the curriculum | $\hat q_k$ | yes | the confound above |
| a share $\varepsilon$ of environments reset from a reference distribution | the reference, fixed across arms | $\hat q_k$ | no | $\varepsilon$ of the data; and misalignment |
| a critic told $\omega$, evaluated at $s_0$ and averaged over the reference | the reference, fixed across arms | $\hat q_k$ | no | relies on the critic generalising over $\omega$; and misalignment |
| nothing: $\lambda$ frozen | — | $\hat q_k$ | — | a value chosen blind |

**Misalignment** is what the second and third rows pay. The constraint says
"be on budget on the reference"; the gradient says "here is how reward trades
against cost on $\hat q_k$". If the reference is harder, $\lambda$ rises until
the one behaviour is on budget on contexts the policy rarely trains on, and
return is given up where caution buys little. Confound and misalignment are
both zero only when the training distribution is the reference, which is the
uniform arm.

**One multiplier per context removes both.** Hold the constraint per context,
$C(\pi,\omega) \le d$ for every $\omega$, with a multiplier $\lambda(\omega)$
updated from the cost observed in that context. On every context the gradient
and the constraint then speak about the same data, and shifting mass between
contexts changes how often each $\lambda(\omega)$ is updated, not what it
converges to. This is a CMDP with one constraint per context, as in Altman's
theory; it is the robust CMDP's constraint without its worst-case return. It
needs a critic told $\omega$, to estimate cost per context, and a policy that
can hold a different trade-off per context; under $\lambda(\omega)$ a blind
policy is a scalar $\lambda$ in disguise, which is why §4 lets the policy see
what a deployed robot sees.

**The study changes in one place.** With the constraint held per context on
the training space, cost generalisation inside that space is enforced rather
than measured; cost outside it and return everywhere remain, which is still
the whole of §4's table.

**The proposal** is a paper in three parts. First, the scalar-$\lambda$ result
on the existing arms plus one reference-anchored arm: the constraint is
enforced on a moving target, and anchoring trades confound for misalignment.
Second, the per-context constraint, which removes both. Third, the twins of
§3 under that learner, so they measure what they are meant to measure.

## 7. What a practitioner takes away

One lesson per part of the paper.

| part | the practitioner learns | and so does what |
|---|---|---|
| 1, the scalar-$\lambda$ result | a Lagrangian learner with one multiplier enforces its budget on whatever it trained on last; a curriculum makes that a moving target | does not read safety off training logs; validates cost on the deployment distribution |
| 2, the per-context constraint | one multiplier per context, with a critic that sees the context, makes the constraint independent of the curriculum | uses $\lambda(\omega)$ whenever a curriculum and a constraint are combined |
| 3a, the evaluation | which curriculum generalises on which type of shift, in return and in cost, with the ceiling showing how much was achievable | picks a curriculum by the kind of shift expected at deployment, not by its training curve |
| 3b, the twins | whether a curriculum's coverage or its non-stationarity is what costs generalisation | chooses between curricula that drift and curricula that jump on evidence rather than habit |

**Part 3a is read through inductive biases.** The survey holds that all
generalisation rests on an inductive bias and asks that a paper name the one
it relies on (§5.4). The agent has three already: an egocentric observation,
so a rotated layout is the same layout; a smooth network, so nearby hazard
counts give nearby behaviour; and one behaviour for all contexts, since the
policy is blind to $\omega$. A curriculum adds none of these; it decides which
get exercised, and in what order. Each curriculum is itself a bias about where
useful training data lies, and the table of generalisation types is where that
bias is seen helping on one type and hindering on another.

| curriculum | its bias: useful data lies … | should help | should hinder |
|---|---|---|---|
| uniform | everywhere equally | nothing in particular; the control | nothing in particular |
| staged | on the easy levels first, then the harder | interpolation between level points | combinatorial shifts whose combination no level contains; anything after a jump that drops a skill |
| PLR, reward score | where the reward critic is most wrong | interpolation where the critic's error is informative | shifts in cost, which the reward critic does not see |
| PLR, cost score | where the cost critic is most wrong | shifts in cost | return, by pulling mass to the densest layouts |
| CURROT, SCG | on the shortest path toward the deployment distribution | the deployment distribution | every test set off that path |

Each cell of "should help" and "should hinder" is a prediction the evaluation
tests. The benchmark itself is the fifth thing a reader takes: factors, test
sets and ceilings on a simulator fast enough for five seeds.

## 8. Questions for the meeting

Each question comes with the consideration that bears on it and the section
that develops it. The first two decide the paper; the rest decide its parts.

| # | question | what bears on it | developed in |
|---|---|---|---|
| 1 | Is "what does a curriculum's coverage and non-stationarity do to a constrained policy's generalisation" the right question for the thesis? | it sits on two gaps the survey names | §1, §2 |
| 2 | Do we adopt one Lagrange multiplier per context? | a single multiplier cannot be made curriculum-independent; anchoring it trades a confound for a misalignment | §6 |
| 3 | Does the twins study run only under the per-context learner, with the scalar-λ arms as its baseline? | under a scalar λ the twins measure λ's trajectory, not the contexts | §3, §6 |
| 4 | For each factor, what would a deployed robot be told, and is that what the policy observes? | a policy told what no deployed robot knows is an oracle; a policy told nothing cannot hold a trade-off per context; hazard count could go either way, friction cannot | §4 |
| 5 | Which types of generalisation go in? | interpolation, combinatorial interpolation and dynamics are the ones a curriculum can affect; extrapolation needs an inductive bias a curriculum lacks | §4 |
| 6 | Are friction and actuator gain acceptable changes to the benchmark? | every factor is a per-episode value and the stock levels stay inside the ranges, but the physics differs from upstream | §4, §5 |

The full design (`research_design.md`) is what changes once these are
answered.