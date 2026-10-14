# Research design: how the training distribution's coverage and non-stationarity determine zero-shot generalisation in constrained RL

Giuseppe Hannen, with Claude. Draft 2026-10-13 for supervisor feedback.

The frame is Kirk, Zhang, Grefenstette and Rocktäschel, *A Survey of
Zero-shot Generalisation in Deep RL*, JAIR 2023
(`literature/zero-shot-generalisation-survey.md`). Section numbers written
as §N refer to that survey. The document is written backwards: first what the
paper shows, then what a reader learns from it, then what must be true and
built for those results to exist.

## 1. The question

A curriculum changes the training distribution over time. Its *coverage* is
which contexts the distribution puts mass on, and how much, by the end of
training. Its *non-stationarity* is how much the distribution changed from
one round to the next on the way there. We ask:

> **How do the coverage and the non-stationarity of the training distribution
> determine how well a constrained policy generalises to contexts it never
> trained on, in return and, separately, in constraint satisfaction?**

Curriculum methods are the instruments: each one produces a different coverage
and a different non-stationarity. Uniform has full coverage and none; staged
ends with level 3's coverage after two jumps; PLR drifts continuously. A contextual constrained MDP with controllable factors
is the apparatus. The product is a mechanism, established by controlled
interventions. A new method follows only if the mechanism dictates one.

## 2. Why this question

The survey ends with gaps its authors could not study because no benchmark
allowed it. Two of those gaps meet in constrained curriculum learning.

The first gap is non-stationarity. The survey writes that "optimisation in RL
has additional issues on top of supervised learning, such as the
non-stationarity of the data distribution", and that these issues "likely
interact with generalisation in a non-trivial way" (§5.3.1, §6.6). Its only
evidence is Igl et al. 2021, who injected non-stationarity artificially and
found that it hurts generalisation even at equal training performance. A
curriculum produces non-stationarity as a side effect of a method people
actually use, and the amount is dialable: a uniform distribution never moves,
a staged one jumps, PLR drifts. The sign of the effect is open, because the
same survey records PLR *improving* generalisation on Procgen (§5.1.3). Nobody
has separated what a curriculum covers from how it moves.

The second gap is multi-objective learning. The survey names "underspecified
or multi-objective reward functions" as "particularly relevant for zero-shot
generalisation and the design of new benchmarks" (§6.2). A constrained learner
optimises $r - \lambda c$, and the multiplier $\lambda$ moves during training.
We have watched it go from 0.3 to 40 under one curriculum and from 0 to 22
under another. So under a constraint a curriculum changes two things at once:
the distribution the policy learns from and the objective it learns. The
unconstrained setting the survey covers cannot show this.

Two further motivations are ours.

Generalisation acquires a second coordinate, and the two coordinates are not
symmetric. The survey measures a gap in return only. A constrained policy also
has a cost, and a Lagrangian learner adjusts λ until training cost equals the
budget. So at the end of training the policy sits exactly on the constraint
boundary, with zero margin. Return has no boundary. When the distribution
shifts, a drop in return is a degradation of degree; a rise in cost, however
small, is a violation, because the policy started on the line. The asymmetry
is in the consequence, not necessarily in the size of the change. A second,
stronger claim is a hypothesis: because the learner spends the whole budget,
it takes exactly as much risk as the training contexts allow, and that risk
calibration may transfer worse than goal-seeking does. Our one-seed 500 M
goal-point run is consistent with both claims but proves neither: every arm
trained at cost 25, the budget; evaluated on level 3 the same policies cost 73
(PLR) and 108 (uniform), while PLR's return fell from 21 to 15 and stayed the
best of the arms. Level 3 is denser than the average context, so part of that
cost rise is difficulty rather than a failure to generalise. Separating the
two needs a reference policy trained on the test contexts, which is why every
test set below has a ceiling arm.

The apparatus the survey asks for half exists. The survey recommends
environments that combine procedural generation with controllable factors
(§4.3), evaluation on several test sets that each isolate one type of
generalisation (§6.3), more work on dynamics variation (§6.4), and "tight
control over the factors of variation" for anyone who wants to understand
generalisation rather than only improve it (§6.5). Goal-point already controls
its hazard counts and seeds its positions, and every factor proposed below is
a per-slot value inside one compiled program.

## 3. What the paper shows

Each figure decides one hypothesis from §6 and carries one takeaway from §4.
Every figure can be drawn whatever the outcome; on an open question a null
result is a result.

| # | figure | what the reader sees | decides |
|---|---|---|---|
| F1 | **The generalisation plane.** One panel per test set. Each point is one arm and one seed, placed at its return gap on the x-axis and its cost gap on the y-axis. The budget is a horizontal line. Motionless policies are marked. | Whether cost generalises worse than return, and which curricula leave the budget. This is the paper's first figure. | H1 |
| F2 | **Radar per arm.** One axis per test set. Two polygons per arm, one for return and one for cost margin, drawn over the uniform arm's polygon as reference. | The survey's multi-dimensional view (§6.3): no curriculum wins on every axis, and the figure shows where each one loses. | all |
| F3 | **Twins.** For each curriculum arm, three bars per test set: the arm, its time-averaged twin, its shuffled twin, with confidence intervals. | Whether non-stationarity changes generalisation when coverage is held equal, and whether its order matters. | H3 |
| F4 | **λ controls.** The same arm with λ free, λ frozen, and λ driven from the deployment distribution. | How much of the curriculum's effect travels through the objective rather than through the contexts. | H4 |
| F5 | **Mechanism scatter.** One point per run. The x-axis is a quantity logged during training: total change of the distribution over rounds, mass placed next to the hole, or training margin. The y-axis is the gap on the matching test set. | Whether the mechanism we log predicts the gap across arms and seeds. This is the "why". | H2, H3 |
| F6 | **Shortfall to the ceiling.** For the combinatorial and dynamics test sets, each arm's distance below a reference policy trained on the test contexts themselves, split by policy class. | Which kinds of generalisation a curriculum can buy and which it cannot, and where a recurrent policy is needed. | H5, H6 |
| T1 | **Per-context constraint table.** For each arm, test set and policy class: the fraction of context cells under budget, the worst cell, and the policy's displacement. | The per-context reading of the constraint, which is what safe deployment means. The displacement column exposes policies that meet the budget by standing still. | H1, H6 |
| T2 | **Benchmark card.** Factors, ranges, test sets, ceilings and protocol, in one table others can reuse. | The apparatus contribution. | — |

## 4. What a reader learns

A practitioner who trains a constrained policy with a curriculum learns five
things.

1. **Where the constraint actually holds.** A Lagrangian learner enforces the
   budget on the contexts it trained on last. Budget compliance does not
   transfer on its own, so cost must be validated on the deployment
   distribution and not read off training logs (F1, T1).
2. **Which part of the curriculum costs generalisation.** If non-stationarity hurts
   (F3), prefer curricula that drift slowly or replay an average over ones
   that jump between stages. If coverage dominates (F5), the final support
   matters more than the path to it. If λ's trajectory dominates (F4), fix the
   distribution the constraint refers to rather than the curriculum.
3. **Which factors a blind policy survives.** For factors the policy cannot
   observe, such as friction or actuator wear, a curriculum can buy robustness
   but not adaptation. The ceilings in F6 say when a recurrent policy is worth
   its cost.
4. **How to read a curriculum paper.** Every curriculum is an inductive bias
   about where useful training data lies (§5.5 below). The results show when
   each bias hinders, which is what the survey asks authors to analyse
   (§5.4).
5. **A reusable benchmark and protocol** (T2): factor kinds, test-set types,
   two outcome coordinates, ceilings, on a simulator fast enough for five
   seeds per arm.

For the field, the paper is the first controlled separation of coverage from
non-stationarity in a curriculum, and the first measurement of how constraint
satisfaction generalises under distribution shift.

## 5. Setting and apparatus

### 5.1 Notation and problem class

A context $\omega \in \Omega$ is the per-episode parameter of the world: its
layout and its physics. A policy $\pi$ earns a return $R(\pi,\omega)$ and
incurs an episode cost $C(\pi,\omega)$ in that context. For a distribution $T$
over contexts, $R(\pi,T)$ and $C(\pi,T)$ are the expectations over
$\omega \sim T$. The budget is $d$. The deployment distribution is $w$, and
$r$ is uniform over $\Omega$. A curriculum produces one training distribution
per round, $q_1, \dots, q_K$; the distribution the gradient actually came from
is the realised one, $\hat q_k$ (`design/intended_vs_realised_curriculum.md`).
A *test set* is a distribution $T$ that no $q_k$ ever samples.

The survey's Definition 5 describes zero-shot policy transfer with a
*controllable* context: the learner may reshape its training distribution
within a fixed support. Our problem is that definition with a constraint
added: maximise $R(\pi,T)$ subject to $C(\pi,T) \le d$.

The constraint can be read in two ways, and we report both. Read in
expectation over $T$, it is what Lagrangian methods enforce. Read per context,
it demands $C(\pi,\omega) \le d$ for every $\omega$ that $T$ can produce, and
that is what safe deployment means.

Every result is a pair, the return gap and the cost gap. Neither is reported
alone, because a policy that stops moving has a zero return gap and meets
every budget.

Every shift from training to a test set has two legs. The first leg runs from
the last training distribution $\hat q_K$ to the uniform distribution $r$. The
curriculum created this leg, it is zero for the uniform arm, and the
constraint was enforced at its start and nowhere else. We call it the
*curriculum shift*. The second leg runs from $r$ to the test set $T$. The
protocol created this leg, and it is what the survey's taxonomy classifies. We
call it the *support shift*. Both legs are measured on every arm.

### 5.2 When a factor can serve as a test dimension

What a test on a factor measures depends on whether the policy can see the
factor. We checked each of goal-point's factors on CPU (2026-10-13).

A **visible** factor changes the observation, so the policy can behave
differently for each of its values. A test on it measures *adaptation*.
Hazard count is visible through lidar density and compass occupancy; corridor
density will be visible the same way; world heading is visible through the
magnetometer.

An **invisible factor that changes the best action** forces one behaviour to
serve every value. A test on it measures *robustness*, which is the
sim-to-real question. Hazard kind is such a factor: swapping a flat disc with
a solid cylinder leaves all 62 observation values unchanged, yet the solid one
costs more to touch. Friction, actuator gain and sensor noise are the same.
These tests are admissible, but only when read against a ceiling policy
trained on the test values, because otherwise a low score cannot be told from
a hard target.

An **invisible factor that does not change the best action** produces a
difficulty curve and nothing else. Goal size is such a factor: the observation
function has no radius input, and driving to the goal's centre is optimal at
every radius. Tests on such factors are not admissible.

A second rule applies to every factor. The curriculum must be able to cause
the outcome. A curriculum acts only through coverage, non-stationarity and λ. It adds no
belief about values it never saw, and the survey states that standard RL does
not extrapolate without such a belief (§4.3). Extrapolation tests are
therefore a robustness panel, not a primary result.

### 5.3 Factors of variation

The survey sorts variation into state, dynamics, observation and reward
(§4.1). Each factor below is a value on the parallel-environment axis, so none
of them creates a new compiled program (`design/what_can_vary_per_slot.md`).

| kind | factor | role | status |
|---|---|---|---|
| state | the four per-kind hazard counts, total ≤ 20 | the level ladder, varied in training. Kind is invisible, so no test set is cut on kind; holes are cut on the total | built |
| state | **corridor density**: the share of active hazards placed within a fixed half-width of the start-to-goal segment, the rest placed outside it | the combinatorial test. The same dimension gives README goal 1 row 7, where the reward and cost landscapes disagree, and the threading probe of goal 2 | to build |
| dynamics | ground friction; actuator gain | the survey's under-studied kind, and the shifts a real robot meets: terrain and wear. Gravity is excluded because it does not vary at deployment for a wheeled robot | to build |
| observation | lidar noise or dropout | cheap, well studied, completes the radar | to build, last |
| reward | the budget $d$ | excluded: one threshold per run (decision with Tristan, 2026-09-22). Varying it changes the task rather than the world | — |

Hazard positions remain seeded procedural generation inside the controlled
counts. That is the combination of PCG and controllable factors the survey
recommends.

### 5.4 Test sets, one per type of generalisation

The survey distinguishes three protocol types for controllable context sets
(its Fig. 3). All three classify the relation between the *support* the
curriculum was allowed to sample, $\Omega_\text{train}$, and the test set. In
*interpolation* the test region lies inside $\Omega_\text{train}$, in a region
cut out before training. In *combinatorial interpolation* it lies in the convex
hull of $\Omega_\text{train}$ but not in it: every factor value was seen, the
combination was not; this tests what Hupkes et al. call systematicity. In
*extrapolation* it lies outside the convex hull. The survey orders their
difficulty as interpolation, then combinatorial, then extrapolation (§4.3).

The curriculum shift of §5.1 is not a fourth type. The three types classify
the second leg of a gap, $r \to T$; the curriculum shift is the first leg,
$\hat q_K \to r$, and is present in every test set below. Evaluating on $r$
itself, which every run already does, measures the first leg alone.

| test set | type | instance on goal-point | read against |
|---|---|---|---|
| $r$ | none: the first leg alone | uniform over Ω, already evaluated every round | the uniform arm, whose first leg is zero |
| $T_\text{hole}$ | interpolation | an interior band of total hazard count removed from training, away from the level points 12, 16 and 20 | a ceiling arm trained on all of Ω |
| $T_\text{comb}$ | combinatorial | training sees corridor dense with outside empty, or corridor empty with outside dense; the test has both dense | a ceiling arm trained on the convex hull |
| $T_\text{dyn}$ | interpolation on dynamics | a hole in the friction range. For a blind policy this is a robustness test; for a recurrent one, adaptation | a ceiling arm, per policy class |
| $T_\text{extra}$ | extrapolation | friction beyond its range. Total hazard count ≥ 21 only after measuring overlap against count, because above 20 the placement overcrowds silently (README, Known defects) | no ceiling claimed; robustness panel |

The Ω₂ placeholder is retired. Its goal-size direction fails §5.2 and its
density direction is confounded by overcrowding.

### 5.5 Curricula as inductive biases

The survey asks authors to name the inductive bias they introduce and to
analyse when it hinders (§5.4). Every curriculum is a belief about where
useful training data lies.

| distribution | the belief | when it should hinder |
|---|---|---|
| uniform | every context is equally informative | never by this route; it is the control |
| staged 1 → 2 → 3 | easy before hard, the ladder's order is the right order, and skill transfers upward | when an "easier" level is a different task rather than a sub-task (the velocity-ant collapse), and when λ calibrated on one stage is wrong on the next |
| PLR with the reward score | the reward critic's error marks where learning happens | when the critic is about equally wrong everywhere (spread 1.6× in the 500 M run) |
| PLR with the cost score | the cost critic's error marks where learning happens | it pulls mass to the densest layouts, where the cost margin is zero; there λ wound up and the policy collapsed |
| CURROT and SCG | move toward $w$ along the shortest path in context space while performance holds | when the shortest path crosses contexts where reward and cost disagree |

### 5.6 Policy class

A deployed robot does not know its own friction or its sensor noise. The
survey makes the same point for sim-to-real (§3.5). So a policy that is *told*
$\omega$ is an oracle, and not a class we can report deployment results for.
Two classes are deployable. A **blind** policy, today's observation, must find
one behaviour for all values of an invisible factor; it measures robustness. A
**recurrent** policy can infer the factor from the episode so far; it measures
adaptation.

One contextualisation is required and is harmless: telling the **critic** the
context while the policy stays blind. The critic exists only during training,
so this changes the learner and not the benchmark. It is standard asymmetric
actor-critic, and it gives the cost critic sharper targets across contexts.

| class | deployable | role | cost to us |
|---|---|---|---|
| blind | yes | baseline; robustness on invisible factors | none |
| critic told ω, policy blind | yes | the required contextualisation | small |
| recurrent policy | yes | adaptation on invisible factors (F6) | moderate: recurrent PPO in CRAX |
| policy told ω | no | diagnostic ceiling for F6 only | small, but changes the observation space; Tristan first |

## 6. Hypotheses

Registered before the runs. Each names the figure that decides it.

- **H1, margin.** Constraint satisfaction generalises worse than return under
  every shift, because training ends at zero cost margin. F1 decides it by
  comparing the relative cost gap $\Delta C / d$ with the relative return gap
  $\Delta R / R$. Phases in which λ reached zero, so the margin was positive,
  should show a smaller cost gap.
- **H2, coverage.** The interpolation gap into the hole is ordered by how much
  mass each arm's $\hat q$ placed next to the hole. F5 decides it.
- **H3, non-stationarity.** At equal coverage, more change over rounds changes generalisation.
  Igl et al. predict for the worse; PLR on Procgen suggests for the better.
  F3 decides the sign, comparing each arm with its twins at matched final
  training return and cost.
- **H4, non-stationarity of the objective.** Part of the curriculum's effect travels through
  λ's trajectory rather than through the contexts. F4 decides it.
- **H5, systematicity.** No arm reaches the ceiling on the combinatorial test
  set, and the shortfall is smaller for arms that covered both halves of the
  training set evenly. F6 decides it.
- **H6, policy class.** On dynamics factors a blind policy cannot meet the
  per-context constraint and a recurrent policy can, and the curriculum's
  effect differs between them. F6 and T1 decide it.

## 7. Experimental design

**Arms.** Uniform, staged, PLR with each of its three scores, and later CURROT
and SCG. Each runs under PPO-Lagrange and under PPO-PID, because λ's dynamics
belong to the learner and one learner is one data point. Each test set that is
harder than training gets a ceiling arm.

**Controls that separate coverage from non-stationarity.** This is the core device. Each
control is a `ContextDistribution` that replays a stored sequence, so the
trainer is untouched.

| control | what it holds fixed | what it isolates |
|---|---|---|
| **time-averaged twin** | the coverage: it samples, without change over time, from the average $\bar q = \frac{1}{K}\sum_k \hat q_k$ of a finished curriculum arm | non-stationarity. Same coverage, no change over rounds |
| **shuffled twin** | the set of per-round distributions, replayed in random order | order. The movement without its direction |
| **frozen λ** | the objective: λ is held at the uniform arm's final value | change in the objective, separated from change in the contexts |
| **$w$-anchored λ** | the distribution the constraint refers to: a share $\varepsilon$ of the parallel environments is reset from $w$, and λ is updated from those alone | whether the curriculum-shift cost gap is the λ mechanism. If it is, this control is also the remedy |

**Mechanism measurements**, all from logs that exist. Coverage per context
cell comes from the `experienced` histograms. Non-stationarity is
$\sum_k \|q_{k+1} - q_k\|_1$ over the `intended` series. The λ trajectory and
the training margin $d - \hat C_k$ come from the training metrics. The cost
critic's error per context comes from the `cost_value_loss` heatmaps.

**Outcome metrics**, per arm and test set, from whole episodes started at
step 0: return, cost, margin, the fraction of context cells under budget, the
worst cell, displacement, both legs of the shift, and distance to the ceiling
where one exists.

**Statistics.** At least five seeds per arm. Interquartile means with
stratified bootstrap intervals, following Agarwal et al. 2021.

## 8. What it requires

**Engineering.** Every item is testable on CPU and none changes the shape of a
compiled program.

| item | size | step |
|---|---|---|
| a training space with holes that every distribution respects: uniform's table of admissible tuples is filtered, PLR's buffer inherits it, staged's level points are validated to lie inside it | small | 0 |
| test sets as `EvaluationRegion`s; per-cell budget metrics; displacement | small | 0 |
| store the realised per-round context sequence at full resolution; today only 12-bin histograms are kept | small | 0 |
| twin distributions that replay a stored sequence, time-averaged or shuffled | small | 1 |
| frozen λ and $w$-anchored λ in the Lagrangian update, via a slot mask | small | 1 |
| critic told ω: a network input flag on the learner only | small | 2 |
| corridor placement: a candidate generator parameterised by the start-to-goal segment, with the share in ω; overlap against share measured first | medium | 3 |
| friction and actuator gain from ω via `tree_replace` in `step`, with level 3 inside the ranges | medium | 3 |
| lidar noise in the observation function | small | 3 |
| recurrent policy | large; deferred unless F6 demands it | 5 |
| CURROT and SCG as `ContextDistribution`s | medium each | 5 |

**Compute.** One 500 M goal-point arm takes about two hours on the RTX 4080.
The full study is roughly five curricula, two learners, two policy classes
and five seeds, about 100 arms; plus two twins per curriculum arm on one
learner, about 50; plus λ controls, about 20; plus ceilings for four test sets
at five seeds, about 20. That is about 190 arms, or 400 GPU-hours, which is
the SLURM cluster. Step 0 is four arms at three seeds, 24 hours, and runs
locally. If 200 M steps turn out to suffice for the generalisation questions,
everything is 2.5× cheaper; step 0's curves will tell.

**Reading before any of this enters the thesis.** Koprulu et al. 2025 (SCG),
to learn which distribution its constraint is enforced on. CARL (Benjamins
et al. 2021), for comparable dynamics protocols. Igl et al. 2021, for the
exact quantity they call non-stationarity. Wu, Dyer and Neyshabur 2021 for the
shuffled-order control in supervised learning; this citation is from memory
and must be verified.

**With Tristan.** Friction and gain as per-slot physics; the told-policy
observation if it is used; anything else that changes the benchmark's
behaviour.

## 9. Steps

| # | step | yields | needs |
|---|---|---|---|
| 0 | The protocol on today's Ω: holes, $T_\text{hole}$, per-cell metrics, the stored sequence. Run PID with uniform, staged, PLR and PLR-cost at three seeds. | F1 and T1 with seeds; the first read on H1 | nothing |
| 1 | Twins and λ controls against the arms of step 0. | F3 and F4 on goal-point alone; the cheapest result in the plan | 0 |
| 2 | Critic told ω. | a sharper cost critic; the prerequisite for dynamics factors | nothing |
| 3 | Corridor density, friction, gain, noise, ceilings. | the benchmark; F6; H2 and H5 | 2 |
| 4 | The interventions the mechanism predicts: a margin on $d$, anchored λ, re-initialisation. | a method, if any | 1 |
| 5 | Recurrent policy; CURROT and SCG. | H6; positioning against prior work | 3 |
| 6 | The full study on the cluster. | the paper | all |

Three thesis chapters ride the same runs: the constrained TMA-challenges table
(README goal 1), skill probes as further `EvaluationRegion`s (goal 2), and
data attribution (goal 3, a separate paper if it works).

## 10. What would change the plan

- If SCG already anchors its constraint to the target distribution or already
  evaluates cost off-distribution, then step 4's remedy is not new. The
  mechanism, the protocol and the twins stand.
- If step 0 shows the cost gap within noise, H1 falls, and the
  non-stationarity study continues on return alone. We know this within two
  weeks.
- If the twins match their arms on every test set, non-stationarity does not
  matter at this scale, and the paper is about coverage and λ. On an open question that is
  a clean negative.
- If F6 shows blind and recurrent policies equal on dynamics, the recurrent
  policy is dropped and the robustness reading suffices.
