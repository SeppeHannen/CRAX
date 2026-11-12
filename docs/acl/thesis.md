# Does a constrained policy keep its cost budget in environments it has not seen?

Working paper of the graduation project, written in the shape the thesis will
take. `README.md` is the task board; this is the argument.

Status, 2026-11-12. The design in §4 and §5 is proposed, not yet agreed. Of
the cases in §5.2, most of case A has run at one seed (§6). Case C and one
curriculum method are built and not run. Case B is not built.

## Abstract

A constrained policy meant for many environments is trained on a distribution
of them, and the standard constraint bounds its cost only on average over that
distribution. We ask whether a policy that meets its cost budget in training
also meets it in environments it has not seen, and whether an automated
curriculum changes that. An average cannot answer this: a policy can be on
budget on average and over it in every hard environment it trains on. We
therefore first require the budget to hold in each training environment.
Starting from PPO-Lagrange, which has one Lagrange multiplier for all
environments, we compare three learners: that one, one whose multiplier is a
function of the environment, and one whose policy is also told the
environment. We hold parts of the environment space out of training and
compare cost with the budget in each held-out environment, under
interpolation, extrapolation and combinatorial generalization. In the first
seed, the single multiplier already breaks the budget in hard environments it
trained on, and only the learner whose policy is told the environment holds it
there and mostly beyond, at a loss of return.

## 1 Introduction

A constrained policy maximises return while keeping its expected cost per
episode under a budget $d$. A policy meant for more than one environment is
trained on a distribution of them. We describe an environment by its
**context** $\omega$, a vector of factors. In our benchmark, the goal-point
task of CRAX (Constrained RL Accelerated with JAX), a robot drives to goals
between hazards, and the factors include how many hazards stand in its arena
and how slippery its floor is. Some factors the policy can observe through its
sensors (the hazards, in its lidar). Others it cannot (the floor).

**The constraint holds on average only.** The usual constraint bounds the cost
averaged over the training distribution. A Lagrangian method enforces it with
one multiplier, the price of cost, fitted to that distribution as a whole. The
average says nothing about any single context. Call a context *hard* when the
budget binds there, that is, when a policy that ignored cost would exceed it,
and *easy* otherwise. With a budget of 25, a policy that costs 5 in three easy
contexts and 85 in a hard one meets the constraint.

**That makes generalization hard to measure.** We want to know whether a
policy stays within its budget in contexts it has not seen. If it is over
budget there, that may be a failure to generalize, or it may be the imbalance
that training already allowed. As long as training itself leaves hard contexts
over budget, a violation in an unseen context does not show which.

**So the study has an order.** We first ask what it takes for the budget to
hold in each training context. Only then can we ask whether it also holds in
contexts held out of training. And only against that baseline can we ask
whether a curriculum, which changes how often each training context is
sampled, makes a difference.

**Research questions.**

- **RQ1.** How can the cost constraint be made to hold in every training
  context, rather than in expectation over the training distribution?
- **RQ2.** To what extent does constraint satisfaction on the training
  contexts transfer zero-shot to held-out contexts, under interpolation,
  extrapolation and combinatorial generalization, and for a factor of
  variation the policy can observe and one it cannot?
- **RQ3.** Does automated curriculum learning over the training contexts
  change zero-shot constraint satisfaction on held-out contexts, relative to
  uniform sampling?
- **RQ4.** If it does, is the effect due to the order in which contexts are
  presented, or to the marginal distribution over contexts that the curriculum
  induces?

Zero-shot means without further training. Interpolation, extrapolation and
combinatorial generalization say where a held-out context lies relative to the
trained ones (§3.2).

**How each question is answered.**

| Question | Answered by | Where |
|---|---|---|
| RQ1 | three learners: standard PPO-Lagrange, then the context told to the multiplier, then also to the policy; compared in the contexts they train on | §4, §5.2 |
| RQ2 | the same runs, scored in contexts held out of training | §5.2 |
| RQ3 | curricula against uniform sampling, on the same training and held-out contexts | §5.4 |
| RQ4 | for each curriculum, a twin run that samples the same mixture of contexts without the curriculum's order | §5.5 |

§2 shows what the literature leaves open behind each question, and §3 states
the problem formally. §6 reports the runs finished so far.

## 2 Related work

The literature leaves three things open, one behind each group of questions:

| Open | Shown in | Question |
|---|---|---|
| Where training covers many environments, the constraint is enforced on their average, even by methods whose policy is told the environment. | §2.1 to §2.3 | RQ1 |
| Whether the budget holds in unseen environments is rarely measured, and then mostly as an average over them. | §2.4 | RQ2 |
| We found no test of whether a curriculum changes that. | §2.5 | RQ3, RQ4 |

### 2.1 The price of cost

Safe reinforcement learning is usually posed as a constrained Markov decision
process: maximise expected return subject to a budget on expected cumulative
cost (Altman, 1999). Lagrangian methods turn the constraint into a penalty
whose weight, the multiplier $\lambda$, is adjusted by dual ascent or a PID
rule until the budget is met (Tessler et al., 2019; Ray et al., 2019; Stooke
et al., 2020). At convergence $\lambda$ is the price of cost in units of
reward. The right price depends strongly on the task: fixed penalty weights
have to be retuned for each task (Bohez et al., 2019), and the best fixed
multiplier differs by orders of magnitude between tasks (Spoor et al., 2025;
Tetteh & Fleming, 2026).

Once training draws from a family of environments, the standard formulation
bounds the cost averaged over the family, so one price is applied to
environments that need different ones. The effect is reported in three places:

- With the multiplier held fixed, the worst-decile episode cost grows from
  about 41 with one hazard to about 397 with sixteen, against a budget of 25
  (Tetteh & Fleming, 2026).
- A multiplier learned on an easier task has to be re-learned on a harder one,
  and the agent is unsafe in between (Usmanova et al., 2024).
- The averaged constraint is named as one reason a domain-randomised policy is
  unsafe at deployment (As et al., 2025).

### 2.2 Methods for varying environments

The literature answers variation between environments in three ways, which
differ in what is assumed to be known about the environment.

| The environment is | Answer | Works | Limit |
|---|---|---|---|
| unknown | robust constrained RL: protect against the worst case in an uncertainty set | Russel et al., 2020; Mankowitz et al., 2020; Queeney & Benosman, 2023; As et al., 2025 | safety is bought by being conservative everywhere |
| inferred | safe meta-RL and latent-task methods: adapt the policy to each new task | Khattar et al., 2023; Guan et al., 2024; Moeini et al., 2025; Xu & Zhu, 2025 | the right trade-off between return and cost is found only by adapting on that task |
| known | contextual MDP: every environment has a context that can be told to the learner | Hallak et al., 2015 | our setting |

Where the policy is told about its environment, the constraint is still
averaged. Koprulu et al. (2025) tell the policy the context and design a
curriculum over contexts, but their constraint bounds the average over the
distribution the policy is meant for. Han & Feng (2026) condition policy and
critics on an inferred latent, but keep one PID-controlled multiplier; they
assume the trained policy is safe in each environment, and nothing enforces
it. So even where the policy is told the environment, the constraint is not.

### 2.3 Multipliers as functions

RQ1 leads to a multiplier that is a function of the context (§4). Multipliers
have been made to depend on other things:

| The multiplier | Works | Why it is not a multiplier per context |
|---|---|---|
| is a function of the state | Bohez et al., 2019; Ma et al., 2021a; Yu et al., 2022; Zhang et al., 2026 | the bound then holds from every state. That is stricter than a budget on expected cost, and can be infeasible or conservative where the budget is satisfiable. |
| is an input to the policy, with dual updates continuing at execution | Calvo-Fullana et al., 2021 | it corrects for cost already incurred; a function of the context sets the price beforehand |

Outside reinforcement learning, a multiplier has been predicted from
constraint features in supervised learning (Narasimhan et al., 2020) and from
network configurations in wireless resource allocation (Uslu et al., 2025).
Policies have also been conditioned on the budget, which varies the budget in
a fixed environment (Liu et al., 2023; Yao et al., 2023).

Two works come closest, from opposite sides. Bohez et al. (2019) train a
multiplier network jointly with a policy conditioned on a task variable, but
the multiplier is statewise and the variable is a reward target. Xu & Zhu
(2025) have a constraint and dual variables per task, but solve for them after
the task is revealed. Missing is the combination: a constraint per context
whose multiplier is learned, during training, as a function of the context.
RQ1 examines it.

### 2.4 Evaluation in unseen environments

Benchmarks do not ask whether the budget holds in unseen environments. Safety
Gym resamples the layout every episode but defines no held-out split (Ray et
al., 2019), Safety-Gymnasium reports on its training tasks (Ji et al., 2023),
and a survey of trustworthy RL lists the relation between safety and
generalization as open (Xu et al., 2022).

Where held-out tests exist, constraint satisfaction degrades: standard safe RL
is safe in 51% of perturbed test cases (Queeney & Benosman, 2023), and a
Lagrangian policy with a budget of 10 in training costs about 36 at deployment
(Han & Feng, 2026).

These tests almost always report cost as an average over test environments,
which repeats in evaluation the averaging that the constraint performs in
training. The exceptions report the share of policies that satisfy the
constraint (Queeney & Benosman, 2023), violation rates per obstacle type
(Chane-Sane et al., 2024) and tail statistics over episodes (Tetteh & Fleming,
2026). Cost has been compared with the budget per test environment for other
kinds of variation (Mankowitz et al., 2020; Yao et al., 2023; Xu & Zhu, 2025);
for the layout of hazards we found no such comparison. RQ2 compares cost with
the budget in each held-out context, and separates the kinds of generalization
that Kirk et al. (2023) distinguish.

### 2.5 Curricula

Koprulu et al. (2025) is the one work we found that designs a curriculum for
constrained RL over contexts. It neither enforces the constraint per context
nor tests on held-out contexts. Whether a curriculum such as Prioritized Level
Replay (Jiang et al., 2021) changes the generalization of constraint
satisfaction is, as far as we found, open. That is RQ3 and RQ4. This part of
the literature has not been searched systematically yet.

## 3 Problem

This section states formally the two constraints that RQ1 contrasts (§3.1)
and what it means to generalize in RQ2 to RQ4 (§3.2).

### 3.1 A budget on average, or in each context

A context $\omega$ from the context space $\Omega$ is drawn at the start of
each episode and stays fixed during it. For a policy $\pi$,
$J_r(\pi, \omega)$ is the expected return and $J_c(\pi, \omega)$ the expected
cost of an episode in $\omega$. The budget is $d$. The training distribution
over contexts is $p$, and the contexts it can draw are $\Omega_\text{train}$.
Both problems below maximise $\mathbb{E}_{\omega \sim p}\,[J_r(\pi, \omega)]$.
They differ in the constraint.

| Constraint | Statement | Multipliers |
|---|---|---|
| Averaged (standard) | $\mathbb{E}_{\omega \sim p}\,[J_c(\pi, \omega)] \le d$ | one, $\lambda$ |
| Per context (ours) | $J_c(\pi, \omega) \le d$ for every $\omega$ in $\Omega_\text{train}$ | one for each context: a function $\lambda(\omega)$ |

A Lagrangian method has one multiplier for each constraint (§2.1), which gives
the last column. §4 builds learners for both rows.

### 3.2 Generalization

Training draws contexts only from $\Omega_\text{train} \subset \Omega$. The
trained policy is then scored zero-shot in the **held-out** contexts, those
outside $\Omega_\text{train}$. Following Kirk et al. (2023), a held-out
context is one of three kinds, by where its values lie relative to the trained
ones:

| Kind | The held-out values are | Example |
|---|---|---|
| Interpolation | between trained values of a factor | 12 hazards, after training on 0 to 10 and on 14 to 16 |
| Extrapolation | beyond every trained value of a factor | 18 hazards, after training on at most 16 |
| Combinatorial | a pairing of values that were each trained on, but never together | many hazards on a slippery floor, after training on many hazards and on slippery floors, never both at once |

**When has a policy generalized?** There is one criterion for cost and one for
return. Both use a **reference run**: the same learner trained on all of
$\Omega$, which shows what was achievable in a held-out context.

- **Cost:** $J_c(\pi, \omega) \le d$ in the held-out $\omega$. The budget is a
  fixed line, so the criterion itself needs no comparison. Where cost is over
  the budget, the reference run says why: if the reference run is over the
  budget there too, the excess is not, or not only, a failure to generalize.
- **Return:** compared with the return of the reference run. A harder context
  lowers return even for a policy that generalizes.

## 4 Learners

RQ1 asks how the budget can be made to hold in every training context. This
section builds three learners, each of which knows more about the context than
the one before, and states for each a hypothesis about where the budget holds.
§5 tests the hypotheses.

### 4.1 Where the context can enter

The context can be told to the multiplier, to the policy, or to both. It does
a different job at each.

| Component | What telling it the context does | Kept after training | May be told |
|---|---|---|---|
| Multiplier | decides *where the budget is enforced*: one $\lambda$ enforces it on the average over contexts, a multiplier per context enforces it in each | no | anything |
| Policy | decides *whether behaviour can differ between contexts*: a policy can act on a different price per context only if it can tell the contexts apart | yes | only what a deployed robot could know |
| Critics | lets them estimate return and cost per context; they are told the context in every learner | no | anything |

The last two columns matter in §5.1: a factor that no robot could know is told
to the multiplier and never to the policy.

### 4.2 Three learners, each with a hypothesis

Starting from standard PPO-Lagrange, the context is told first to the
multiplier and then also to the policy. That makes three learners. Each
removes the reason the one before it is expected to fail.

| | Multiplier | Policy input | Hypothesis, for the contexts the learner trains on | In the code |
|---|---|---|---|---|
| L1 | one $\lambda$ | sensors | **H1.** The budget holds on average over the training distribution and is exceeded in the hard contexts. | `scalar blind` |
| L2 | $\lambda(\omega)$ | sensors | **H2.** The budget holds in every training context, at the price of return in contexts that are easier than they look. | `additive blind` |
| L3 | $\lambda(\omega)$ | sensors and context | **H3.** The budget holds in every training context and is met tightly, with cost at the budget, wherever it binds, without L2's loss of return. | `additive told` |

The policy update is PPO-Lagrange's in all three. The policy is trained on the
advantage $A_r - \lambda A_c$: the reward advantage minus the cost advantage
at the price $\lambda$. The learners differ in which $\lambda$ prices a
transition and in the policy's input. Below, $c$ is a cost per step,
$\tilde d = d / T$ the budget per step for episodes of $T$ steps, $\eta$ a
step size, and $[\cdot]_+$ clips at zero.

**L1, one multiplier.** This is standard PPO-Lagrange. One $\lambda$ prices
every transition. After each batch it is moved by the mean cost $\bar c$ of
the whole batch: $\lambda \leftarrow [\lambda + \eta\,(\bar c - \tilde d)]_+$.
Over-spending in hard contexts is cancelled by under-spending in easy ones, so
$\lambda$ stops moving once the average is on budget. Hence H1.

**L2, a multiplier per context.** The multiplier becomes a function
$\lambda(\omega)$, and a transition in context $\omega_t$ is priced at
$\lambda(\omega_t)$. Each transition gives a target,
$[\lambda(\omega_t) + \eta\,(c_t - \tilde d)]_+$, and $\lambda(\cdot)$ is
fitted to the targets. A context's multiplier now rises for as long as that
context is over budget, whatever the rest of the batch does. Over-spending
there is no longer cancelled elsewhere, as far as the form of
$\lambda(\omega)$ can hold contexts apart (§4.3). But the policy still has
only its sensors. Where two contexts look alike to it, such as two floors of
different drag, it cannot behave differently in them, and the only way to
satisfy the harder one is to be cautious in both. Hence H2.

**L3, a multiplier per context and a policy that is told the context.** The
policy can now hold a different trade-off between return and cost in each
context. Hence H3.

H1 to H3 are about training contexts, which is what RQ1 asks. For held-out
contexts (RQ2) we state no hypotheses yet. The fourth combination, one
$\lambda$ with a policy that is told the context, is not studied: no
hypothesis needs it.

### 4.3 The form of $\lambda(\omega)$

$\lambda(\omega)$ is not a free function. It is
$f\big(\sum_i h_i(\omega^i)\big)$, where $\omega^i$ is factor $i$ of the
context: each factor adds $h_i$ to a difficulty, and $f$ turns difficulty into
a price. Every $h_i$ is nondecreasing in the direction that makes its factor
harder (more hazards, a smaller goal, lower drag), and $f$ is nondecreasing.
This assumes that difficulty only rises along each factor and that factors do
not interact (`design/contextual_learner.md`). We use this form, `additive` in
the code, because in earlier runs it kept more return than a free network at
the same cost (`design/generalization_evaluation.md`).

## 5 Experiments

Stage 1 (§5.2) answers RQ1 and RQ2, stage 2 (§5.4) answers RQ3 and stage 2b
(§5.5) answers RQ4. All stages use one environment (§5.1) and one measurement
(§5.3).

### 5.1 Environment: one factor the policy can observe, one it cannot

H2 says a policy pays for what it cannot observe, and RQ2 asks about a factor
the policy can observe and one it cannot. The context therefore needs one of
each.

**Task.** Goal-point from CRAX. A point robot drives to goals in a fenced
arena of 5 m by 5 m; its return is the number of goals it reaches. It pays
cost while it is on a passable hazard or touches a solid one. The budget is a
mean cost of 25 per episode of 1,000 steps. The policy is feed-forward,
without memory, and its sensors include a lidar that tells passable hazards
from solid ones.

**Context.** Six factors:

| Factor | Range | Observable to the policy |
|---|---|---|
| Flat discs (passable) | 0–12 | partly, through the lidar |
| Solid cylinders | 0–8 | partly, through the lidar |
| Flat squares (passable) | 0–6 | partly, through the lidar |
| Solid cubes | 0–4 | partly, through the lidar |
| Goal radius | 0.14–0.22 m | no |
| Drag: how strongly the floor brakes the robot, as a multiple of the benchmark's | 0.5–1.5 | no |

Two quantities carry the experiments:

- **Hazard total**: the sum of the four counts. Contexts with a total over 20
  are not part of the space. The policy can observe the total in part: the
  lidar shows the hazards within reach.
- **Drag.** Low drag is a slippery floor: at 0.5 the robot's top speed and the
  time it needs to stop are both twice the benchmark's. The policy cannot
  observe the drag: without memory it cannot infer it from how the robot
  accelerates.

**Who is told what.**

| Component | Is told |
|---|---|
| $\lambda(\omega)$ and the critics | every factor that varies in the run, the drag included |
| Policy of L3 | the four counts and the goal radius; never the drag, because no robot knows its floor in advance |
| Policy of L1 and L2 | nothing |

For H2 this means two things. On hazards, the lidar lets a policy tell
contexts apart, so L2 should lose little return. On drag, no policy can tell
contexts apart, so L2 should lose much, and L3 with it: each has to be
cautious enough for the most slippery floor at all times.

**Differences from the benchmark** (`upstream_notes.md`, `known_defects.md`):

| Difference | Why |
|---|---|
| Hazard counts vary per episode. The benchmark's three levels (totals 12, 16 and 20) are three points of the context space. | holding contexts out needs a space of them |
| Drag is new. It scales the damping of the robot's two slide joints. | that damping is what brakes this robot; its friction coefficients change nothing |
| The lidar reports passable and solid hazards separately. | the earlier lidar could not tell them apart, which caused rare, very costly episodes |
| Contacts are solved with the Newton solver of MJX, MuJoCo's JAX backend. | the stock solver kicked a robot away from a hazard it pressed on, which made pressing cheap |

### 5.2 Stage 1: does the budget hold in training, and outside it? (RQ1, RQ2)

Training samples contexts uniformly from $\Omega_\text{train}$, and the
held-out contexts are the test. Three cases follow from the two quantities:
each alone, then both.

| Case | Varies in training | Trained on | Held out | Kind |
|---|---|---|---|---|
| A. Hazards | hazard counts and goal radius; drag fixed at 1 | totals 0–10 and 14–16 | totals 11–13 | interpolation |
| | | | totals 17–20 | extrapolation |
| B. Drag | drag only; hazards fixed | drag 0.65–0.95 and 1.1–1.5 | drag 0.95–1.1 | interpolation |
| | | | drag 0.5–0.65 | extrapolation |
| C. Hazards and drag | everything | every total and every drag, but totals 11–20 only at drag 0.9–1.5 | totals 11–20 at drag 0.5–0.9 | combinatorial |

- **Cases A and B each hold out values of one quantity**, so that a result can
  be traced to it. In case C nothing is outside the trained range of either
  quantity; only the pairing is new.
- **The goal radius is never held out.** Where it varies, it is sampled the
  same way in training and in testing.
- **In case B, extrapolation is towards low drag**, the unsafe side.
- **In case B the hazards are fixed at a total where the budget binds**: 14,
  going by case A's first runs (§6). If the hazards varied too, an effect of
  drag would be diluted, because most contexts are far under the budget. In
  those runs the mean cost is under 10 up to seven hazards.
- **Every learner is trained twice per case**: once with the held-out contexts
  left out, and once as the reference run (§3.2).

RQ1 is read in the trained contexts: which of H1 to H3 hold. RQ2 is read in
the held-out contexts, by the criteria of §3.2.

### 5.3 Measurement

After training, the final policy is rolled out zero-shot in every **cell** of
its case: a part of the context space with one hazard total, one narrow band
of drag, or both. Every run is evaluated in the same contexts.

| Case | A cell is | Cells | Episodes per cell |
|---|---|---|---|
| A | one hazard total | 21 | 128 |
| B | a band of drag 0.05 wide | 20 | 128 |
| C | one hazard total by a band of drag 0.1 wide | 210 | 32 |

| Reported | Why |
|---|---|
| Mean cost per cell, against the budget | an average over all held-out contexts can hide one cell over the budget and another under it |
| Median cost and the share of episodes over the budget, per cell | a mean can be carried by a few costly episodes |
| Return per cell, next to cost and against the reference run | a policy that stands still meets any budget |

The reasons in full are in `design/generalization_evaluation.md`.

### 5.4 Stage 2: does a curriculum change it? (RQ3)

A curriculum changes how often each training context is sampled, and nothing
else: it draws from the same $\Omega_\text{train}$ and is scored in the same
cells. Each curriculum run is compared with the uniform run of the same
learner and seed, which is why stage 2 needs stage 1 first.

| Choice | Value | Why |
|---|---|---|
| Case | C | it has both quantities |
| Learners | L1 and L2 | L3 is left out: its policy cannot be told the drag (§5.1) |
| Samplings | uniform (stage 1's runs); Prioritized Level Replay with two priority scores; further methods still to be chosen | the further methods should differ from it in principle, not in tuning |

**H4.** A curriculum changes which contexts meet the budget only through the
single $\lambda$, which settles for whatever is currently being sampled. With
$\lambda(\omega)$ that path is closed, so a curriculum should change less for
L2 than for L1.

### 5.5 Stage 2b: order or mixture? (RQ4)

A curriculum differs from uniform sampling in two ways at once: which contexts
it samples most (its mixture) and when it samples them (its order). To
separate the two, every curriculum run gets a twin: a run that samples from a
fixed distribution equal to what the curriculum sampled over its whole run.
The twin has the curriculum's mixture without its order. If the twin does as
well as the curriculum, the order did not matter.

### 5.6 Training and number of runs

Every run trains for 500 M steps on 8,192 parallel environments and takes
about 2.6 hours on an RTX 4080 SUPER. We use three seeds. One is not enough:
in earlier experiments two runs of the same setup differed as much as
different setups did. At about a hundred runs, the full design needs a
cluster. How to run is in `README.md`.

| Stage | Runs per seed | At three seeds |
|---|---|---|
| 1 | 3 cases × 3 learners × 2 (held out, reference) = 18 | 54 |
| 2 and 2b | 2 learners × 4 curricula × 2 (curriculum, twin) = 16 | 48 |

The count for stages 2 and 2b assumes four curricula; their number is not
fixed yet.

## 6 Results

Case A, seed 0. The four runs scored so far, on the CPU while the GPU trained;
the references of L2 and L3 were still training. One seed is a lean, not a
result.

The tables below are each run's final checkpoint. The write-up,
`experiments/2026-11-12_goal_point_hazard_total_500M.md`, adds the mean of the
last five evaluations and the multiplier through training. Two readings change
with it: L1's final checkpoint was saved in a cost spike (its 33.2 on totals
11–13 is 25.9 over the last five evaluations), and L2's multiplier at its
hardest contexts was still rising at 500 M (11.9 to 14.6 over the last 85 M
steps), so H2 is not decided.

Mean cost per region, and how many of its hazard totals have a mean cost over
the budget of 25:

| Run | Totals 0–10, 14–16 | Totals 11–13 | Totals 17–20 | Return at 17–20 |
|---|---|---|---|---|
| L1 | 19.0, 4 of 14 over | 33.2, 3 of 3 over | 133.9, 4 of 4 over | 19.9 |
| L2 | 9.7, 2 of 14 over | 19.7, 0 of 3 over | 67.6, 4 of 4 over | 16.8 |
| L3 | 10.1, 0 of 14 over | 21.1, 0 of 3 over | 20.6, 1 of 4 over | 10.6 |
| L1, reference | 7.7, 2 of 14 over | 16.5, 0 of 3 over | 69.2, 4 of 4 over | 20.5 |

In the first three rows, totals 11–13 and 17–20 were held out. The reference
trained on every total.

Mean cost at the hardest totals:

| Run | 14 | 15 | 16 | 17 | 18 | 19 | 20 |
|---|---|---|---|---|---|---|---|
| L1 | 43 | 63 | 85 | 111 | 116 | 105 | 204 |
| L2 | 23 | 30 | 35 | 69 | 42 | 55 | 104 |
| L3 | 23 | 24 | 24 | 20 | 34 | 15 | 13 |
| L1, reference | 23 | 30 | 32 | 57 | 53 | 53 | 115 |

- **H1 fits.** L1 costs 43 to 85 at totals 14 to 16, which it trained on,
  while its easy totals are far under the budget.
- **H3 fits.** L3 is under the budget at every trained total and sits at 23 to
  24 where the budget binds.
- **H2 does not fit yet.** L2 costs 30 and 35 at totals 15 and 16, which it
  trained on. Conditioning the constraint more than halves L1's cost there,
  but the blind policy is not cautious enough.
- **Interpolation:** L2 and L3 stay under the budget at all three held-out
  totals.
- **Extrapolation:** only L3 mostly stays under, and it pays with return,
  which falls from 26.6 on trained totals to 10.6.
- **Much of L1's excess at 17–20 is not a failure to generalize.** Its
  reference trained on those totals and still costs 53 to 115 there.
- No episode left the arena.

Curves per hazard total: <https://claude.ai/artifact/7vBvEwBQaojELVDjXhv94v>.
Score tables: `runs/scores/goal_point_hazard_total_500M/`.

**October 2026.** Four findings, one seed each, measured before the contact
solver was replaced on 2026-11-11 (`known_defects.md`). Their costs are not
comparable with the tables above.

| Finding | Source |
|---|---|
| One multiplier holds the budget only on average: 33 to 78 on level 3 against 25 | `experiments/2026-10-15_goal_point_learners_500M.md` |
| $\lambda(\omega)$ with a told policy holds it there: 16 to 25 | same |
| Mean cost was carried by rare, very costly episodes, caused by a lidar that could not tell passable from solid hazards | `experiments/2026-10-19_goal_point_cost_tail_500M.md` |
| A camera did not help and made evaluations swing | `decisions/2026-10-18_vision_set_aside.md` |

## References

Only what §2 and §3 cite. Links to the sources, and how each citation was
checked, are in the related work draft
(<https://claude.ai/artifact/WEYNxELy5183HDA9zaoLQS>; the matrix behind it is
<https://claude.ai/artifact/9KUxV86nGkimyARfmx7xh6>). Kirk et al. (2023) and
Jiang et al. (2021) are not in the draft.

- Altman (1999). Constrained Markov Decision Processes.
- As et al. (2025). SPiDR: A Simple Approach for Zero-Shot Safety in Sim-to-Real Transfer. NeurIPS.
- Bohez et al. (2019). Value constrained model-free continuous control. arXiv 1902.04623.
- Calvo-Fullana et al. (2021). State Augmented Constrained Reinforcement Learning. arXiv 2102.11941.
- Chane-Sane et al. (2024). CaT: Constraints as Terminations for Legged Locomotion Reinforcement Learning. IROS.
- Guan et al. (2024). Cost-aware Offline Safe Meta Reinforcement Learning with Robust In-Distribution Online Task Adaptation. AAMAS.
- Hallak, Di Castro & Mannor (2015). Contextual Markov Decision Processes. arXiv 1502.02259.
- Han & Feng (2026). Transferable Reinforcement Learning via Probabilistic Latent Embeddings and Dynamic Policy Adaptation for Sim-to-Real Deployment. arXiv 2605.27659.
- Ji et al. (2023). Safety-Gymnasium: A Unified Safe Reinforcement Learning Benchmark. NeurIPS Datasets and Benchmarks.
- Jiang, Grefenstette & Rocktäschel (2021). Prioritized Level Replay. ICML.
- Khattar et al. (2023). A CMDP-within-online framework for Meta-Safe Reinforcement Learning. ICLR.
- Kirk, Zhang, Grefenstette & Rocktäschel (2023). A Survey of Zero-shot Generalisation in Deep Reinforcement Learning. JAIR.
- Koprulu, Simão, Jansen & Topcu (2025). Safety-Prioritizing Curricula for Constrained Reinforcement Learning. ICLR.
- Liu et al. (2023). Constrained Decision Transformer for Offline Safe Reinforcement Learning. ICML.
- Ma et al. (2021a). Feasible Actor-Critic: Constrained Reinforcement Learning for Ensuring Statewise Safety. arXiv 2105.10682.
- Mankowitz et al. (2020). Robust Constrained Reinforcement Learning for Continuous Control with Model Misspecification. arXiv 2010.10644.
- Moeini et al. (2025). Safe In-Context Reinforcement Learning. arXiv 2509.25582.
- Narasimhan et al. (2020). Approximate Heavily-Constrained Learning with Lagrange Multiplier Models. NeurIPS.
- Queeney & Benosman (2023). Risk-Averse Model Uncertainty for Distributionally Robust Safe Reinforcement Learning. NeurIPS.
- Ray, Achiam & Amodei (2019). Benchmarking Safe Exploration in Deep Reinforcement Learning.
- Russel, Benosman & Van Baar (2020). Robust Constrained-MDPs. arXiv 2010.04870.
- Spoor et al. (2025). An Empirical Study of Lagrangian Methods in Safe Reinforcement Learning. arXiv 2510.17564.
- Stooke, Achiam & Abbeel (2020). Responsive Safety in Reinforcement Learning by PID Lagrangian Methods. ICML.
- Tessler, Mankowitz & Mannor (2019). Reward Constrained Policy Optimization. ICLR.
- Tetteh & Fleming (2026). Safe on Average, Unsafe in the Tail. arXiv 2610.09508.
- Usmanova, As, Kamgarpour & Krause (2024). Log Barriers for Safe Black-box Optimization with Application to Safe Reinforcement Learning. JMLR.
- Uslu et al. (2025). Fast State-Augmented Learning for Wireless Resource Allocation with Dual Variable Regression. arXiv 2506.18748.
- Xu et al. (2022). Trustworthy Reinforcement Learning Against Intrinsic Vulnerabilities: Robustness, Safety, and Generalizability. arXiv 2209.08025.
- Xu & Zhu (2025). Efficient Safe Meta-Reinforcement Learning: Provable Near-Optimality and Anytime Safety. NeurIPS.
- Yao et al. (2023). Constraint-Conditioned Policy Optimization for Versatile Safe Reinforcement Learning. NeurIPS.
- Yu, Ma, Li & Chen (2022). Reachability Constrained Reinforcement Learning. ICML.
- Zhang, Yang, Lyu, Li & Zhang (2026). Augmented Lagrangian Multiplier Network for State-wise Safety in Reinforcement Learning. arXiv 2605.00667.
