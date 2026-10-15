# Watching what the policy did: requirements for the episode review

2026-10-15, requirements for review before anything is built. Branch
`episode-review`; to be merged into `contextual-learner` for testing.

Every result so far is a curve: a mean over 128 episodes, once per evaluation.
Curves show *that* something happened and never *how*. Twice already we read
a curve wrong until we found the mechanism by hand; this makes the mechanism
visible on demand. The data comes from the evaluations every run already
performs; nothing new is simulated. This document says why, then what a
reviewer must be able to do, what each episode carries, and what we expect the
first viewing to show.

## 0. Why

**The question of the thesis is about mechanism, and the curves cannot show
one.** The study asks how a curriculum's coverage and non-stationarity change
what a constrained policy learns. Every claim of that kind — "λ priced the
path out", "the policy never found a safe route", "the curriculum fed harder
contexts" — is a claim about what happens *inside episodes*, and we have been
making such claims from aggregates. On 2026-10-12 we attributed the `plr:cost`
collapse to the curriculum moving when it had not; on 2026-10-14 the smoke run
ended under budget on level 3 and the honest answer to "did it learn to be
safe or learn to stand still" was "the return is 0.5, so probably the latter".
Both times the evidence we needed was an episode and the learner's view of it.

**The decision in front of us depends on it.** The 500 M arms cost a GPU-day
each. Whether to run them at budget 25 turns on one fact: on level 3 around
round 60–80, did the policy have a path to the goals that λ then made
unprofitable, or did it never have one? If the first, the budget binds and the
result is the method's. If the second, level 3 is infeasible for this policy at
this budget and running five arms there measures the budget, not the
curriculum. Twenty minutes of watching settles what no number in the logs can.

**The learner's beliefs are as much the result as the behaviour.** A
constrained learner acts on three estimates — the value of a state in reward,
in cost, and the exchange rate between them — and the Lagrangian failure mode
we keep meeting is a disagreement between those estimates and what happens.
Seeing V, V_c, λ and the advantages *along a trajectory* is how one tells a
critic that is wrong from a multiplier that is too high from a policy that has
stopped exploring. The dashboard shows each as one number per round; the
disagreement lives per step.

**It is cheap and it compounds.** The episodes are already simulated, 128 per
target per evaluation; keeping four of them is a few megabytes per run and no
extra compute. Every experiment from now on carries its own evidence, so the
next "why did it do that" is answered by opening a record instead of by a day
of scripting. The write-ups gain figures that show a behaviour, not a mean.

## 1. What a reviewer can do

**Find an episode in three choices.** Pick an arm (`uniform per_context told`),
a moment in training (one of the run's evaluations, labelled by environment
step), and a target (deployment, uniform, …). A list shows the recorded
episodes of that moment with their return, cost and context, the same numbers
the Verdict panel averages. One click opens one.

**Watch it.** The scene replays with play and pause, at normal speed or slowed.
A timeline under the scene shows where in the episode you are; drag it to any
step, or step forward and back one step at a time with the arrow keys. The
step counter, return so far and cost so far are always visible.

**See what the learner thought at every step.** Next to the scene, traces over
the same time axis: reward and cost per step and cumulated; the reward critic's
and the cost critic's value of the current state; the three advantages, reward,
cost and their Lagrangian combination; the action. A cursor on every trace
follows the timeline, and clicking a trace moves the scene there. The context
ω and the multiplier λ the learner applied to it sit in the header.

**Compare two moments.** Pick a second evaluation and the viewer shows two
episodes of the same target side by side, scenes and traces on one shared
timeline. Round 60 against round 160 on level 3 is the first thing to look at.

**Nothing else.** No editing, no re-running, no training data. One command
starts it against a W&B group; it reads records and never computes anything
about them.

## 2. What one recorded episode carries

One row per step, all taken at the moment of the evaluation:

| | what it is | why it is there |
|---|---|---|
| scene | every body's pose | to replay it |
| observation, context | what the policy saw; which context the episode ran in | to know what it was told |
| action | what the policy did, as the evaluation ran it | behaviour |
| reward, cost | what happened | the two things being traded |
| V, V_c | the two critics' value of the state | what the learner believed each state was worth |
| λ | the multiplier applied to this context | the exchange rate at that moment |
| A_r, A_c, A_r − λA_c | the advantages, by the loss's own code on this episode | where estimate and outcome disagreed, and which way the learner would push |

**Straight from the learner's code, not reconstructed.** The values and
advantages are produced by the same functions the training loss uses, with the
same constants, applied to the recorded episode at recording time. One
difference is stated rather than hidden: the loss standardises advantages over
a training batch; an episode has no batch, so the record holds them raw and
the viewer labels them so.

**An evaluation episode, not a training one.** These episodes are the frozen
policy's, run for scoring; none of them was in a gradient. What the record
shows is the learner's view of its own behaviour at that moment. The data a
gradient was computed from is a different thing and is out of scope (§5).

**Size.** Four episodes per target per evaluation by default; with three
targets and twenty-one evaluations that is 252 episodes per 500 M run, about a
megabyte each. Stored with the run in W&B.

## 3. What we expect to see

Each row is a question the first viewing settles, and the two pictures that
would settle it.

| where to look | if the policy gave up a path it had | if it never had one |
|---|---|---|
| uniform arm, level 3, rounds 60–80 (return 16 → 11, cost 122 → 40) | the agent threads towards goals, brushing hazards; V_c rises along the path; A_c is positive where it moves | the agent crosses hazards or stalls near the start; V_c is high already at the first state |
| uniform arm, level 3, round 160 (return 0.5, cost 2) | still from step 0; V ≈ 0, V_c ≈ 0, and with λ ≈ 8 every move is a net loss in A_r − λA_c | the same picture: the two cases differ earlier, not here |
| the same checkpoint on uniform contexts | sparse layouts: the agent navigates, V high, λ small — the told policy keeps its behaviours apart | — |
| staged arm, level 3, round 120 | as the first row | as the first row |

The first row decides whether the 500 M arms run at budget 25: if the policy
had a path and λ priced it out, the budget binds and the result is the
method's; if it never had one, level 3 is infeasible for this policy at 100 M
and the budget is what to discuss.

## 4. What has to exist for this

In the order it is built; each line is testable on CPU with the tiny trainer.

| piece | one sentence |
|---|---|
| the record | a typed artifact holding §2, written once per recorded episode, read by the viewer |
| recording | the evaluator keeps the first K of its episodes whole; the agent hands it the function that applies its critics, its multiplier and the loss's advantage code to an episode, as it hands the trainer its learning signals today |
| the whole learner in the checkpoint | the cost critic and the multiplier state are saved next to the policy at every evaluation, so a checkpoint can later be run on any context |
| the viewer | a local page: the three choices, the scene with play / pause / step / scrub, the traces, the comparison |
| the switch | `--record_episodes K`, default 4 |

The merge point with `contextual-learner`: there the multiplier is a function
λ(ω) and the observation is `{sensors, context}`; the recorder takes both from
the agent, so only the agent-side function changes.

## 5. Out of scope

- **Training transitions**, the data a gradient was computed from. A different
  record, built only if the evaluation episodes leave a question open.
- **Running a checkpoint on a context chosen in the viewer.** The whole-learner
  checkpoint makes it possible from a script; a button for it waits until the
  first viewing shows which contexts are worth probing.
- **Tristan's browser player.** It has the physics and the task logic but no
  observation vector, so it cannot run our policies. Its renderer could replay
  our scenes later.
