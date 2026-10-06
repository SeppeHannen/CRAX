## P1 — reward–safety conflict
counts: REPORTED=2 DESIGNED=2 REVIEWED=0 ARGUED=0
verdict: supported
strongest:
- 2310.12567: Safety-Gymnasium deliberately makes velocity produce reward while velocity itself is constrained, and its benchmark analysis reports the resulting reward–cost trade-off.
- 2606.20376: CRAX explicitly designs its environment families so that higher reward typically incurs higher cost and stricter safety sacrifices reward.
notes: This is the clearest-supported candidate. Evidence includes both benchmark design rationale and experiments rather than only conceptual arguments.

## P2 — mostly infeasible
counts: REPORTED=0 DESIGNED=0 REVIEWED=0 ARGUED=0
verdict: no evidence
strongest:
- none: no qualifying evidence item found.
notes: 2404.10064 studies feasible and infeasible regions extensively, but it does not report that most contexts in an environment family have no budget-feasible policy. I therefore did not map generic state/policy infeasibility onto P2.

## P3 — hard tail
counts: REPORTED=0 DESIGNED=0 REVIEWED=0 ARGUED=0
verdict: no evidence
strongest:
- none: no qualifying evidence item found.
notes: I found curriculum and hard-task literature, but not a constrained-RL report matching the specific hypothesis that most contexts are trivially safe while the deployment distribution concentrates on a rare hard tail. CRAX has harder levels, but they are an explicit progression rather than a rare deployment tail.

## P4 — context-dependent safety
counts: REPORTED=1 DESIGNED=0 REVIEWED=0 ARGUED=1
verdict: supported
strongest:
- 2601.21094: Eight safe-RL baselines that are safe on training patients show a substantial safety-generalization gap on unseen patients with shifted physiological dynamics.
- 2209.06866: Robust constrained RL argues that training/deployment transition-model mismatch can degrade performance and cause constraint violations.
notes: The diabetes result is unusually direct family-level evidence: patient physiology is the context and the same trained policy becomes less safe under unseen contexts. The robust-CMDP paper is only supporting argument here because its full text was not accessed.

## P5 — dual-variable lag
counts: REPORTED=0 DESIGNED=0 REVIEWED=0 ARGUED=0
verdict: no evidence
strongest:
- none: no qualifying evidence item found.
notes: Several papers strongly support multiplier oscillation, sensitivity, and task/budget dependence, but none I could verify tested the specific sequence in P5: distribution shift makes a previously calibrated multiplier wrong and safety suffers until it re-converges. Those near-misses are kept as NEW:dual-instability and NEW:dual-sensitivity instead.

## P6 — uninformative sub-budget cost
counts: REPORTED=0 DESIGNED=0 REVIEWED=0 ARGUED=0
verdict: no evidence
strongest:
- none: no qualifying evidence item found.
notes: 2206.02675 is close but different: it reports sparse/delayed information about distance to violation and improves that signal through safety-state augmentation. I found no report isolating the exact P6 mechanism in which costs occur below budget yet provide no information about the constraint boundary.

## NEW:environment-stochasticity — stochastic or dynamic safety-relevant environment behavior
counts: REPORTED=2 DESIGNED=0 REVIEWED=0 ARGUED=0
verdict: supported
strongest:
- 2310.12567: Safety-Gymnasium reports pronounced algorithm-performance oscillations specifically in highly stochastic navigation tasks.
- 2305.13681: GUARD reports that dynamic collision-inducing ghosts increase training cost variance for constrained methods relative to static hazards.
notes: These two benchmarks independently point to environment stochasticity or moving hazards as a family property worth isolating. GUARD supplies a particularly clean static-hazard versus dynamic-ghost comparison.

## NEW:dual-instability — Lagrange-multiplier oscillation and tuning instability
counts: REPORTED=1 DESIGNED=0 REVIEWED=0 ARGUED=1
verdict: supported
strongest:
- 2007.03964: PID Lagrangian was motivated by observed oscillation and overshoot of ordinary Lagrangian updates causing constraint violations during training.
- 2310.12567: Safety-Gymnasium documents the practical trade-off where aggressive multiplier updates oscillate and slow updates impede convergence.
notes: This is strong evidence for generic dual dynamics being hard, but it is not evidence for P5's distribution-shift-induced lag. The Stooke full text could not be accessed, so its extraction row deliberately contains no quote.

## NEW:constraint-dominated-exploration — hard contexts fill exploration with violations and sparse progress
counts: REPORTED=1 DESIGNED=0 REVIEWED=0 ARGUED=0
verdict: supported
strongest:
- 2606.20376: In CRAX's hardest difficulty levels, direct training often fails to find a good policy because exploration becomes dominated by constraint violations and sparse progress.
notes: This is especially relevant to a TeachMyAgent-style situation because CRAX varies concrete family parameters such as hazard density and goal size across difficulty levels. Its curriculum experiment directly targets the resulting training problem.

## NEW:constraint-regime — constraint restrictiveness changes optimization behavior
counts: REPORTED=2 DESIGNED=0 REVIEWED=0 ARGUED=0
verdict: supported
strongest:
- 2510.17564: Within a fixed task, the best Lagrange-multiplier update mechanism changes with the cost limit.
- 2606.20376: Tightening safety bounds causes a substantially asymmetric loss of score across CRAX tasks.
notes: The evidence says that a nominally identical environment can become qualitatively different for constrained training as the feasible performance region is tightened. This is not P2: the settings remain trainable rather than being shown infeasible.

## NEW:sparse-delayed-cost — safety feedback is sparse or arrives only at violation
counts: REPORTED=1 DESIGNED=0 REVIEWED=0 ARGUED=0
verdict: supported
strongest:
- 2206.02675: Experiments show that ordinary episodic constraint feedback only informs the learner after violation, whereas an augmented safety state continuously exposes distance to violation.
notes: This is the closest empirical neighbor to P6 but is materially different, so it is kept separate. A useful situation would vary how early the cost signal becomes informative before the safety boundary.

## NEW:dual-sensitivity — the appropriate multiplier depends strongly on task and budget
counts: REPORTED=1 DESIGNED=0 REVIEWED=0 ARGUED=0
verdict: supported
strongest:
- 2510.17564: Empirical λ-profiles across eight Safety-Gymnasium tasks show that the optimal multiplier is highly task-dependent and cost-limit-dependent.
notes: This is a static sensitivity result, not a lag-after-shift result. It nevertheless gives a concrete prerequisite for P5: changing context or constraint regime can change the multiplier that would be appropriate.

## NEW:high-dimensional-control — action-space and workspace complexity slows constrained learning
counts: REPORTED=1 DESIGNED=0 REVIEWED=0 ARGUED=0
verdict: supported
strongest:
- 2305.13681: GUARD reports slower convergence for tasks with high-dimensional robot action spaces and complex workspaces.
notes: This property is naturally expressible as an environment-family axis through agent morphology or degrees of freedom. It may be less specific to constrained RL than the other NEW properties, so a thesis situation would need an unconstrained control to establish the safety-specific component.

## Ambiguities
None; no evidence row was marked with `?`. Close matches were instead separated into NEW properties rather than forced onto P2, P5, or P6.

## Search log
date: 2026-10-06 Europe/Amsterdam
arXiv API status: NOT COMPLETED. Direct export.arxiv.org/API access returned an inaccessible/cache-miss result in the available tooling. I therefore could not reproduce the protocol's API hit lists or screen Q2's ranked first 60. I substituted web searches restricted to arxiv.org and direct arXiv abstract/full-text pages.
Semantic Scholar status: NOT COMPLETED. Semantic Scholar citation/API links exposed from arXiv returned cache-miss/inaccessible results. I substituted accessible full-text reference lists plus targeted web searches. Exhaustive backward/forward citation screening and formal snowball saturation therefore cannot be claimed.
Q1 executed: site:arxiv.org/abs "safe reinforcement learning" benchmark
Q2 executed: site:arxiv.org/abs ("constrained reinforcement learning" OR "safe reinforcement learning") (generalization OR "distribution shift" OR "unseen environments")
Q3 executed: site:arxiv.org/abs (Lagrangian OR "Lagrange multiplier") "reinforcement learning" (oscillation OR instability OR overshoot)
Q4 executed: site:arxiv.org/abs ("constrained MDP" OR CMDP OR "constrained reinforcement learning") (infeasible OR "strictly feasible" OR feasibility)
Q5 executed: site:arxiv.org/abs "safe reinforcement learning" "trade-off" cost reward
Q6 executed: site:arxiv.org/abs (contextual OR "domain randomization") ("constrained reinforcement learning" OR "safe reinforcement learning" OR CMDP)
QA1 executed: site:arxiv.org/abs curriculum ("constrained reinforcement learning" OR "safe reinforcement learning" OR CMDP OR "constrained Markov")
QA2 executed: site:arxiv.org/abs ("unsupervised environment design" OR "prioritized level replay" OR "environment design") (safe OR safety OR constraint)
additional executed: site:arxiv.org/abs/ "safe reinforcement learning" benchmark
additional executed: site:arxiv.org/abs/ "constrained reinforcement learning" generalization
additional executed: site:arxiv.org/abs/ "safe reinforcement learning" "distribution shift"
additional executed: site:arxiv.org/abs/ "safe reinforcement learning" "unseen environments"
additional executed: site:arxiv.org/abs "contextual" "safe reinforcement learning"
additional executed: site:arxiv.org/abs "domain randomization" "safe reinforcement learning"
additional executed: site:arxiv.org/abs "distribution shift" "safe reinforcement learning" constrained
additional executed: site:arxiv.org/abs "generalization" "constrained reinforcement learning" safety
additional executed: site:arxiv.org/abs constrained reinforcement learning infeasible constraint feasibility experiments
additional executed: site:arxiv.org/abs CMDP infeasible safe reinforcement learning feasibility
additional executed: site:arxiv.org/abs Lagrangian safe reinforcement learning oscillation PID multiplier instability
additional executed: site:arxiv.org/abs safe reinforcement learning remaining budget state augmentation cost signal
additional executed: Klink "Safe Curriculum Generation"
additional executed: site:arxiv.org Klink curriculum safe reinforcement learning
additional executed: site:openreview.net Klink "Safe Curriculum Generation"
additional executed: site:arxiv.org/abs "Lagrange multiplier" "distribution shift" reinforcement learning safety
additional executed: site:arxiv.org/abs Lagrangian nonstationary "safe reinforcement learning" multiplier
additional executed: site:arxiv.org/abs "dual variable" "safe reinforcement learning" nonstationary
additional executed: site:arxiv.org/abs "Lagrangian" "constraint violation" "distribution shift" RL
additional executed: site:arxiv.org/abs safe reinforcement learning "remaining budget" state augmentation
additional executed: site:arxiv.org/abs safe reinforcement learning "budget" "state augmentation" constraint
additional executed: site:arxiv.org/abs constrained reinforcement learning "cost budget" sparse signal
additional executed: site:arxiv.org/abs safe reinforcement learning sparse cost signal constraint boundary
additional executed: site:arxiv.org/abs curriculum "safe reinforcement learning"
additional executed: site:arxiv.org/abs curriculum "constrained reinforcement learning"
additional executed: site:arxiv.org/abs "domain randomization" "constrained reinforcement learning"
additional executed: site:arxiv.org/abs "environment design" "safe reinforcement learning"
additional executed: site:arxiv.org/abs/ "Safety-Prioritizing Curricula for Constrained Reinforcement Learning"
additional executed: site:arxiv.org/pdf/2202.06558 "remaining safety budget"
additional executed: site:arxiv.org/abs/2202.06558 "safety budget"
additional executed: "Saute RL" "safety budget" state augmentation
additional executed: "Saute RL" cost budget constraint boundary
snowball substitute: one targeted reference/citation pass was performed from included papers using accessible arXiv/ar5iv reference lists and title searches; it produced the eligible snowball rows shown above. Because Semantic Scholar forward-citation enumeration was unavailable, the protocol's stop criterion cannot be verified.
identifier rule: the id-less 2019 Safety Gym technical report was used only as a conceptual search seed and omitted from CSV rows. The searched Safe Curriculum Generation / Safety-Prioritizing Curricula item was also omitted because I could not verify an allowed arXiv id or DOI.
