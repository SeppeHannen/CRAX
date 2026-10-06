```csv
# screening.csv
paper_id,title,year,source,stage_reached,decision,criterion
2310.12567,"Safety-Gymnasium: A Unified Safe Reinforcement Learning Benchmark",2023,"seed:2310.12567",2,include,"contains P1 and stochasticity evidence"
2305.09304,"OmniSafe: An Infrastructure for Accelerating Safe Reinforcement Learning Research",2023,"seed:2305.09304",2,exclude,"no property evidence"
2606.20376,"CRAX: Fast Safe Reinforcement Learning Benchmarking",2026,"seed:2606.20376",2,include,"reports hard-level failure and reward-safety tradeoff"
2007.03964,"Responsive Safety in Reinforcement Learning by PID Lagrangian Methods",2020,"seed:2007.03964",1,include,"abstract names Lagrangian oscillation failure"
2402.02025,"A Survey of Constraint Formulations in Safe Reinforcement Learning",2024,"seed:2402.02025",1,exclude,"review abstract lacks family-specific failure"
2006.12136,"Safe Reinforcement Learning via Curriculum Induction",2020,"seed:2006.12136",1,exclude,"stage 2 unavailable in available tooling"
2103.09815,"TeachMyAgent: a Benchmark for Automatic Curriculum Learning in Deep RL",2021,"seed:2103.09815",1,exclude,"not constrained or safe RL"
2110.02102,"CARL: A Benchmark for Contextual and Adaptive Reinforcement Learning",2021,"seed:2110.02102",1,exclude,"not constrained or safe RL"
2202.06558,"Saute RL: Almost Surely Safe Reinforcement Learning Using State Augmentation",2022,"seed:2202.06558",1,exclude,"stage 2 unavailable in available tooling"
2012.02096,"Emergent Complexity and Zero-shot Transfer via Unsupervised Environment Design",2020,"seed:2012.02096",1,exclude,"not constrained or safe RL"
2010.03934,"Prioritized Level Replay",2021,"seed:2010.03934",1,exclude,"not constrained or safe RL"
2203.01302,"Evolving Curricula with Regret-Based Environment Design",2022,"seed:2203.01302",1,exclude,"not constrained or safe RL"
2405.01677,"Balance Reward and Safety Optimization for Safe Reinforcement Learning: A Perspective of Gradient Manipulation",2024,Q5,2,include,"reports reward-safety failure across multiple tasks"
2404.10064,"The Feasibility Theory of Constrained Reinforcement Learning: A Tutorial Study",2024,Q4,2,exclude,"feasibility theory lacks family-distribution evidence"
2601.21094,"Safety Generalization Under Distribution Shift in Safe Reinforcement Learning: A Diabetes Testbed",2026,Q2,2,include,"reports safety failure under physiological shift"
2206.02675,"Effects of Safety State Augmentation on Safe Exploration",2022,"snowball:2202.06558",2,include,"reports sparse delayed safety signal"
2510.17564,"Towards a Practical Understanding of Lagrangian Methods in Safe Reinforcement Learning",2025,Q3,2,include,"reports task-dependent multiplier behavior"
2305.13681,"GUARD: A Safe Reinforcement Learning Benchmark",2023,Q1,2,include,"reports family factors affecting convergence and variance"
2205.10330,"A Review of Safe Reinforcement Learning: Methods, Theory and Applications",2022,Q1,1,exclude,"review abstract lacks family-specific failure"
2306.09303,"Datasets and Benchmarks for Offline Safe Reinforcement Learning",2023,Q1,1,exclude,"offline-only; no online family training evidence"
2008.06626,"Safe Reinforcement Learning in Constrained Markov Decision Processes",2020,Q4,1,exclude,"no environment-family variation named in abstract"
2403.15928,"Safe Reinforcement Learning for Constrained Markov Decision Processes with Stochastic Stopping Time",2024,Q4,1,exclude,"no environment-family variation named in abstract"
2209.06866,"Robust Constrained Reinforcement Learning",2022,Q2,1,include,"abstract argues train-test model mismatch failure"
2305.12967,"Lagrangian-based online safe reinforcement learning for state-constrained systems",2023,Q3,1,exclude,"single state-constrained setup; no family variation"
2312.00342,"Efficient Off-Policy Safe Reinforcement Learning Using Trust Region Conditional Value at Risk",2023,Q2,1,exclude,"distribution shift is replay-buffer not environment family"
2605.18842,"Safe Continual Reinforcement Learning under Nonstationarity via Adaptive Safety Constraints",2026,Q2,1,exclude,"nonstationarity not parameterized as environment family"
2002.10126,"Safe reinforcement learning for probabilistic reachability and safety specifications: A Lyapunov-based approach",2020,"snowball:2206.02675",1,exclude,"single safety specification; no family variation"
2108.06266,"Safe Learning in Robotics: From Learning-Based Control to Safe Reinforcement Learning",2021,Q1,1,exclude,"review abstract lacks family-specific failure"
1705.10528,"Constrained Policy Optimization",2017,"snowball:2206.02675",1,exclude,"stage 2 unavailable in available tooling"
2002.06506,"First Order Constrained Optimization in Policy Space",2020,"snowball:2310.12567",1,exclude,"stage 2 unavailable in available tooling"
2011.05869,"CRPO: A New Approach for Safe Reinforcement Learning with Convergence Guarantee",2021,"snowball:2405.01677",1,exclude,"stage 2 unavailable in available tooling"
```

```csv
# evidence.csv
paper_id,title,year,venue,property,statement,environment,evidence_type,location,quote
2310.12567,"Safety-Gymnasium: A Unified Safe Reinforcement Learning Benchmark",2023,"NeurIPS 2023",P1,"Velocity tasks deliberately make faster and therefore higher-reward behavior costly once the velocity safety constraint binds.","Safety-Gymnasium velocity tasks",DESIGNED,"Sec. 4.1, Supported Constraints","Velocity-Constraint involves safety tasks using MuJoCo agents. In these tasks, agents aim for higher reward by moving faster, but they must also adhere to velocity constraints for safety."
2310.12567,"Safety-Gymnasium: A Unified Safe Reinforcement Learning Benchmark",2023,"NeurIPS 2023",P1,"Across benchmark experiments reward and episodic cost move in a trade-off, with unconstrained reward maximization producing risky behavior.","Safety-Gymnasium velocity tasks",REPORTED,"Sec. 6, Reward and Cost","Episodic reward and cost exhibit a trade-off relationship. Unconstrained algorithms aim to maximize reward through risky behaviors."
2310.12567,"Safety-Gymnasium: A Unified Safe Reinforcement Learning Benchmark",2023,"NeurIPS 2023","NEW:environment-stochasticity","Navigation tasks with greater stochasticity exhibit pronounced oscillations in safe-RL performance.","Safety-Gymnasium navigation tasks",REPORTED,"Sec. 6, Randomness and Oscillation","However, pronounced oscillations are present in navigation tasks characterized by high stochasticity."
2310.12567,"Safety-Gymnasium: A Unified Safe Reinforcement Learning Benchmark",2023,"NeurIPS 2023","NEW:dual-instability","Lagrangian multiplier learning rates that are too large cause oscillation while rates that are too small slow convergence.","Safety-Gymnasium SafePO Lagrangian methods",ARGUED,"Appendix A.1, Lagrangian Multiplier Settings","A high learning rate induces excessive oscillations, impedes convergence speed, and hinders the algorithm’s ability to attain the desired solution. Conversely, a low learning rate slows down convergence and adversely affects training."
2606.20376,"CRAX: Fast Safe Reinforcement Learning Benchmarking",2026,"arXiv",P1,"CRAX deliberately constructs tasks in which increased performance generally incurs more cost and safety requires sacrificing reward.","CRAX environment suites",DESIGNED,"Sec. 1","Each task defines reward and cost signals, inducing a trade-off between performance and safety: achieving high reward typically requires incurring higher cost, while satisfying safety constraints necessitates sacrificing some reward."
2606.20376,"CRAX: Fast Safe Reinforcement Learning Benchmarking",2026,"arXiv","NEW:constraint-dominated-exploration","Direct training on the hardest contexts often fails to find good policies because exploration is dominated by violations and sparse progress.","CRAX Level 3 tasks",REPORTED,"Sec. 5.2, Fig. 4","when trained directly on the hardest difficulty level, agents often struggle to discover a good policy, as exploration becomes dominated by constraint violations and sparse progress."
2606.20376,"CRAX: Fast Safe Reinforcement Learning Benchmarking",2026,"arXiv","NEW:constraint-regime","Making the safety bound stricter produces an asymmetric and substantially larger performance loss.","CRAX tasks with varied safety bounds",REPORTED,"Sec. 5.4, Fig. 6","Tightening the bound leads to a substantially larger drop in score than the gains obtained by increasing it by the same amount."
2007.03964,"Responsive Safety in Reinforcement Learning by PID Lagrangian Methods",2020,"ICML 2020","NEW:dual-instability","Ordinary Lagrangian learning exhibits oscillation and overshoot that causes constraint violations during training.","Safety Gym",REPORTED,"full text not accessed",""
2405.01677,"Balance Reward and Safety Optimization for Safe Reinforcement Learning: A Perspective of Gradient Manipulation",2024,"arXiv",P1,"On velocity-constrained OmniSafe tasks established constrained baselines struggle to preserve safety while pursuing reward.","OmniSafe SafetyHopperVelocity-v1 and SafetyAntVelocity-v1",REPORTED,"Sec. 5.2, Fig. 5","SOTA baselines such as CUP and PPOLag struggle to ensure safety, and their reward performance is worse than our algorithm."
2601.21094,"Safety Generalization Under Distribution Shift in Safe Reinforcement Learning: A Diabetes Testbed",2026,"ICML 2026",P4,"Policies satisfying training-patient safety constraints degrade and violate deployment criteria on patients with different physiological dynamics.","Unified diabetes simulator, unseen patient contexts",REPORTED,"Sec. 6.1, Table 1","policies that achieve high TIR and low Risk Index on the training patients show substantial degradation when deployed on physiologically distinct patients."
2209.06866,"Robust Constrained Reinforcement Learning",2022,"arXiv",P4,"Mismatch between training and deployment transition dynamics can cause performance degradation and constraint violation, motivating robust constrained policies.","CMDPs with uncertain transition dynamics",ARGUED,"full text not accessed",""
2206.02675,"Effects of Safety State Augmentation on Safe Exploration",2022,"NeurIPS 2022","NEW:sparse-delayed-cost","With ordinary episodic constraint feedback the learner only learns about violation after an episode, while safety-state augmentation supplies distance-to-violation information continuously.","Pendulum swing-up and point-goal safe RL",REPORTED,"Sec. 3.3, discussion of Fig. 4","While an algorithm without the safety state will receive information on constraint violation after an episode, the safety state would constantly inform the algorithm of the distance toward a violation."
2510.17564,"Towards a Practical Understanding of Lagrangian Methods in Safe Reinforcement Learning",2025,"arXiv","NEW:constraint-regime","Within the same Safety-Gymnasium task, changing the cost limit can change which Lagrange-multiplier update method is best.","Eight Safety-Gymnasium navigation and velocity tasks",REPORTED,"Sec. 6, Table 1","there is no single method that consistently performs best within a task, but rather depends on the chosen cost limit."
2510.17564,"Towards a Practical Understanding of Lagrangian Methods in Safe Reinforcement Learning",2025,"arXiv","NEW:dual-sensitivity","The multiplier value appropriate for constraint satisfaction varies substantially with both the task and the selected cost limit.","Eight Safety-Gymnasium navigation and velocity tasks",REPORTED,"Appendix A.1, Fig. A.1","This indicates that the optimal value λ* is highly task-dependent and dependent on the value of the cost limit."
2305.13681,"GUARD: A Safe Reinforcement Learning Benchmark",2023,"TMLR","NEW:environment-stochasticity","Replacing static hazards with dynamic collision-inducing ghosts raises training variance for multiple constrained methods.","GUARD Goal_Point_8Ghosts versus hazard tasks",REPORTED,"Sec. 6, Fig. 8","These tasks exhibit higher variance during the training process due to the collision-inducing behaviors of the dynamic ghosts."
2305.13681,"GUARD: A Safe Reinforcement Learning Benchmark",2023,"TMLR","NEW:high-dimensional-control","Contexts with higher-dimensional robot action spaces and more complex workspaces converge more slowly.","GUARD Walker and Arm task variants",REPORTED,"Sec. 6, Fig. 8","tasks that involve high-dimensional robot action spaces and complex workspaces suffer from slower convergence due to the increased complexity of the learning problem."
```

```csv
# by_product_A.csv
paper_id,title,year,interaction
2006.12136,"Safe Reinforcement Learning via Curriculum Induction",2020,"An automatic instructor learns which reset controller to deploy as a curriculum so exploration remains safe while the student learns the task."
2606.20376,"CRAX: Fast Safe Reinforcement Learning Benchmarking",2026,"Safe RL agents are trained sequentially through increasing environment difficulty levels with parameters carried between levels while safety constraints remain active."
```

```markdown
<!-- synthesis.md -->
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
```

The main surprise is how sharply the evidence separates from the hypotheses. **P1 and P4 survive; P2, P3, P5, and P6 do not in their stated forms.** Instead, practitioners report stochastic/dynamic hazards, violation-dominated exploration on hard levels, constraint-regime sensitivity, sparse/delayed safety information, and fragile dual dynamics. Safety-Gymnasium explicitly reports both reward–cost conflict and oscillations in stochastic navigation, while GUARD independently reports higher variance from dynamic ghosts and slower convergence in higher-dimensional contexts. citeturn31view3turn34view0 CRAX is particularly thesis-relevant: its hardest contexts make exploration dominated by violations and sparse progress, and its curriculum helps some families but not Safe Goal. citeturn25view0turn25view1 The diabetes study gives unusually clean evidence for P4 under genuinely unseen dynamics. citeturn29view3 The multiplier literature supports instability and task/budget sensitivity, but not the specific distribution-shift/re-convergence mechanism posited by P5. citeturn26view0turn29view1

The largest limitation is procedural rather than evidential: the attached protocol requires ranked arXiv-API screening and Semantic Scholar citation enumeration, but those endpoints were inaccessible in this environment; the substitutions and incomplete stages are recorded rather than silently treated as completed. fileciteturn0file0 With another hour and those APIs available, I would concentrate almost entirely on P2/P3/P5/P6 and resolve the id-less curriculum papers before expanding the already-supported properties.