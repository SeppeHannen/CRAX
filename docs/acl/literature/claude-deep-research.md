```csv
# screening.csv
paper_id,title,year,source,stage_reached,decision,criterion
2007.03964,Responsive Safety in Reinforcement Learning by PID Lagrangian Methods,2020,seed:2007.03964,2,include,abstract names oscillation/overshoot failure; evidence in full text
2606.20376,CRAX: Fast Safe Reinforcement Learning Benchmarking,2026,seed:2606.20376,2,include,benchmark for safe RL; multi-level; evidence found
2310.12567,Safety-Gymnasium: A Unified Safe Reinforcement Learning Benchmark,2023,seed:2310.12567,2,include,benchmark for safe RL; oscillation reported (full text not fetched)
2305.09304,OmniSafe: An Infrastructure for Accelerating Safe Reinforcement Learning Research,2023,seed:2305.09304,2,exclude,no property evidence (full text not accessed)
2402.02025,A Survey of Constraint Formulations in Safe Reinforcement Learning,2024,seed:2402.02025,2,exclude,no property evidence (abstract only accessed)
2006.12136,Safe Reinforcement Learning via Curriculum Induction,2020,seed:2006.12136,2,include,curriculum + constrained learner (by-product A)
2202.06558,Saute RL: Almost Surely Safe Reinforcement Learning Using State Augmentation,2022,seed:2202.06558,2,exclude,no property evidence (abstract only accessed)
2103.09815,TeachMyAgent: a Benchmark for Automatic Curriculum Learning in Deep RL,2021,seed:2103.09815,1,exclude,not constrained/safe RL (seed for structure only)
2110.02102,Contextualize Me -- The Case for Context in Reinforcement Learning,2023,seed:2110.02102,1,exclude,not constrained/safe RL (not opened)
2012.02096,Emergent Complexity and Zero-shot Transfer via Unsupervised Environment Design,2020,seed:2012.02096,1,exclude,not constrained/safe RL (not opened)
2010.03934,Prioritized Level Replay,2021,seed:2010.03934,1,exclude,not constrained/safe RL (not opened)
2203.01302,Evolving Curricula with Regret-Based Environment Design,2022,seed:2203.01302,1,exclude,not constrained/safe RL (not opened)
2206.02675,Effects of Safety State Augmentation on Safe Exploration,2022,snowball:2202.06558,2,include,names sparse-cost violation failure; budget curriculum
2503.08241,HASARD: A Benchmark for Vision-Based Safe Reinforcement Learning in Embodied Agents,2025,snowball:2606.20376,2,include,safe RL benchmark with levels; design rationale (snippet only)\[1\]
2609.15315,Evaluation Metrics for Safe Reinforcement Learning,2026,snowball:2606.20376,2,include,large comparison on CRAX; reports violation under eval shift
2610.05569,Factoriax: A GPU-Accelerated Factorio-Style Simulator for Reinforcement Learning,2026,snowball:2606.20376,1,exclude,not safe/constrained RL
2609.37070,Predictive Safety Curricula for Robust Legged Locomotion,2026,QA1,1,include,curriculum + safety cost (by-product A); full text unreadable
2511.02690,Curriculum Design for Trajectory-Constrained Agent: Compressing Chain-of-Thought Tokens in LLMs,2025,QA1,1,exclude,constraint is token length not safety cost\[2\]
2502.15119,CurricuVLM: Towards Safe Autonomous Driving via Personalized Safety-Critical Curriculum Learning with Vision-Language Models,2025,QA1,1,include,safety curriculum for RL driving (by-product A; learner unverified)
2309.14091,On the Benefit of Optimal Transport for Curriculum Reinforcement Learning,2023,QA1,1,exclude,no constrained/safe learner
2412.04426,Towards Fast Safe Online Reinforcement Learning via Policy Finetuning,2024,Q3,2,include,reports Lagrange multiplier mismatch after distribution change\[3\]
2512.23770,SB-TRPO: Towards Safe Reinforcement Learning with Hard Constraints,2025,Q3,2,include,names over-conservatism/multiplier ratchet (snippet only)
2301.10339,AutoCost: Evolving Intrinsic Cost for Zero-violation Reinforcement Learning,2023,Q3,2,include,reports oscillation around cost limit (snippet only)
2601.18142,Enhance the Safety in Reinforcement Learning by ADRC Lagrangian Methods,2026,snowball:2305.09304,2,exclude,no property evidence (full text not accessed)
2510.17564,Towards a Practical Understanding of Lagrangian Methods in Safe Reinforcement Learning,2025,snowball:2305.09304,2,include,large comparison across 8 tasks and cost limits\[4\]
2405.20860,Enhancing Efficiency of Safe Reinforcement Learning via Sample Manipulation,2024,snowball:2305.09304,1,exclude,no named failure in abstract/snippet
2112.12228,Direct Behavior Specification via Constrained Reinforcement Learning,2022,Q4,2,include,reports multiplier blow-up under unsatisfiable constraint
2105.10682,Feasible Actor-Critic: Constrained Reinforcement Learning for Ensuring Statewise Safety,2021,Q4,2,include,theory naming infeasibility (P2 rule)
2205.07536,Reachability Constrained Reinforcement Learning,2022,Q4,2,include,theory naming infeasible states (P2 rule)
1907.01475,Generalizing from a few environments in safety-critical reinforcement learning,2019,Q2,2,include,evaluates safety across environments; reports failure on unseen
2601.21094,Safety Generalization Under Distribution Shift in Safe Reinforcement Learning: A Diabetes Testbed,2026,Q2,2,include,reports violation under shift (abstract)
2506.11033,Adaptive Shielding for Safe Reinforcement Learning under Hidden-Parameter Dynamics Shifts,2025,Q2,2,include,constrained learners across hidden-parameter family; failure reported\[5\]
2605.18842,Safe Continual Reinforcement Learning under Nonstationarity via Adaptive Safety Constraints,2026,Q2,2,include,constrained RL across nonstationary contexts\[6\]
2509.25582,Safe In-Context Reinforcement Learning,2025,Q2,2,exclude,no property evidence (full text not accessed)
2205.14691,On the Robustness of Safe Reinforcement Learning under Observational Perturbations,2022,QA2,2,exclude,no property evidence (observation attack not context family)
2509.18648,SPiDR: A Simple Approach for Zero-Shot Safety in Sim-to-Real Transfer,2025,Q6,2,include,domain randomisation + constrained learner; DR violation reported
2301.12593,Risk-Averse Model Uncertainty for Distributionally Robust Safe Reinforcement Learning,2023,Q6,2,include,safe RL across perturbed dynamics incl. DR baseline (snippet)
2108.06266,Safe Learning in Robotics: From Learning-Based Control to Safe Reinforcement Learning,2021,Q6,1,exclude,review without named family failure in abstract
2609.15915,Safe Meta-Reinforcement Learning via Information Space Reachability,2026,snowball:2506.11033,2,include,argues context-dependent safety; meta-RL CMDP experiments\[7\]
2101.00531,Context-Aware Safe Reinforcement Learning for Non-Stationary Environments,2021,snowball:2506.11033,2,exclude,no property evidence beyond abstract claim
2606.11266,Seeing Before Colliding: Anticipatory Safe RL with Frozen Vision-Language Models,2026,snowball:2202.06558,2,include,names reactive-cost failure of Lagrangian (snippet only)
2608.30283,BCPPO: Bachelier-Inspired Constrained Proximal Policy Optimization for Tail-Risk-Aware Safe Reinforcement Learning,2026,Q5,2,include,argues expected-cost budget hides rare high-cost outcomes
```
```csv
# evidence.csv
paper_id,title,year,venue,property,statement,environment,evidence_type,location,quote
2606.20376,CRAX: Fast Safe Reinforcement Learning Benchmarking,2026,arXiv,P1,Every task is built so that high reward requires incurring cost.,CRAX all suites,DESIGNED,Sec. 1,"Each task defines reward and cost signals, inducing a trade-off between performance and safety: achieving high reward typically requires incurring higher cost, while satisfying safety constraints necessitates sacrificing some reward."\[8\]
2606.20376,CRAX: Fast Safe Reinforcement Learning Benchmarking,2026,arXiv,P1,Tighter budgets force PPO-Lagrangian to give up reward in every task.,CRAX Level tasks with d in {15;25;35},REPORTED,Sec. 5.4 Fig. 6,PPOLag increasingly sacrifices rewards to adhere to tighter safety bounds in all tasks.\[8\]
2606.20376,CRAX: Fast Safe Reinforcement Learning Benchmarking,2026,arXiv,NEW:hard-context-exploration,At the hardest level exploration is swamped by violations so no good policy is found.,CRAX Level 3 tasks,REPORTED,Sec. 5.2 (refers to Fig. 3),"when trained directly on the hardest difficulty level, agents often struggle to discover a good policy, as exploration becomes dominated by constraint violations and sparse progress."\[8\]
2606.20376,CRAX: Fast Safe Reinforcement Learning Benchmarking,2026,arXiv,NEW:over-conservatism,Direct training leaves budget unused; transfer helps; the opposite holds in Safe Goal.,CRAX Safe Pathway and Safe Height Level 3,REPORTED,Sec. 5.2 Fig. 4,"Here, direct training yields overly conservative policies that underutilize the available safety budget, while safety-transferred agents make better use of it."\[8\]
2606.20376,CRAX: Fast Safe Reinforcement Learning Benchmarking,2026,arXiv,NEW:over-conservatism,PPO-Lagrangian satisfies all bounds at Levels 2-3 but earns low reward.,CRAX 8 environments x 3 levels,REPORTED,Sec. 5.1 Table 3,"PPOLag achieves the highest safety percentage, being the only baseline to satisfy all cost bounds on Levels 2 and 3. However, it fails to reach high rewards."\[8\]
2606.20376,CRAX: Fast Safe Reinforcement Learning Benchmarking,2026,arXiv,NEW:budget-regime-sensitivity,The reward-budget trade-off is non-linear; tightening hurts more than loosening helps.,CRAX tasks with PPOLag,REPORTED,Sec. 5.4 Fig. 6,Tightening the bound leads to a substantially larger drop in score than the gains obtained by increasing it by the same amount.\[8\]
2606.20376,CRAX: Fast Safe Reinforcement Learning Benchmarking,2026,arXiv,NEW:constraint-type-dependence,Method success depends on the kind of constraint; Saute works only on some.,CRAX Reacher/Pathway vs others Level 1,REPORTED,Sec. 5.1 Fig. 3,PPOSauté satisfies the cost bound in Reacher and Pathway but otherwise behaves close to unconstrained PPO.\[8\]
2503.08241,HASARD: A Benchmark for Vision-Based Safe Reinforcement Learning in Embodied Agents,2025,ICLR,P1,Reward and cost are designed to be tightly coupled so cost reduction costs reward.,HASARD ViZDoom scenarios,DESIGNED,full text not accessed,\[1\]
2007.03964,Responsive Safety in Reinforcement Learning by PID Lagrangian Methods,2020,ICML,NEW:multiplier-oscillation,Gradient-ascent Lagrange multiplier oscillates with cost and overshoots the limit during training.,Safety Gym DoggoButton1 (cost limit 200) and others,REPORTED,Sec. 1 Fig. 1,"Cost overshoot and oscillations are in fact inherent to the learning dynamics (Platt & Barr, 1988; Wah et al., 2000), and we witnessed numerous problematic cases in our own experiments."\[9\]
2007.03964,Responsive Safety in Reinforcement Learning by PID Lagrangian Methods,2020,ICML,P5?,The multiplier is updated slowly so the penalty lags behind constraint-violating policies.,general (Lagrangian actor-critic),ARGUED,Sec. 2,"Convergence proofs have relied upon updating the multiplier more slowly than the policy parameters (Tessler et al., 2018; Paternain et al., 2019), implying many constraint-violating policy iterations may occur before the penalty comes into full effect."\[9\]
2007.03964,Responsive Safety in Reinforcement Learning by PID Lagrangian Methods,2020,ICML,P5?,Cost pressure from reward learning is nonstationary so λ needs to respond dynamically.,general,ARGUED,Sec. 5,"As the agent learns for rewards, the upward pressure on costs from reward-learning can change, requiring dynamic response."\[9\]
2310.12567,Safety-Gymnasium: A Unified Safe Reinforcement Learning Benchmark,2023,NeurIPS Datasets and Benchmarks,NEW:multiplier-oscillation,Lagrangian methods oscillate around the cost limit more than projection methods.,Safety-Gymnasium tasks,REPORTED,full text not accessed,\[10\]
2301.10339,AutoCost: Evolving Intrinsic Cost for Zero-violation Reinforcement Learning,2023,AAAI,NEW:multiplier-oscillation,"He, Zhao and Liu (AAAI 2023) report that CPO and PPO-Lagrangian stay around the threshold and fail to reach zero violation even with a zero cost limit.",Safety Gym tasks,REPORTED,Fig. 1,CPO and PPO-Lagrangian both fail to achieve zero violation even with a zero cost limit
2112.12228,Direct Behavior Specification via Constrained Reinforcement Learning,2022,ICML,P2,A constraint that is long to satisfy drives its multiplier very large and destabilises learning.,Arena / OpenWorld (Unity),ARGUED,Sec. 4.2,"a constraint that enforces a behavior which is long to discover can end up reaching very large multiplier values. It then leads to very large policy updates and destabilizes the learning dynamics."\[11\]
2112.12228,Direct Behavior Specification via Constrained Reinforcement Learning,2022,ICML,P2,With an impossible constraint the multiplier grows without bound and critic/performance collapse.,Arena (never touch the ground),REPORTED,Sec. 6.1 Fig. 3; full text not accessed for this section,\[11\]
2112.12228,Direct Behavior Specification via Constrained Reinforcement Learning,2022,ICML,NEW:feasibility-first-stall,Under many constraints the reward gets almost no weight while a feasible policy is sought.,Arena with 3+ constraints,ARGUED,Sec. 4.3,"leaves very little to no traction to improve on the main task while the process is looking for a feasible policy."\[11\]
2105.10682,Feasible Actor-Critic: Constrained Reinforcement Learning for Ensuring Statewise Safety,2021,arXiv,P2,For infeasible states the statewise multiplier diverges to infinity under primal-dual ascent.,theory + safe control tasks,ARGUED,full text not accessed,\[12\]
2205.07536,Reachability Constrained Reinforcement Learning,2022,ICML,P2,Constraints cannot be satisfied outside the feasible set so multipliers diverge without a cap.,theory + safe control tasks,ARGUED,full text not accessed,\[13\]
2412.04426,Towards Fast Safe Online Reinforcement Learning via Policy Finetuning,2024,arXiv (OpenReview-reviewed),P5,The multiplier learned offline is orders of magnitude off the value needed online.,OSRL\[3\] BallCircle,REPORTED,Sec. 3.2 Challenge II,"in BallCircle, the offline Lagrange multiplier value obtained using the BEAR-lag algorithm is approximately 1500, whereas during online finetuning, the SAC-lag requires a value of only about 0.65."
2412.04426,Towards Fast Safe Online Reinforcement Learning via Policy Finetuning,2024,arXiv (OpenReview-reviewed),P5,A mis-initialised multiplier after the shift causes violations or stagnation.,OSRL\[3\] BallCircle and CarRun,REPORTED,Sec. 3.2 Fig. 2,"Improper initialization can lead to extensive constraint violations or training stagnation, an issue we term as the Lagrange multiplier mismatch."\[3\]
2506.11033,Adaptive Shielding for Safe Reinforcement Learning under Hidden-Parameter Dynamics Shifts,2025,arXiv,P4,Across hidden-parameter contexts constrained baselines cannot balance safety and return even with parameters observed.,Safety-Gymnasium\[5\] with randomised gravity/damping/mass/friction,REPORTED,Sec. 5.2 RQ1 Fig. 1,"These results highlight the challenge of balancing safety and returns, even when hidden parameters are provided as inputs, in the presence of varying hidden parameters."
2506.11033,Adaptive Shielding for Safe Reinforcement Learning under Hidden-Parameter Dynamics Shifts,2025,arXiv,NEW:shift-violation,Out-of-distribution contexts break the safety-performance trade-off of baselines.,OOD\[5\] hidden parameters [0.15;0.3] and [1.7;2.5],REPORTED,Sec. 5.2 RQ2 Fig. 2,"The remaining algorithms exhibit high variance and sensitivity to dynamics shifts, failing to maintain a consistent safety-performance trade-off across environments."
2609.15915,Safe Meta-Reinforcement Learning via Information Space Reachability,2026,arXiv,P4,Whether a state is safe depends on the task posterior so safety is context-dependent.,meta-RL\[7\] CMDP tasks,ARGUED,Sec. I,"safety under task uncertainty is inherently belief-dependent: the same physical state may admit different safety guarantees under different task posteriors."
2605.18842,Safe Continual Reinforcement Learning under Nonstationarity via Adaptive Safety Constraints,2026,arXiv,NEW:stale-constraint,A constraint fixed for one context stops describing safety after the context changes.,highway-env\[14\] merge-v0,ARGUED,Sec. 1,"First, a policy may satisfy an outdated constraint that no longer captures safety in the changed environment."
2509.18648,SPiDR: A Simple Approach for Zero-Shot Safety in Sim-to-Real Transfer,2025,arXiv,NEW:shift-violation,Domain-randomised constrained training often violates constraints on the target dynamics.,RWRL / Safety Gym / RaceCar / real robots,REPORTED,Sec. 1 Fig. 3,"domain randomization lacks safety guarantees and often fails to satisfy the constraints in practice (cf. Queeney and Benosman, 2024, and Figure 3)."\[15\]
2509.18648,SPiDR: A Simple Approach for Zero-Shot Safety in Sim-to-Real Transfer,2025,arXiv,P3?,Averaging cost over randomised contexts can underestimate cost in the deployment context.,domain-randomised CMDP,ARGUED,Sec. 4.1,"since simulators only approximate the real world with limited precision, as well as due to averaging over dynamics, the costs in Equation (2) may underestimate the true costs in M*."\[15\]
2601.21094,Safety Generalization Under Distribution Shift in Safe Reinforcement Learning: A Diabetes Testbed,2026,arXiv,NEW:shift-violation,Policies that meet the constraint on the training patient violate it on unseen patients.,GlucoSim diabetes simulator,REPORTED,Abstract,policies satisfying constraints during training frequently violate safety requirements on unseen patients.\[16\]
1907.01475,Generalizing from a few environments in safety-critical reinforcement learning,2019,arXiv (SafeML workshop),NEW:shift-violation,Agents safe on few training levels fail catastrophically on unseen levels.,gridworlds and CoinRun,REPORTED,Abstract,We find RL algorithms can fail dangerously on unseen test environments even when performing perfectly on training environments.\[17\]
2301.12593,Risk-Averse Model Uncertainty for Distributionally Robust Safe Reinforcement Learning,2023,NeurIPS,NEW:shift-violation,Misspecified domain randomisation gives little safety benefit over standard safe RL under perturbation.,RWRL perturbed dynamics,REPORTED,full text not accessed,\[18\]
2510.17564,Towards a Practical Understanding of Lagrangian Methods in Safe Reinforcement Learning,2025,arXiv,NEW:budget-regime-sensitivity,λ is highly sensitive and constraint restrictiveness changes with the cost limit within a task.,8\[19\] Safety-Gymnasium tasks,REPORTED,Abstract,Our results reveal the highly sensitive nature of λ and further show that the restrictiveness of the constraint cost can vary across different cost limits within the same task.\[4\]
2609.15315,Evaluation Metrics for Safe Reinforcement Learning,2026,arXiv,NEW:eval-mode-shift,A policy within budget with exploration noise violates it when evaluated deterministically.,CRAX Safe Circle Point Level 1,REPORTED,full text not accessed,\[20\]
2206.02675,Effects of Safety State Augmentation on Safe Exploration,2022,NeurIPS,P6?,Sparse and unknown cost gives the learner too little signal and causes violations.,Safety Gym / safe pendulum,ARGUED,Abstract,"Often the safety cost is sparse and unknown, which unavoidably leads to constraint violations"\[21\]
2206.02675,Effects of Safety State Augmentation on Safe Exploration,2022,NeurIPS,NEW:budget-history-dependence,Safe action depends on cost already spent so a Markov policy without budget state fails.,custom static point goal (Safety Gym),ARGUED,Sec. 2.2 Fig. 1,"at the crossing of these paths robot needs to know which path it took to this state. Switching from the green path to the blue one will lead to a constraint violation."\[21\]
2606.11266,Seeing Before Colliding: Anticipatory Safe RL with Frozen Vision-Language Models,2026,arXiv,P6?,Cost arrives only after collision and λ only rises after the budget is exceeded.,racing tasks (FormulaOne L2),ARGUED,full text not accessed,\[22\]
2608.30283,BCPPO: Bachelier-Inspired Constrained Proximal Policy Optimization for Tail-Risk-Aware Safe Reinforcement Learning,2026,arXiv,P3?,An expected-cost budget is met even when a rare fraction of rollouts incurs large cost.,Safety-Gymnasium Push1 etc.,ARGUED,full text not accessed,\[23\]
2512.23770,SB-TRPO: Towards Safe Reinforcement Learning with Hard Constraints,2025,arXiv,NEW:zero-budget-ratchet,With zero budget the multiplier only grows and traps the policy in conservative regimes.,Safety-Gymnasium with cost limit 0,ARGUED,full text not accessed,\[24\]
```
```csv
# by_product_A.csv
paper_id,title,year,interaction
2006.12136,Safe Reinforcement Learning via Curriculum Induction,2020,A learned teacher picks reset-controller interventions per stage so that the CMDP student never violates constraints during training.\[25\]\[26\]
2206.02675,Effects of Safety State Augmentation on Safe Exploration,2022,The curriculum is over the safety budget itself: PI or Q-learning controllers raise the budget from strict to target during training.\[21\]
2606.20376,CRAX: Fast Safe Reinforcement Learning Benchmarking,2026,A hand-made curriculum over difficulty levels (more hazards; smaller goals) is run with seven safe RL learners; it helps Reacher/Pathway/Height but not Goal.\[8\]
2503.08241,HASARD: A Benchmark for Vision-Based Safe Reinforcement Learning in Embodied Agents,2025,"The HASARD abstract states the benchmark 'features three difficulty levels and two action spaces' and that 'incrementally training across difficulty levels offers an implicit learning curriculum' for its Lagrangian/PID safe RL baselines."
2509.18648,SPiDR: A Simple Approach for Zero-Shot Safety in Sim-to-Real Transfer,2025,Domain randomisation over dynamics is added to the cost constraint as a pessimism penalty from model disagreement.\[15\]
2301.12593,Risk-Averse Model Uncertainty for Distributionally Robust Safe Reinforcement Learning,2023,Domain randomisation of dynamics is the main baseline for a constrained learner and is beaten when its training distribution is misspecified.\[18\]
2506.11033,Adaptive Shielding for Safe Reinforcement Learning under Hidden-Parameter Dynamics Shifts,2025,Constrained learners train on randomised hidden physical parameters and are tested on held-out parameter ranges.\[5\]
2609.37070,Predictive Safety Curricula for Robust Legged Locomotion,2026,Contexts are allocated by a learned or empirical CVaR of episode safety return (from search snippet; full text unreadable).\[27\]
2502.15119,CurricuVLM: Towards Safe Autonomous Driving via Personalized Safety-Critical Curriculum Learning with Vision-Language Models,2025,Safety-critical driving scenarios are generated as a curriculum; whether the learner is a CMDP solver was not verified.
```
```markdown
# synthesis.md
## P1 — reward–safety conflict
counts: REPORTED=1 DESIGNED=2 REVIEWED=0 ARGUED=0
verdict: supported
strongest:
- 2606.20376: CRAX builds every task so that high reward needs cost, and Fig. 6 shows PPO-Lagrangian losing reward as the budget tightens in all tasks.\[8\]
- 2503.08241: HASARD's design ties reward and cost so that lowering cost necessarily costs reward (design rationale; full text not read).\[1\]
notes: This is the default design of every safe-RL benchmark (Safety Gym's report says the same, but it has no arXiv id or DOI so it cannot be a row).\[28\] The best curriculum-specific report is Koprulu, Simão, Jansen and Topcu, "Safety-Prioritizing Curricula for Constrained Reinforcement Learning" (ICLR 2025 proceedings; code built on CURROT with OmniSafe PPOLag), whose abstract says existing curricula "may generate tasks that cause RL agents to violate safety constraints during training and behave suboptimally after"; CURROT moves goals over hazards, but no arXiv id or DOI was found, so the id rule excludes it. P1 is supported, but as a property of single contexts, not of how the conflict is spread over Ω.

## P2 — mostly infeasible
counts: REPORTED=1 DESIGNED=0 REVIEWED=0 ARGUED=3
verdict: supported
strongest:
- 2112.12228: an impossible constraint made the multiplier grow without bound and collapsed critic and performance (Arena, Fig. 3).\[11\]
- 2105.10682: theory shows statewise multipliers diverge on infeasible states.\[12\]
notes: The evidence is about one infeasible CMDP or state. No paper reports a family where most ω are infeasible, so the distributional version of P2 is not evidenced. CRAX's "exploration dominated by violations" at Level 3 is the nearest family-level report.\[8\]

## P3 — hard tail
counts: REPORTED=0 DESIGNED=0 REVIEWED=0 ARGUED=2
verdict: argued only
strongest:
- 2509.18648: averaging the constraint over randomised dynamics can underestimate cost in the deployment context.\[15\]
- 2608.30283: an expected-cost budget is met even when a rare fraction of rollouts is very costly.\[23\]
notes: Koprulu, Simão, Jansen and Topcu, "Risk-Aware Curriculum Generation for Heavy-Tailed Task Distributions" (UAI 2023), would be the natural report, but no id was verified and it is unconstrained. A report showing an averaged constraint satisfied while rare hard contexts violate it would change this verdict.

## P4 — context-dependent safety
counts: REPORTED=1 DESIGNED=0 REVIEWED=0 ARGUED=1
verdict: supported
strongest:
- 2506.11033: across randomised physical parameters, constrained baselines fail to balance safety and return even when the parameters are given as inputs.\[5\]
- 2609.15915: argues that the same state has different safety under different task posteriors.\[7\]
notes: The support is thin (one experimental paper). CARL (2110.02102) was not opened, so its context-blind result is not counted here. The 2506.11033 result covers both blind and context-aware policies, so it does not isolate "one behaviour cannot be safe everywhere".

## P5 — dual-variable lag
counts: REPORTED=2 DESIGNED=0 REVIEWED=0 ARGUED=2
verdict: supported
strongest:
- 2412.04426: a multiplier calibrated offline (≈1500) is far from what online finetuning needs (≈0.65), which causes violations or stagnation.\[3\]
- 2007.03964: argues that slow multiplier updates let many constraint-violating iterations pass before the penalty bites.\[9\]
notes: Both reported rows come from one paper, and the shift there is offline→online, not a curriculum moving over Ω.\[3\] No paper reports λ lag at a curriculum stage switch or a change in task distribution. Stooke et al.'s oscillation happens under a fixed distribution and is filed as NEW:multiplier-oscillation.

## P6 — uninformative sub-budget cost
counts: REPORTED=0 DESIGNED=0 REVIEWED=0 ARGUED=2
verdict: argued only
strongest:
- 2206.02675: argues that sparse, unknown cost leads to violations and adds a remaining-budget state to give distance-to-boundary.\[21\]
- 2606.11266: argues that cost is reactive and λ grows only after the budget is exceeded.\[22\]
notes: The Sauté seed (2202.06558) abstract gives no failure report for P6. An ablation showing Lagrangian learners blind to the boundary while under budget would make this supported.

## NEW:multiplier-oscillation — λ/cost oscillation and overshoot around the limit
counts: REPORTED=3 DESIGNED=0 REVIEWED=0 ARGUED=0
verdict: supported
strongest:
- 2007.03964: cost and multiplier oscillate with a 90° phase lag on DoggoButton1, and the authors saw numerous problematic cases.\[9\]
- 2310.12567: Lagrangian methods oscillate more than projection methods in Safety-Gymnasium (full text not read).\[10\]
notes: This is the best-documented failure overall, but it is in-distribution. It matters for situations because any change in context distribution adds a disturbance to a controller that is already badly damped.

## NEW:over-conservatism — feasible but budget-underusing or trivially safe policies
counts: REPORTED=2 DESIGNED=0 REVIEWED=0 ARGUED=0
verdict: supported
strongest:
- 2606.20376: direct training on CRAX Pathway/Height leaves budget unused, and PPOLag satisfies every Level 2–3 bound with low reward.\[8\]
- 2606.20376: unconstrained-to-safe transfer fixes this in Pathway, but the reverse holds in Goal.\[8\]
notes: Over-conservatism, not infeasibility, may explain why CRAX's curriculum helps Pathway/Height but not Goal. This is our hypothesis, not a claim by the authors.

## NEW:hard-context-exploration — violation-dominated exploration in the hardest contexts
counts: REPORTED=1 DESIGNED=0 REVIEWED=0 ARGUED=0
verdict: supported
strongest:
- 2606.20376: at Level 3, exploration is dominated by constraint violations and sparse progress.\[8\]
notes: Single source. It is close to P2 but differs: the contexts are feasible, just hard to find a safe path in.

## NEW:budget-regime-sensitivity — the same task changes character with the cost limit
counts: REPORTED=2 DESIGNED=0 REVIEWED=0 ARGUED=0
verdict: supported
strongest:
- 2510.17564: constraint restrictiveness varies across cost limits within one task, and λ is highly sensitive.\[4\]\[19\]
- 2606.20376: tightening the bound hurts far more than loosening it helps.\[8\]
notes: This suggests the budget d should be a context dimension in ω.

## NEW:shift-violation — constraint satisfied in training contexts, violated in unseen ones
counts: REPORTED=5 DESIGNED=0 REVIEWED=0 ARGUED=0
verdict: supported
strongest:
- 2601.21094: across eight safe RL algorithms, three diabetes types and three age groups, policies that satisfy the constraint on the training patient frequently violate it on unseen patients; test-time shielding gives "Time-in-Range gains of 13--14%" for PPO-Lag and CPO.
- 2509.18648: domain-randomised constrained training often fails the constraints on target dynamics.\[15\]
notes: This is the most-reported family-level property. 1907.01475 is not a CMDP learner (it counts catastrophes).

## NEW:constraint-type-dependence — method success depends on the kind of constraint
counts: REPORTED=1 DESIGNED=0 REVIEWED=0 ARGUED=0
verdict: supported
strongest:
- 2606.20376: Saute meets the bound only in Reacher/Pathway and otherwise behaves like unconstrained PPO.\[8\]
notes: Single source.

## NEW:feasibility-first-stall — reward gets no traction until a feasible policy is found
counts: REPORTED=0 DESIGNED=0 REVIEWED=0 ARGUED=1
verdict: argued only
strongest:
- 2112.12228: with normalised multipliers and many constraints, the main task gets almost no weight while feasibility is sought.\[11\]
notes: The Sec. 6.1 multi-constraint experiment (not retrieved) may contain the matching report.\[11\]

## NEW:budget-history-dependence — the safe action depends on budget already spent
counts: REPORTED=0 DESIGNED=0 REVIEWED=0 ARGUED=1
verdict: argued only
strongest:
- 2206.02675: crossing paths need knowledge of the cost already incurred; a policy that ignores it violates the constraint.\[21\]
notes: The authors say experiments confirm it, but no figure-level quote was extracted.\[21\]

## NEW:stale-constraint — a fixed constraint stops describing safety after the context changes
counts: REPORTED=0 DESIGNED=0 REVIEWED=0 ARGUED=1
verdict: argued only
strongest:
- 2605.18842: a policy may satisfy an outdated constraint that no longer captures safety.\[14\]
notes: Single-author preprint, single domain.\[14\]

## NEW:zero-budget-ratchet — with d=0 the multiplier only increases
counts: REPORTED=0 DESIGNED=0 REVIEWED=0 ARGUED=1
verdict: argued only
strongest:
- 2512.23770: once increased, the multiplier cannot decrease, which traps policies in conservative regimes.\[24\]
notes: Snippet only. It is the zero-budget limit of P6, since there is no sub-budget region at all.

## NEW:eval-mode-shift — stochastic-training safety does not carry over to deterministic evaluation
counts: REPORTED=1 DESIGNED=0 REVIEWED=0 ARGUED=0
verdict: supported
strongest:
- 2609.15315: Spoor, Plaat, Moerland and co-authors (submitted 14 Sep 2026; SafeRLEval suite) ask "whether training-time behavior is representative of behavior of the final converged policy"; their snippet shows a converged PPO-Lag policy under budget with noise that violates it when evaluated deterministically on CRAX, but the abstract does not confirm this specific case.
notes: Single-seed illustration according to the snippet; not read in full.\[20\]

## Ambiguities
- 2007.03964 P5? (two rows): the lag argument concerns timescale separation under a fixed distribution, not a shift in the context distribution.
- 2509.18648 P3?: the argument is about averaging over dynamics hiding the target context; that fits a hard tail only if the target is a rare context.
- 2608.30283 P3?: the tail is over trajectories, not contexts.
- 2206.02675 P6?: sparse cost is about the signal being uninformative, but not specifically below the budget.
- 2606.11266 P6?: the argument is about cost arriving too late (reactive), which is related to P6 but not identical.
- Papers excluded only because no arXiv id or DOI was verified: Ray, Achiam, Amodei 2019 Safety Gym (read in full; it supports P1 by design and explicitly randomises layouts for generalisation); Koprulu et al. ICLR 2025 Safety-Prioritizing Curricula for Constrained Reinforcement Learning (read in full; the core by-product A paper and P1 evidence);\[25\] Klink et al. 2022 CURROT; Koprulu et al. UAI 2023; Yao et al. 2023 CCPO.\[25\]\[28\]

## Search log
Date of all searches: 2026-10-06.
Impossible protocol steps:
- arXiv API field-restricted queries (export.arxiv.org) were refused (robots disallowed). Two attempts were made: the Q3 string and the QA1 string, both as abs: field queries.
- Semantic Scholar Graph API returned HTTP 429 on both attempts, so the full reference and citation lists for the seeds could not be pulled.
- OpenReview PDFs were blocked by bot detection, so REVIEWED=0 everywhere reflects a tooling gap, not an absence of evidence.
- Google Scholar forward citations were not available.
What was done instead:
- A general web search engine was used as a proxy for each Q/QA query.
- Snowballing was done manually from reference lists and from citing papers that surfaced in search results.
- The protocol hit counts (69, 137, 8, 60, 10, 31, 6, 35) could not be reproduced.
- Q2 was not screened to 60, because the web search returns about 10 results per query.
- One round of snowballing was done through one delegated sub-search, from 2506.11033/2412.04426. It added 2609.15915 (included), 2605.18842 (included) and 2101.00531 (excluded). A second round was not run, so the stop criterion was not verified.
Web-search strings as executed (each mapped to the nearest protocol query):
1. Safe Curriculum Generation constrained CURROT Klink [QA1]
2. Benchmarking Safe Exploration in Deep Reinforcement Learning Ray Achiam Amodei [seed Safety Gym]
3. arxiv Safety-Prioritizing Curricula for Constrained Reinforcement Learning Koprulu [QA1]
4. CRAX Fast Safe Reinforcement Learning Benchmarking Tomilin curriculum [seed 2606.20376]
5. "Safety-Prioritizing Curricula" arxiv.org abs [QA1]
6. safe reinforcement learning generalization unseen environments constraint violation distribution shift [Q2]
7. constrained reinforcement learning infeasible constraint feasibility Lagrange multiplier diverges [Q4]
8. Safety Gym "Benchmarking Safe Exploration" cdn.openai.com pdf difficulty levels hazards [seed]
9. Lagrange multiplier drops to zero constraint satisfied safe RL cost signal vanishes conservative [Q3/P6]
10. unsupervised environment design safety constraint violations adversarial levels safe RL [QA2]
11. Sauté RL safety budget state augmentation sparse cost Lagrangian fails [seed 2202.06558]
12. OmniSafe benchmark results Lagrangian methods oscillation cost limit Safety-Gymnasium failure [seed 2305.09304/Q3]
13. safe reinforcement learning domain randomization constraint satisfaction varying dynamics contextual CMDP [Q6]
14. multi-task constrained reinforcement learning per-task Lagrange multiplier heterogeneous constraints [P5]
15. safe RL over-conservative policy cost limit zero hard tail rare hazardous scenarios constrained RL [Q5/P3]
16. Koprulu Simão Jansen Topcu arXiv 2024 safe curriculum constrained reinforcement learning [QA1]
17. HASARD vision-based safe RL benchmark difficulty levels agents shortcut hazards reward cost [snowball 2606.20376]
Incidental search-snippet hits not opened and not obviously in scope were not given screening rows.
```

**What surprised me.** The best-documented failures were not on the list. They are multiplier oscillation, over-conservatism, and constraints that hold on training contexts but break on unseen ones (NEW:shift-violation, 5 REPORTED rows). The best reported evidence for P5 involves an offline→online shift, not a change in curriculum.\[3\] No paper shows a family where most contexts are infeasible (P2) or a rare hard tail (P3). For the CRAX negative result in Goal, the paper itself points to a difference in conservatism between suites, not infeasibility.

**What the protocol missed.** Two key sources have no arXiv id or DOI, so the id rule drops them: Safety Gym and the ICLR 2025 SCG paper. The cost budget d is itself a context dimension (NEW:budget-regime-sensitivity), and the protocol has no slot for it.

**Next hour.** First, get the full citation lists for 2606.20376, 2006.12136 and SCG from Semantic Scholar once rate limits allow. Second, read the OpenReview threads for SCG, HASARD and CRAX for reviewer-raised failures. Third, open CARL (2110.02102) for P4. Fourth, look for a curriculum paper that logs λ across stage switches; if none exists, that gap is itself a thesis contribution.

## Sources

1. [(PDF) HASARD: A Benchmark for Vision-Based Safe Reinforcement Learning in Embodied Agents](https://www.researchgate.net/publication/389748757_HASARD_A_Benchmark_for_Vision-Based_Safe_Reinforcement_Learning_in_Embodied_Agents)
2. [Curriculum Design for Trajectory-Constrained Agent:](https://openreview.net/pdf?id=zDU5sfYK1Z)
3. [Towards Fast Safe Online Reinforcement Learning via Policy Finetuning](https://arxiv.org/pdf/2412.04426)
4. [An Empirical Study of Lagrangian Methods in Safe ...](https://arxiv.org/pdf/2510.17564)
5. <https://arxiv.org/pdf/2506.11033>
6. <https://arxiv.org/abs/2605.18842>
7. [Safe Meta-Reinforcement Learning via Information Space Reachability](https://arxiv.org/html/2609.15915v1)
8. <https://arxiv.org/html/2606.20376v1>
9. <https://arxiv.org/pdf/2007.03964>
10. [Safety-Gymnasium: A Unified Safe Reinforcement Learning Benchmark](https://arxiv.org/pdf/2310.12567)
11. [Direct Behavior Specification via Constrained Reinforcement Learning](https://arxiv.org/pdf/2112.12228)
12. [Feasible Actor-Critic: Constrained Reinforcement Learning for Ensuring Statewise Safety](https://arxiv.org/html/2105.10682.pdf)
13. [Reachability Constrained Reinforcement Learning](https://arxiv.org/pdf/2205.07536)
14. [Safe Continual Reinforcement Learning under Nonstationarity via Adaptive Safety Constraints](https://arxiv.org/html/2605.18842v1)
15. <https://arxiv.org/pdf/2509.18648>
16. <https://arxiv.org/abs/2601.21094>
17. <https://arxiv.org/abs/1907.01475>
18. [Risk-Averse Model Uncertainty for Distributionally Robust Safe Reinforcement Learning](https://arxiv.org/pdf/2301.12593)
19. [Towards a Practical Understanding of Lagrangian Methods in Safe Reinforcement Learning](https://arxiv.org/html/2510.17564)
20. [Evaluation Metrics for Safe Reinforcement Learning](https://arxiv.org/html/2609.15315v1)
21. <https://arxiv.org/pdf/2206.02675>
22. [Seeing Before Colliding: Anticipatory Safe RL with Frozen Vision-Language Models](https://arxiv.org/html/2606.11266)
23. [BCPPO: Bachelier-Inspired Constrained Proximal Policy Optimization for Tail-Risk-Aware Safe Reinforcement Learning](https://arxiv.org/pdf/2608.30283)
24. [Safety-Biased Policy Optimisation: Towards Hard-Constrained Reinforcement Learning via Trust Regions](https://arxiv.org/html/2512.23770v1)
25. <https://proceedings.iclr.cc/paper_files/paper/2025/file/120ed726cf129dbeb8375b6f8a0686f8-Paper-Conference.pdf>
26. <https://arxiv.org/pdf/2006.12136>
27. [Predictive Safety Curricula for Robust Legged Locomotion](https://arxiv.org/html/2609.37070v1)
28. <https://cdn.openai.com/safexp-short.pdf>
