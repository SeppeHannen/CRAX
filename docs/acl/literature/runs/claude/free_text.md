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
