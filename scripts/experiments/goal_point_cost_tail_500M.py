"""The cost tail: does the policy need to *see* that a gap is closed, to *remember*
that it has tried it, or both?

On level 3 the per-context learner's mean cost is carried by one episode in two hundred: the
agent enters a gap between two solid hazards too narrow to pass and re-approaches it for the
rest of the episode (``docs/acl/decisions/2026-10-18_vision_set_aside.md`` §1–2; measured
2026-10-19: 76 % of all cost is solid contact, contact is intermittent, median 32 steps between
touches). Two things the observation lacks, each one arm, and their combination:

* ``hazard_lidar=surface_by_kind`` — the hazard lidar as two arrays, passable and solid, reading
  the closeness of the nearest *surface* over the arc each hazard subtends (``crax/envs/safe_goal.py``,
  ``HazardLidar``): the fact that the gap is closed, in the present observation;
* a memory — an exponential trace of (sensors, action, cost) with the median inter-touch interval,
  32 steps, as its time constant: that the agent has been touching, and not moving, for a while.

**The memory was removed from the code after this experiment** (it changed neither the tail nor
the return; ``docs/acl/experiments/2026-10-19_goal_point_cost_tail_500M.md``). The two arms that
used it, ``memory`` and ``geometry_memory``, ran from the tag ``experiments-2026-10`` and are no
longer defined here; check that tag out to rerun or replay them.

The control is the settled learner of 2026-10-15 (`uniform additive told`) rerun on the current
code, with the context normaliser fix. Everything else is the sensors run of 2026-10-15. Judged
on level 3 and on Ω by return, mean cost and the tail the evaluator now logs (median, share over
the budget, share over 4× it), then on several thousand replayed level-3 and level-1 episodes.

A fifth arm, added 2026-10-19 after the four had run: the surface lidar under PLR (reward score),
the best policy of 2026-10-15, so that the lidar's effect is read on the same distribution as the
replayed centre-lidar policy it is compared with (the four arms' control stood still this seed).

    .venv/bin/python -m training.experiments.snellius scripts/experiments/goal_point_cost_tail_500M.py \\
        --partition gpu_a100 --time 03:00:00            # the four arms, 2026-10-19 16:00
    .venv/bin/python -m training.experiments.snellius scripts/experiments/goal_point_cost_tail_500M.py \\
        --partition gpu_a100 --time 03:00:00 --only plr_geometry
    # or, one after another on this machine (~1.6 h each at 85 k SPS):
    setsid nohup .venv/bin/python -m training.experiments.local scripts/experiments/goal_point_cost_tail_500M.py \\
        > runs/logs/goal_point_cost_tail_500M_chain.log 2>&1 < /dev/null & disown
"""
from training.experiments import Arm, Experiment

UNIFORM = ("--context_distribution", "uniform")
GEOMETRY = ("--env_kwargs", '{"hazard_lidar": "surface_by_kind"}')

EXPERIMENT = Experiment(
    group="goal_point_cost_tail_500M",
    common=(
        "--env_name", "safe_goal_point", "--alg", "ppo_lag",
        "--deployment_distribution", "level:3",
        "--constraint", "additive", "--policy_context", "told",
        "--initial_lambda_lagr", "0.1", "--lagrangian_coef_rate", "1",
        "--num_envs", "8192", "--episode_length", "1000",
        "--unroll_length", "20", "--batch_size", "1024", "--num_minibatches", "32", "--num_updates_per_batch", "4",
        "--num_timesteps", "500_000_000", "--num_evals", "21",
        "--record_episodes", "4", "--store_model", "true",
        "--measure_performance", "--skip_rollout", "--skip_video",
    ),
    arms=(
        Arm("control", UNIFORM),
        Arm("geometry", (*UNIFORM, *GEOMETRY)),
        Arm("plr_geometry", ("--context_distribution", "plr", *GEOMETRY)),
    ),
)
