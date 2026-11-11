#!/usr/bin/env bash
# The experiments of the generalization evaluation (docs/acl/README.md, steps 1, 3, 4, 5) on one
# machine, one after another. Seed by seed: every experiment at seed 0 before any at seed 1, so
# the first complete comparison exists after the first eight runs. After each block of runs the
# group is scored (training/score.py) and its figure redrawn, so results appear as the chain goes.
#
#   scripts/experiments/generalization_chain.sh            # seeds 0 1 2
#   scripts/experiments/generalization_chain.sh 1 2        # only these seeds
#
# Run it from a checkout nobody edits (git worktree add ../CRAX-run <commit>), detached:
#
#   PYTHON=/path/to/.venv/bin/python setsid nohup scripts/experiments/generalization_chain.sh \
#       > runs/logs/generalization_chain.log 2>&1 < /dev/null & disown
#
# A training job that fails stops the chain. To stop it by hand: kill the process group of this
# script (kill -- -<pid of the script>); the run in progress is lost, the finished ones are in W&B.
set -euo pipefail

PYTHON=${PYTHON:-.venv/bin/python}
BASELINE=goal_point_hazard_total_500M
CURRICULA=goal_point_hazard_total_curricula_500M
EXPERIMENTS=("$BASELINE" goal_point_drag_500M goal_point_combination_500M "$CURRICULA")
SEEDS=("${@:-0 1 2}")

for seed in ${SEEDS[@]}; do
    for experiment in "${EXPERIMENTS[@]}"; do
        definition="scripts/experiments/$experiment.py"
        labels=$("$PYTHON" -m training.experiments.local "$definition" --list | grep "_seed${seed}\$")
        echo "$(date +%Y%m%d_%H%M%S) chain: $experiment seed $seed"
        # shellcheck disable=SC2086  # one argument per label
        "$PYTHON" -m training.experiments.local "$definition" --only $labels

        # The combination has 210 cells, all rolled out at once: fewer episodes per cell.
        episodes=128
        if [ "$experiment" = goal_point_combination_500M ]; then episodes=32; fi
        groups=("$experiment")
        if [ "$experiment" = "$CURRICULA" ]; then groups+=("$BASELINE"); fi  # drawn next to the uniform runs
        # A failed score or figure must not cost the remaining training runs: say so and go on.
        "$PYTHON" -m training.score --group "$experiment" --episodes "$episodes" \
            && "$PYTHON" -m scripts.analysis.generalization_figures --groups "${groups[@]}" \
            || echo "$(date +%Y%m%d_%H%M%S) chain: scoring or drawing $experiment failed; the training runs continue"
    done
done
echo "$(date +%Y%m%d_%H%M%S) chain: done"
