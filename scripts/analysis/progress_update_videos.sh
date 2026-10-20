#!/usr/bin/env bash
# The side-by-side videos of docs/acl/progress_update_2026-10-21.md §5: the published baseline
# (uniform, scalar λ, blind policy; trained 2026-10-15) against the best contextual policy
# (PLR, λ(ω) additive, told, surface lidar by kind; trained 2026-10-19) on levels 1-3.
# Per level the episode nearest each policy's median cost among its above-median-return
# episodes ("typical"), and on level 3 also each policy's worst episode ("trap").
# Chosen 2026-10-20 from the replay tables (/tmp/pick_episodes.py over the records below).
set -euo pipefail
cd "$(dirname "$0")/../.."
DATA=../crax-data
BASE=$DATA/replays_2026-10-20_scalar_blind/safe_goal_point_ctx_uniform_scalar_blind_ppo_lag_seed0_20261015_173731_432147
BEST=$DATA/replays_2026-10-19/safe_goal_point_ctx_plr_additive_told_hazard_lidar_surface_by_kind_ppo_lag_seed0_20261019_220249_048179
OUT=docs/acl/experiments/figures/2026-10-19_goal_point_cost_tail_500M/videos
LEFT="baseline: scalar λ, blind policy, centre lidar"
RIGHT="λ(ω) additive, told policy, surface lidar (PLR)"
video() { MUJOCO_GL=egl .venv/bin/python -m scripts.analysis.side_by_side_video --left "$LEFT" "$2" --right "$RIGHT" "$3" --output "$OUT/$1.mp4"; }

video level1_typical "$BASE/seed1/000511180800_level1_113.episode.npz" "$BEST/seed1/000511180800_level1_365.episode.npz"
video level2_typical "$BASE/seed1/000511180800_level2_129.episode.npz" "$BEST/seed12/000511180800_level2_177.episode.npz"
video level3_typical "$BASE/seed2/000511180800_level3_160.episode.npz" "$BEST/seed1/000511180800_level3_301.episode.npz"
video level3_worst   "$BASE/seed1/000511180800_level3_176.episode.npz" "$BEST/seed2/000511180800_level3_84.episode.npz"
