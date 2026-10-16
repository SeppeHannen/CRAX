"""The episode review: a local page that lets a reviewer find, watch and compare
the recorded evaluation episodes of a W&B group (``docs/acl/design/episode_review.md``).

    python -m training.review --group <wandb group>

* :mod:`library` — finds a group's records (local run directories first, W&B
  files otherwise) and indexes them by arm, evaluation step and target.
* :mod:`renderer` — one frame of one record, from its scene, through MuJoCo.
* :mod:`server` — the HTTP server behind the page; reads records, never computes.
"""
