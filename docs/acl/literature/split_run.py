"""Split one deep-research output into the four files the protocol asked for.

The models return four fenced blocks in a fixed order (screening, evidence,
by-product A, synthesis) followed by free text. This writes them to
``runs/<model>/`` so the CSVs can be verified and diffed. Citation markers that
some tools interleave into CSV fields (``\\[12\\]``) are stripped; nothing else
is edited.

Usage:
    .venv/bin/python docs/acl/literature/split_run.py claude-deep-research.md runs/claude
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

FENCED_BLOCK = re.compile(r"^```[a-z]*\n(.*?)^```", re.S | re.M)
CITATION_MARKER = re.compile(r"\\\[\d+\\\]|\[\d+\]")
FILE_COMMENT = re.compile(r"^(#|<!--)\s*\S+\.(csv|md)\s*(-->)?\s*\n")
OUTPUT_NAMES = ("screening.csv", "evidence.csv", "by_product_A.csv", "synthesis.md")


def split(source: Path, destination: Path) -> None:
    text = source.read_text()
    blocks = FENCED_BLOCK.findall(text)
    if len(blocks) < len(OUTPUT_NAMES):
        raise ValueError(f"{source}: expected {len(OUTPUT_NAMES)} fenced blocks, found {len(blocks)}")
    destination.mkdir(parents=True, exist_ok=True)
    for name, block in zip(OUTPUT_NAMES, blocks):
        body = FILE_COMMENT.sub("", block, count=1)
        if name.endswith(".csv"):
            body = CITATION_MARKER.sub("", body)
        (destination / name).write_text(body)
        print(f"{destination / name}: {body.count(chr(10))} lines")
    free_text = text[text.rfind("```") + 3 :].strip()
    (destination / "free_text.md").write_text(free_text + "\n")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    split(Path(sys.argv[1]), Path(sys.argv[2]))
