"""The return–cost mass grid as a static PNG, for slides and documents.

    .venv/bin/python -m scripts.analysis.return_cost_mass_image \\
        --policy "label" <records root> [--policy ...] --target level3 --output <png> [--cell 28]

The same picture as the interactive page (``return_cost_mass_page.py``) — one panel per
policy, a cell one goal wide and ``--cost_cell`` cost tall, colour = share of episodes, the
≥ ``--ceiling`` row on top with its share of episodes and of all cost, marginal bars with their
percentages, the budget (red, dashed) and the budget left for the ordinary episodes (orange,
dotted), × the mean and ○ the median — drawn by PIL at a chosen cell size, so it is crisp at
any size and needs no browser.
"""
from __future__ import annotations

import argparse
import json
import math
import pathlib
from typing import List, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from training.evaluation.record import METADATA_KEY, SUFFIX

FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
FONT_BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
STOPS = np.array([[255, 255, 255], [198, 219, 239], [66, 146, 198], [8, 48, 107]], dtype=float)


def read_outcomes(root: pathlib.Path, target: str) -> Tuple[np.ndarray, np.ndarray]:
    returns: List[float] = []
    costs: List[float] = []
    for path in sorted(root.rglob(f"*{SUFFIX}")):
        with np.load(path) as contents:
            if json.loads(str(contents[METADATA_KEY]))["evaluation_name"] != target:
                continue
            returns.append(float(contents["reward"].sum()))
            costs.append(float(contents["cost"].sum()))
    if not returns:
        raise FileNotFoundError(f"no {target!r} episode records under {root}")
    return np.array(returns), np.array(costs)


def colour(share: float, top: float) -> Tuple[int, int, int]:
    t = min(1.0, max(0.0, math.sqrt(share / top))) if top > 0 else 0.0
    k = min(2, int(t * 3))
    f = t * 3 - k
    rgb = STOPS[k] + (STOPS[k + 1] - STOPS[k]) * f
    return tuple(int(round(v)) for v in rgb)


def draw_panel(
    label: str, returns: np.ndarray, costs: np.ndarray, *, cols: int, cost_cell: float, ceiling: float, budget: float,
    cell: int, top: float, row_max: float, col_max: float,
) -> Image.Image:
    body_rows = int(round(ceiling / cost_cell))
    rows = body_rows + 1
    tail = body_rows
    n = len(costs)
    row_of = lambda c: tail if c >= ceiling else int(c // cost_cell)
    col_of = lambda g: max(0, min(cols - 1, int(round(g))))
    grid = np.zeros((rows, cols))
    for r, c in zip(returns, costs):
        grid[row_of(c), col_of(r)] += 1
    row_share, col_share = grid.sum(1) / n, grid.sum(0) / n
    cost_share_tail = costs[costs >= ceiling].sum() / costs.sum()
    tail_per_episode = costs[costs >= ceiling].sum() / n

    left, header, side, under = 90, 70, int(cols * cell * 0.30) + 60, int(rows * cell * 0.26) + 40
    grid_w, grid_h = cols * cell, (rows + 1) * cell
    image = Image.new("RGB", (left + grid_w + side, header + grid_h + under), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype(FONT, max(10, cell // 2))
    small = ImageFont.truetype(FONT, max(9, int(cell * 0.42)))
    bold = ImageFont.truetype(FONT_BOLD, max(12, cell // 2 + 2))

    y_top = lambda r: header if r == tail else header + (rows - r) * cell
    row_h = lambda r: 2 * cell if r == tail else cell
    x_left = lambda c: left + c * cell
    y_cost = lambda c: y_top(tail) + cell if c >= ceiling else y_top(int(c // cost_cell)) + (1 - (c / cost_cell - int(c // cost_cell))) * cell

    for r in range(rows):
        for c in range(cols):
            draw.rectangle([x_left(c), y_top(r), x_left(c + 1) - 1, y_top(r) + row_h(r) - 1], fill=colour(grid[r, c] / n, top))
    strong = max(1, round(5 / cost_cell))
    for c in range(cols + 1):
        draw.line([(x_left(c), header), (x_left(c), header + grid_h)], fill="#aaaaaa" if c % 5 == 0 else "#e4e4e4")
    for r in range(body_rows + 1):
        y = header + (rows - r) * cell
        draw.line([(left, y), (left + grid_w, y)], fill="#aaaaaa" if r % strong == 0 else "#e4e4e4")
    yt = y_top(tail) + row_h(tail)
    for x in range(left, left + grid_w, 6):
        draw.line([(x, yt), (x + 3, yt)], fill="#333333")
    draw.rectangle([left, header, left + grid_w, header + grid_h], outline="#333333")

    yb = y_cost(budget)
    for x in range(left, left + grid_w + side, 12):
        draw.line([(x, yb), (x + 7, yb)], fill="#d62728", width=2)
    yl = y_cost(budget - tail_per_episode)
    for x in range(left, left + grid_w + side, 6):
        draw.line([(x, yl), (x + 2, yl)], fill="#e07b00", width=2)

    s = max(5, int(cell * 0.35))
    mx, my = left + (returns.mean() + 0.5) * cell, y_cost(costs.mean())
    draw.line([(mx - s, my - s), (mx + s, my + s)], fill="black", width=2)
    draw.line([(mx - s, my + s), (mx + s, my - s)], fill="black", width=2)
    mdx, mdy = left + (np.median(returns) + 0.5) * cell, y_cost(np.median(costs))
    draw.ellipse([mdx - s, mdy - s, mdx + s, mdy + s], outline="black", width=2)

    for r in range(0, body_rows + 1, strong):
        draw.text((left - 6, header + (rows - r) * cell), f"{r * cost_cell:g}", fill="#333", font=font, anchor="rm")
    draw.text((left - 6, y_top(tail) + cell), f"≥{ceiling:g}", fill="#333", font=font, anchor="rm")
    draw.text((left - 6, yb - 10), "budget", fill="#d62728", font=bold, anchor="rm")
    # the orange value goes below its line, and further down if the budget label is in the way
    draw.text((left - 6, yl + (20 if abs(yl - yb) < 24 else 10)), f"{budget - tail_per_episode:.1f}", fill="#e07b00", font=bold, anchor="rm")

    bar_w = side - 130
    for r in range(rows):
        w = row_share[r] / row_max * bar_w if row_max else 0
        if r == tail:
            wc = min(cost_share_tail / row_max * bar_w, bar_w) if row_max else 0
            draw.rectangle([x_left(cols) + 3, y_top(r) + 1, x_left(cols) + 3 + w, y_top(r) + cell - 1], fill="#9aa5b1")
            draw.rectangle([x_left(cols) + 3, y_top(r) + cell, x_left(cols) + 3 + wc, y_top(r) + 2 * cell - 1], fill="#b56576")
            draw.text((x_left(cols) + 6 + w, y_top(r) + cell / 2), f"{100 * row_share[r]:.1f}% of episodes", fill="#333", font=small, anchor="lm")
            draw.text((x_left(cols) + 6 + wc, y_top(r) + 1.5 * cell), f"{100 * cost_share_tail:.0f}% of all cost", fill="#b56576", font=small, anchor="lm")
        else:
            draw.rectangle([x_left(cols) + 3, y_top(r) + 1, x_left(cols) + 3 + w, y_top(r) + cell - 2], fill="#9aa5b1")
            if row_share[r] > 0:
                draw.text((x_left(cols) + 6 + w, y_top(r) + cell / 2), f"{100 * row_share[r]:.1f}%", fill="#333", font=small, anchor="lm")

    axis_y = header + grid_h + 16
    for c in range(cols):
        if c % 5 == 0 or cell >= 16:
            draw.text((x_left(c) + cell / 2, header + grid_h + 8), str(c), fill="#333", font=font, anchor="mm")
    bar_max = under - 50
    for c in range(cols):
        h = col_share[c] / col_max * bar_max if col_max else 0
        draw.rectangle([x_left(c) + 1, axis_y, x_left(c + 1) - 2, axis_y + h], fill="#9aa5b1")
        if col_share[c] >= 0.005 and cell >= 14:
            draw.text((x_left(c) + cell / 2, axis_y + h + 3), f"{100 * col_share[c]:.0f}", fill="#333", font=small, anchor="mt")
    draw.line([(left, axis_y), (left + grid_w, axis_y)], fill="#333")
    draw.text((left + grid_w, image.height - 6), "goals in the episode  ·  bars: % of episodes per goal count", fill="#555", font=small, anchor="rb")
    draw.text((x_left(cols) + 4, header + grid_h + 14), "% per cost band", fill="#555", font=small, anchor="lm")

    draw.text((left, 8), label, fill="#111", font=bold, anchor="lt")
    draw.text((left, 8 + bold.size + 8),
              f"{n} episodes · goals {returns.mean():.1f} · cost mean {costs.mean():.1f}, median {np.median(costs):.1f} · worst {costs.max():.0f}",
              fill="#444", font=font, anchor="lt")
    return image


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--policy", nargs=2, action="append", metavar=("LABEL", "RECORDS_ROOT"), required=True)
    parser.add_argument("--target", default="level3")
    parser.add_argument("--budget", type=float, default=25.0)
    parser.add_argument("--ceiling", type=float, default=50.0)
    parser.add_argument("--cost_cell", type=float, default=2.5)
    parser.add_argument("--cell", type=int, default=28, help="cell size in pixels")
    parser.add_argument("--output", type=pathlib.Path, required=True)
    arguments = parser.parse_args()
    outcomes = [(label, *read_outcomes(pathlib.Path(root), arguments.target)) for label, root in arguments.policy]
    cols = int(max(r.max() for _, r, _ in outcomes)) + 1
    body_rows = int(round(arguments.ceiling / arguments.cost_cell))

    def shares(returns: np.ndarray, costs: np.ndarray) -> Tuple[float, float, float]:
        grid = np.zeros((body_rows + 1, cols))
        for r, c in zip(returns, costs):
            grid[body_rows if c >= arguments.ceiling else int(c // arguments.cost_cell), max(0, min(cols - 1, int(round(r))))] += 1
        grid /= len(costs)
        return grid.max(), grid.sum(1).max(), grid.sum(0).max()

    top, row_max, col_max = (max(values) for values in zip(*(shares(r, c) for _, r, c in outcomes)))
    panels = [draw_panel(label, r, c, cols=cols, cost_cell=arguments.cost_cell, ceiling=arguments.ceiling, budget=arguments.budget,
                         cell=arguments.cell, top=top, row_max=row_max, col_max=col_max) for label, r, c in outcomes]
    gap = 24
    image = Image.new("RGB", (sum(p.width for p in panels) + gap * (len(panels) - 1), max(p.height for p in panels)), "white")
    x = 0
    for panel in panels:
        image.paste(panel, (x, 0))
        x += panel.width + gap
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    image.save(arguments.output)
    print(f"wrote {arguments.output} ({image.width}x{image.height})")


if __name__ == "__main__":
    main()
