"""An interactive page: where a policy's episodes land in the return–cost plane, on a regular grid,
with the mass of any region you drag over.

    .venv/bin/python -m scripts.analysis.return_cost_mass_page \\
        --policy "centre lidar" <records root> --policy "surface lidar by kind" <records root> \\
        --target level3 --budget 25 --ceiling 50 --cost_cell 2.5 --output <page.html>

One panel per policy, side by side, the same grid on every panel: a cell is one goal wide (x) and
``--cost_cell`` cost units tall (y); its colour is the share of that policy's episodes that landed
in it. Episodes costing ``--ceiling`` or more are gathered in one marked row at the top, so the
tail is on the page without stretching it. Along the right edge of each panel the row sums (the
share of episodes per cost band), under it the column sums (the share per goal count), as bars
with their percentages written in. Drag a rectangle on any panel: it is applied to every panel,
and the box on the right says what share of each policy's episodes it holds, with the mean goals
and cost inside it. The budget is the dashed line; × the mean, ○ the median.

One self-contained HTML file: the episodes' outcomes are embedded as JSON and the drawing is
plain canvas, so it opens from disk with no server and no dependency. Reads return and cost per
episode straight from the record files, so records from before the frames were added work too.
"""
from __future__ import annotations

import argparse
import json
import pathlib
from typing import Dict, List

import numpy as np

from training.evaluation.record import METADATA_KEY, SUFFIX


def read_outcomes(root: pathlib.Path, target: str) -> Dict[str, List[float]]:
    returns: List[float] = []
    costs: List[float] = []
    for path in sorted(root.rglob(f"*{SUFFIX}")):
        with np.load(path) as contents:
            if json.loads(str(contents[METADATA_KEY]))["evaluation_name"] != target:
                continue
            returns.append(round(float(contents["reward"].sum()), 3))
            costs.append(round(float(contents["cost"].sum()), 3))
    if not returns:
        raise FileNotFoundError(f"no {target!r} episode records under {root}")
    return {"returns": returns, "costs": costs}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--policy", nargs=2, action="append", metavar=("LABEL", "RECORDS_ROOT"), required=True)
    parser.add_argument("--target", default="level3")
    parser.add_argument("--budget", type=float, default=25.0)
    parser.add_argument("--ceiling", type=float, default=50.0, help="costs at or above this share one row at the top")
    parser.add_argument("--cost_cell", type=float, default=2.5, help="height of a cell in cost units; a cell is always one goal wide")
    parser.add_argument("--output", type=pathlib.Path, required=True)
    arguments = parser.parse_args()
    if arguments.ceiling <= arguments.budget:
        raise ValueError("the ceiling must lie above the budget, or the budget line has no cells to sit between")
    if (arguments.ceiling / arguments.cost_cell) % 1 or (arguments.budget / arguments.cost_cell) % 1:
        raise ValueError("the budget and the ceiling must be whole numbers of cost cells, so their lines fall on grid lines")
    payload = {
        "target": arguments.target,
        "budget": arguments.budget,
        "ceiling": arguments.ceiling,
        "cost_cell": arguments.cost_cell,
        "policies": [{"label": label, **read_outcomes(pathlib.Path(root), arguments.target)} for label, root in arguments.policy],
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(PAGE.replace("__PAYLOAD__", json.dumps(payload)))
    print(f"wrote {arguments.output} ({sum(len(p['returns']) for p in payload['policies'])} episodes)")


PAGE = r"""<!doctype html>
<html><head><meta charset="utf-8"><title>Return–cost mass</title>
<style>
  body { font: 13px/1.4 system-ui, sans-serif; margin: 16px; color: #222; }
  h1 { font-size: 16px; margin: 0 0 6px; }
  .hint { color: #555; margin-bottom: 10px; max-width: 1400px; }
  .controls { margin: 6px 0 12px; color: #444; }
  .controls label { margin-right: 20px; }
  .layout { display: flex; gap: 20px; align-items: flex-start; }
  .panels { display: flex; gap: 20px; flex: 1 1 auto; min-width: 0; }
  .panel { flex: 1 1 0; min-width: 0; }
  .panel h2 { font-size: 13px; font-weight: 600; margin: 0 0 2px; }
  .panel .stats { color: #444; margin-bottom: 6px; }
  canvas { display: block; cursor: crosshair; }
  #selection { flex: 0 0 300px; position: sticky; top: 12px; padding: 10px 12px; border: 1px solid #ccc; background: #fafafa; }
  #selection table { border-collapse: collapse; margin-top: 6px; width: 100%; }
  #selection td, #selection th { padding: 3px 8px 3px 0; text-align: left; vertical-align: top; }
  #selection th { font-weight: 600; color: #555; padding-top: 8px; }
  #selection .big { font-size: 18px; font-weight: 600; }
  #hover { position: fixed; pointer-events: none; background: #222; color: #fff; padding: 3px 6px; border-radius: 3px; font-size: 12px; display: none; }
</style></head>
<body>
<h1 id="title"></h1>
<div class="hint" id="hint"></div>
<div class="controls">
  <label><input type="checkbox" id="sqrt" checked> square-root colour scale (lifts faint cells)</label>
  <label><input type="checkbox" id="numbers"> write every cell's % in it</label>
</div>
<div class="layout"><div class="panels" id="panels"></div><div id="selection"></div></div>
<div id="hover"></div>
<script>
const DATA = __PAYLOAD__;
const BUDGET = DATA.budget, CEIL = DATA.ceiling, CC = DATA.cost_cell;
const MAXGOALS = Math.max(...DATA.policies.map(p => Math.max(...p.returns)));
const COLS = MAXGOALS + 1;                 // goal columns 0..MAXGOALS, one goal each
const BODY_ROWS = Math.round(CEIL / CC);   // cost rows [0,CC), [CC,2CC), ... below the ceiling
const ROWS = BODY_ROWS + 1;                // plus one row for cost >= ceiling
const TAIL_ROW = BODY_ROWS;
const MARGIN = { left: 48, top: 8, bottom: 18 };
const SIDE = 0.18, UNDER = 0.22;           // marginal bars: fraction of the grid width / height
let SQRT = true, NUMBERS = false;
let selection = null;                      // {c0,c1,r0,r1} inclusive cell ranges, shared across panels

function rowOf(cost) { return cost >= CEIL ? TAIL_ROW : Math.floor(cost / CC); }
function colOf(goals) { return Math.max(0, Math.min(COLS - 1, Math.round(goals))); }
const pct = (x, d = 1) => (100 * x).toFixed(d) + '%';

const policies = DATA.policies.map(p => {
  const n = p.returns.length;
  const grid = Array.from({ length: ROWS }, () => new Float64Array(COLS));
  const rows = new Float64Array(ROWS), cols = new Float64Array(COLS);
  for (let i = 0; i < n; i++) { const r = rowOf(p.costs[i]), c = colOf(p.returns[i]); grid[r][c] += 1; rows[r] += 1; cols[c] += 1; }
  const med = a => { const s = [...a].sort((x, y) => x - y); return s[Math.floor(s.length / 2)]; };
  const mean = a => a.reduce((x, y) => x + y, 0) / a.length;
  const total = p.costs.reduce((x, y) => x + y, 0);
  // Share of all cost per row, and what the tail (cost >= ceiling) takes from the mean: those episodes'
  // cost spread over every episode. The budget left for the others is BUDGET minus that.
  const costShare = new Float64Array(ROWS);
  for (let i = 0; i < n; i++) costShare[rowOf(p.costs[i])] += p.costs[i] / total;
  const tailCosts = p.costs.filter(c => c >= CEIL);
  const tailPerEpisode = tailCosts.reduce((x, y) => x + y, 0) / n;
  return { ...p, n, grid, rows, cols, costShare, tailPerEpisode, tailWorst: tailCosts.length ? Math.max(...tailCosts) : 0,
           meanR: mean(p.returns), meanC: mean(p.costs), medR: med(p.returns), medC: med(p.costs),
           under: p.costs.filter(c => c <= BUDGET).length / n, over4: p.costs.filter(c => c > 4 * BUDGET).length / n };
});
const TOP = Math.max(...policies.map(p => Math.max(...p.grid.flatMap(row => Array.from(row).map(v => v / p.n)))));
const ROWMAX = Math.max(...policies.map(q => Math.max(...q.rows) / q.n)), COLMAX = Math.max(...policies.map(q => Math.max(...q.cols) / q.n));

function colour(share) {
  let t = share / TOP; if (SQRT) t = Math.sqrt(t);
  t = Math.min(1, Math.max(0, t));
  const stops = [[255,255,255],[198,219,239],[66,146,198],[8,48,107]];
  const k = Math.min(2, Math.floor(t * 3)), f = t * 3 - k, a = stops[k], b = stops[k + 1];
  return `rgb(${Math.round(a[0] + (b[0]-a[0]) * f)},${Math.round(a[1] + (b[1]-a[1]) * f)},${Math.round(a[2] + (b[2]-a[2]) * f)})`;
}
const textOn = share => (SQRT ? Math.sqrt(share / TOP) : share / TOP) > 0.55 ? '#fff' : '#222';

const panelsDiv = document.getElementById('panels');
const canvases = policies.map((p, i) => {
  const div = document.createElement('div'); div.className = 'panel';
  const h = document.createElement('h2'); h.textContent = p.label; div.appendChild(h);
  const s = document.createElement('div'); s.className = 'stats';
  s.textContent = `${p.n} episodes on ${DATA.target} · goals ${p.meanR.toFixed(1)} · cost mean ${p.meanC.toFixed(1)}, median ${p.medC.toFixed(1)} · under budget ${pct(p.under, 0)} · `
    + `episodes ≥ ${CEIL}: ${pct(p.rows[TAIL_ROW] / p.n)} of episodes, ${pct(p.costShare[TAIL_ROW], 0)} of all cost, worst ${p.tailWorst.toFixed(0)} · budget left for the rest ${(BUDGET - p.tailPerEpisode).toFixed(1)}`;
  div.appendChild(s);
  const canvas = document.createElement('canvas'); div.appendChild(canvas); panelsDiv.appendChild(div);
  canvas.dataset.index = i;
  return canvas;
});

// Geometry is recomputed from the panel's width on every draw, so the grid fills what it is given.
let G = null;
function geometry(cssWidth) {
  const cell = Math.max(6, Math.floor((cssWidth - MARGIN.left) / (COLS * (1 + SIDE))));
  const gridW = COLS * cell, side = Math.round(gridW * SIDE), gridH = (ROWS + 1) * cell, under = Math.round(gridH * UNDER) + 30;
  return { cell, gridW, side, gridH, under, W: MARGIN.left + gridW + side, H: MARGIN.top + gridH + under + MARGIN.bottom };
}
function yTop(r) { return r === TAIL_ROW ? MARGIN.top : MARGIN.top + (ROWS - r) * G.cell; }   // tail row on top, 2 cells tall
function rowHeight(r) { return r === TAIL_ROW ? 2 * G.cell : G.cell; }
function xLeft(c) { return MARGIN.left + c * G.cell; }
function yOfCost(cost) { return cost >= CEIL ? yTop(TAIL_ROW) + G.cell : yTop(Math.floor(cost / CC)) + (1 - (cost / CC - Math.floor(cost / CC))) * G.cell; }
const costLabel = r => r === TAIL_ROW ? `≥${CEIL}` : `${r * CC}–${(r + 1) * CC}`;

function draw() {
  G = geometry(canvases[0].parentElement.clientWidth);
  const dpr = window.devicePixelRatio || 1;
  policies.forEach((p, i) => {
    const canvas = canvases[i];
    canvas.width = G.W * dpr; canvas.height = G.H * dpr; canvas.style.width = G.W + 'px'; canvas.style.height = G.H + 'px';
    const ctx = canvas.getContext('2d'); ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.fillStyle = '#fff'; ctx.fillRect(0, 0, G.W, G.H);
    const small = `${Math.max(8, Math.min(11, G.cell * 0.45))}px system-ui`;
    // cells
    for (let r = 0; r < ROWS; r++) for (let c = 0; c < COLS; c++) {
      const share = p.grid[r][c] / p.n;
      ctx.fillStyle = colour(share); ctx.fillRect(xLeft(c), yTop(r), G.cell, rowHeight(r));
      if (NUMBERS && share > 0 && G.cell >= 18) { ctx.fillStyle = textOn(share); ctx.font = small; ctx.textAlign = 'center'; ctx.textBaseline = 'middle'; ctx.fillText((100 * share).toFixed(1), xLeft(c) + G.cell / 2, yTop(r) + rowHeight(r) / 2); }
    }
    // grid lines: faint every cell, stronger every 5 goals and every 5 cost units
    const strongEvery = Math.max(1, Math.round(5 / CC));
    for (let c = 0; c <= COLS; c++) { ctx.strokeStyle = c % 5 ? '#e4e4e4' : '#aaa'; ctx.beginPath(); ctx.moveTo(xLeft(c) + .5, MARGIN.top); ctx.lineTo(xLeft(c) + .5, MARGIN.top + G.gridH); ctx.stroke(); }
    for (let r = 0; r <= BODY_ROWS; r++) { const y = MARGIN.top + (ROWS - r) * G.cell; ctx.strokeStyle = r % strongEvery ? '#e4e4e4' : '#aaa'; ctx.beginPath(); ctx.moveTo(MARGIN.left, y + .5); ctx.lineTo(xLeft(COLS), y + .5); ctx.stroke(); }
    ctx.strokeStyle = '#333'; ctx.setLineDash([3, 3]); ctx.beginPath(); ctx.moveTo(MARGIN.left, yTop(TAIL_ROW) + rowHeight(TAIL_ROW) + .5); ctx.lineTo(xLeft(COLS), yTop(TAIL_ROW) + rowHeight(TAIL_ROW) + .5); ctx.stroke(); ctx.setLineDash([]);
    ctx.strokeStyle = '#333'; ctx.strokeRect(MARGIN.left + .5, MARGIN.top + .5, G.gridW, G.gridH);
    // budget line, across the grid and the side bars
    const yb = yOfCost(BUDGET);
    ctx.strokeStyle = '#d62728'; ctx.setLineDash([7, 4]); ctx.lineWidth = 1.6; ctx.beginPath(); ctx.moveTo(MARGIN.left, yb); ctx.lineTo(xLeft(COLS) + G.side, yb); ctx.stroke(); ctx.setLineDash([]); ctx.lineWidth = 1;
    // what the tail leaves: the budget minus the >= ceiling episodes' cost spread over every episode
    const left = BUDGET - p.tailPerEpisode, yl = yOfCost(left);
    ctx.strokeStyle = '#e07b00'; ctx.setLineDash([2, 3]); ctx.lineWidth = 1.6; ctx.beginPath(); ctx.moveTo(MARGIN.left, yl); ctx.lineTo(xLeft(COLS) + G.side, yl); ctx.stroke(); ctx.setLineDash([]); ctx.lineWidth = 1;
    ctx.fillStyle = '#e07b00'; ctx.font = 'bold 11px system-ui'; ctx.textAlign = 'right'; ctx.textBaseline = 'middle';
    ctx.fillText(left.toFixed(1), MARGIN.left - 5, yl + (yl > yb - 12 && yl < yb + 12 ? 10 : 0));
    // mean ×, median ○
    const px = g => MARGIN.left + (g + .5) * G.cell;
    ctx.strokeStyle = '#000'; ctx.lineWidth = 2;
    const mx = px(p.meanR), my = yOfCost(p.meanC), s = Math.max(5, G.cell * 0.35);
    ctx.beginPath(); ctx.moveTo(mx - s, my - s); ctx.lineTo(mx + s, my + s); ctx.moveTo(mx - s, my + s); ctx.lineTo(mx + s, my - s); ctx.stroke();
    ctx.beginPath(); ctx.arc(px(p.medR), yOfCost(p.medC), s, 0, 2 * Math.PI); ctx.stroke(); ctx.lineWidth = 1;
    // y labels (cost) at the strong lines, and the tail row
    ctx.fillStyle = '#333'; ctx.font = '11px system-ui'; ctx.textAlign = 'right'; ctx.textBaseline = 'middle';
    for (let r = 0; r <= BODY_ROWS; r += strongEvery) ctx.fillText(r * CC, MARGIN.left - 5, MARGIN.top + (ROWS - r) * G.cell);
    ctx.fillText(`≥${CEIL}`, MARGIN.left - 5, yTop(TAIL_ROW) + G.cell);
    ctx.fillStyle = '#d62728'; ctx.font = 'bold 11px system-ui'; ctx.fillText('budget', MARGIN.left - 5, yb - 9);
    // right: share of episodes per cost band (grey) with its number; the tail row also shows its share of all cost (red)
    ctx.font = small; ctx.textBaseline = 'middle'; ctx.textAlign = 'left';
    for (let r = 0; r < ROWS; r++) {
      const share = p.rows[r] / p.n, w = share / ROWMAX * (G.side - 38), y = yTop(r), h = rowHeight(r);
      if (r === TAIL_ROW) {
        const wc = Math.min(p.costShare[r] / ROWMAX * (G.side - 38), G.side - 40);
        ctx.fillStyle = '#9aa5b1'; ctx.fillRect(xLeft(COLS) + 2, y + 1, w, G.cell - 1);
        ctx.fillStyle = '#b56576'; ctx.fillRect(xLeft(COLS) + 2, y + G.cell, wc, G.cell - 1);
        ctx.fillStyle = '#333'; ctx.fillText(`${pct(share)} of episodes`, xLeft(COLS) + 5 + w, y + G.cell / 2);
        ctx.fillStyle = '#b56576'; ctx.fillText(`${pct(p.costShare[r], 0)} of all cost`, xLeft(COLS) + 5 + wc, y + 1.5 * G.cell);
      } else {
        ctx.fillStyle = '#9aa5b1'; ctx.fillRect(xLeft(COLS) + 2, y + 1, w, h - 2);
        if (share > 0 && h >= 9) { ctx.fillStyle = '#333'; ctx.fillText(pct(share), xLeft(COLS) + 5 + w, y + h / 2); }
      }
    }
    // below: share per goal count, bar hanging down from the axis, the goal count on the axis and the number under the bar
    const axisY = MARGIN.top + G.gridH + 14;
    ctx.fillStyle = '#333'; ctx.textAlign = 'center'; ctx.textBaseline = 'middle'; ctx.font = '11px system-ui';
    for (let c = 0; c < COLS; c++) if (c % 5 === 0 || G.cell >= 16) ctx.fillText(c, xLeft(c) + G.cell / 2, MARGIN.top + G.gridH + 7);
    const barMax = G.under - 46;
    const tiny = `${Math.max(7, Math.min(10, G.cell * 0.42))}px system-ui`;
    for (let c = 0; c < COLS; c++) {
      const share = p.cols[c] / p.n, h = share / COLMAX * barMax;
      ctx.fillStyle = '#9aa5b1'; ctx.fillRect(xLeft(c) + 1, axisY, G.cell - 2, h);
      if (share > 0 && G.cell >= 14) {
        // the number upright under its bar: whole percent when the cell is narrow, one decimal when wide
        ctx.fillStyle = '#333'; ctx.font = tiny; ctx.textAlign = 'center'; ctx.textBaseline = 'top';
        ctx.fillText(G.cell >= 26 ? pct(share) : Math.round(100 * share), xLeft(c) + G.cell / 2, axisY + h + 2);
      }
    }
    ctx.strokeStyle = '#333'; ctx.beginPath(); ctx.moveTo(MARGIN.left, axisY + .5); ctx.lineTo(xLeft(COLS), axisY + .5); ctx.stroke();
    ctx.fillStyle = '#555'; ctx.font = '11px system-ui'; ctx.textAlign = 'left'; ctx.textBaseline = 'alphabetic';
    ctx.fillText('share per cost band', xLeft(COLS) + 4, MARGIN.top + G.gridH + 12);
    ctx.textAlign = 'right'; ctx.fillText(`goals in the episode  ·  bars: share of episodes per goal count${G.cell < 26 ? ' (whole %)' : ''}`, xLeft(COLS), G.H - 4);
    // selection
    if (selection) {
      const { c0, c1, r0, r1 } = selection;
      const x = xLeft(c0), w = xLeft(c1 + 1) - x, y = yTop(r1), h = yTop(r0) + rowHeight(r0) - y;
      ctx.fillStyle = 'rgba(255,140,0,0.18)'; ctx.fillRect(x, y, w, h);
      ctx.strokeStyle = '#ff8c00'; ctx.lineWidth = 2; ctx.strokeRect(x + 1, y + 1, w - 2, h - 2); ctx.lineWidth = 1;
    }
  });
  describeSelection();
}

function cellAt(canvas, ev) {
  const rect = canvas.getBoundingClientRect();
  const x = ev.clientX - rect.left, y = ev.clientY - rect.top;
  const c = Math.floor((x - MARGIN.left) / G.cell);
  const r = y < yTop(TAIL_ROW) + rowHeight(TAIL_ROW) ? TAIL_ROW : ROWS - 1 - Math.floor((y - MARGIN.top) / G.cell);
  return { c: Math.max(0, Math.min(COLS - 1, c)), r: Math.max(0, Math.min(ROWS - 1, r)), inside: x >= MARGIN.left && x < xLeft(COLS) && y >= MARGIN.top && y < MARGIN.top + G.gridH };
}

function describeSelection() {
  const box = document.getElementById('selection');
  if (!selection) { box.innerHTML = '<b>Drag</b> a rectangle on a panel to read the share of each policy\'s episodes inside it. Click to clear.'; return; }
  const { c0, c1, r0, r1 } = selection;
  const costText = r1 === TAIL_ROW ? `${r0 * CC} and above` : `${r0 * CC} to ${(r1 + 1) * CC}`;
  let html = `<b>Selection</b><br>goals ${c0}${c1 > c0 ? '–' + c1 : ''}, cost ${costText}<table>`;
  policies.forEach(p => {
    let n = 0, sr = 0, sc = 0;
    for (let i = 0; i < p.n; i++) { const r = rowOf(p.costs[i]), c = colOf(p.returns[i]); if (c >= c0 && c <= c1 && r >= r0 && r <= r1) { n++; sr += p.returns[i]; sc += p.costs[i]; } }
    html += `<tr><th colspan="2">${p.label}</th></tr><tr><td class="big">${pct(n / p.n)}</td><td>${n} of ${p.n} episodes<br>mean inside: ${n ? (sr / n).toFixed(1) : '–'} goals, cost ${n ? (sc / n).toFixed(1) : '–'}</td></tr>`;
  });
  box.innerHTML = html + '</table>';
}

let dragStart = null, moved = false;
canvases.forEach(canvas => {
  canvas.addEventListener('mousedown', ev => { const cell = cellAt(canvas, ev); if (!cell.inside) return; dragStart = cell; moved = false; selection = { c0: cell.c, c1: cell.c, r0: cell.r, r1: cell.r }; draw(); });
  canvas.addEventListener('mousemove', ev => {
    const cell = cellAt(canvas, ev), hover = document.getElementById('hover');
    if (dragStart) { moved = true; selection = { c0: Math.min(dragStart.c, cell.c), c1: Math.max(dragStart.c, cell.c), r0: Math.min(dragStart.r, cell.r), r1: Math.max(dragStart.r, cell.r) }; draw(); }
    if (cell.inside) {
      const p = policies[+canvas.dataset.index], share = p.grid[cell.r][cell.c] / p.n;
      hover.style.display = 'block'; hover.style.left = (ev.clientX + 12) + 'px'; hover.style.top = (ev.clientY + 12) + 'px';
      hover.textContent = `goals ${cell.c}, cost ${costLabel(cell.r)}: ${pct(share, 2)} (${p.grid[cell.r][cell.c]} episodes)`
        + (cell.r === TAIL_ROW ? ` — this row: ${pct(p.costShare[TAIL_ROW], 0)} of all cost, worst ${p.tailWorst.toFixed(0)}` : '');
    } else hover.style.display = 'none';
  });
  canvas.addEventListener('mouseleave', () => { document.getElementById('hover').style.display = 'none'; });
});
window.addEventListener('mouseup', () => { if (dragStart && !moved) selection = null; dragStart = null; draw(); });
window.addEventListener('resize', draw);
document.getElementById('sqrt').addEventListener('change', ev => { SQRT = ev.target.checked; draw(); });
document.getElementById('numbers').addEventListener('change', ev => { NUMBERS = ev.target.checked; draw(); });
document.getElementById('title').textContent = `Episodes in the return–cost plane on ${DATA.target} (budget ${BUDGET})`;
document.getElementById('hint').textContent = `A cell is one goal wide and ${CC} cost units tall; its colour is the share of that policy's episodes in it, on one scale for every panel. The top row gathers every episode at cost ${CEIL} or above; its two bars are its share of episodes (grey) and of all cost (red). Other bars at the right: share of episodes per cost band; bars below, hanging from the axis: share per goal count. Dashed red: the budget. Dotted orange: the budget left for the other episodes once the ≥${CEIL} episodes' cost is spread over all of them. ×: mean. ○: median. Hover a cell for its share; drag to select a region.`;
draw();
</script>
</body></html>
"""


if __name__ == "__main__":
    main()
