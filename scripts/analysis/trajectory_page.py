"""An interactive page that replays an experiment's arms through the return–cost plane.

Every evaluation of every arm is one point (cost, return) on each evaluation target; the
page plays them in training order with a slider, a fading trace behind each arm, and the
budget line. One self-contained HTML file: the data is embedded as JSON and the drawing is
plain canvas, so it opens from disk with no server and no dependency (the review page's
approach). Two experiments can be shown side by side when they share arms.

    page = TrajectoryPage(title=..., targets=(("deployment", "Level 3 (w)"), ("uniform", "Uniform over Ω")))
    page.add_experiment("sensors only", frames_by_arm_label, colour_by_arm, marker_by_arm)
    page.write(path)
"""
from __future__ import annotations

import dataclasses
import json
import pathlib
from typing import Dict, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd

from scripts.analysis.goal_point_500M_figures import BUDGET, series

Marker = str  # "circle" | "square" | "cross"


@dataclasses.dataclass(frozen=True)
class ArmStyle:
    colour: str
    marker: Marker
    distribution: str
    learner: str


@dataclasses.dataclass
class TrajectoryPage:
    title: str
    targets: Sequence[Tuple[str, str]]  # (evaluation name, panel title)
    experiments: List[dict] = dataclasses.field(default_factory=list)

    def add_experiment(self, label: str, frames: Mapping[str, pd.DataFrame], styles: Mapping[str, ArmStyle]) -> None:
        """``frames``: one history per arm label; every arm's evaluation series on every target."""
        arms = []
        for arm_label, frame in frames.items():
            style = styles[arm_label]
            per_target = {}
            for evaluation, _ in self.targets:
                steps, returns = series(frame, f"evaluation/{evaluation}/episode_reward")
                _, costs = series(frame, f"evaluation/{evaluation}/episode_cost")
                if returns.shape != costs.shape:
                    raise ValueError(f"{arm_label}: {evaluation} has {returns.size} return and {costs.size} cost evaluations")
                per_target[evaluation] = {"steps": _round(steps), "return": _round(returns), "cost": _round(costs)}
            arms.append({"label": arm_label, "colour": style.colour, "marker": style.marker,
                         "distribution": style.distribution, "learner": style.learner, "targets": per_target})
        self.experiments.append({"label": label, "arms": arms})

    def write(self, path: pathlib.Path) -> None:
        payload = {"title": self.title, "budget": BUDGET, "targets": [{"name": name, "title": title} for name, title in self.targets],
                   "experiments": self.experiments}
        path.write_text(_HTML.replace("__DATA__", json.dumps(payload)).replace("__TITLE__", self.title))


def _round(values: np.ndarray) -> List[float]:
    return [round(float(v), 3) for v in values]


_HTML = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>__TITLE__</title>
<style>
 body{font:14px system-ui,sans-serif;margin:16px;color:#222}
 h1{font-size:16px;margin:0 0 8px}
 .controls{display:flex;gap:14px;align-items:center;margin:8px 0;flex-wrap:wrap}
 .controls label{display:flex;gap:6px;align-items:center}
 input[type=range]{width:420px}
 .panels{display:flex;gap:16px;flex-wrap:wrap}
 canvas{border:1px solid #ddd;background:#fff}
 .legend{display:flex;gap:18px;flex-wrap:wrap;margin-top:8px;font-size:12px}
 .legend span{display:inline-flex;align-items:center;gap:5px;cursor:pointer;user-select:none}
 .legend span.off{opacity:.3}
 .swatch{width:12px;height:12px;display:inline-block;border-radius:50%}
 #tip{position:fixed;pointer-events:none;background:#fff;border:1px solid #999;padding:4px 6px;font-size:12px;display:none}
</style></head><body>
<h1>__TITLE__</h1>
<div class="controls">
 <button id="play">▶ play</button>
 <input id="step" type="range" min="0" max="0" value="0">
 <span id="where"></span>
 <label>trace <input id="trace" type="range" min="0" max="20" value="4" style="width:90px"> <span id="tracen">4</span> evaluations</label>
 <label>show <select id="which"></select></label>
</div>
<div class="panels" id="panels"></div>
<div class="legend" id="legend"></div>
<div id="tip"></div>
<script>
const DATA = __DATA__;
const MARKERS = {circle:(c,x,y,r)=>{c.beginPath();c.arc(x,y,r,0,7);}, square:(c,x,y,r)=>{c.beginPath();c.rect(x-r,y-r,2*r,2*r);},
                 cross:(c,x,y,r)=>{c.beginPath();c.moveTo(x-r,y-r);c.lineTo(x+r,y+r);c.moveTo(x-r,y+r);c.lineTo(x+r,y-r);}};
const W=520,H=420,PAD={l:52,r:14,t:28,b:40}, XMIN=3, XMAX=300, YMIN=0, YMAX=27;
const hidden = new Set();
let mode = "all";
const numEvaluations = Math.max(...DATA.experiments.flatMap(e => e.arms.flatMap(a => Object.values(a.targets).map(t => t.steps.length))));
const stepInput = document.getElementById("step"); stepInput.max = numEvaluations - 1; stepInput.value = numEvaluations - 1;
const which = document.getElementById("which");
for (const [value, label] of [["all","all experiments"], ...DATA.experiments.map((e,i)=>[String(i), e.label])]) {
  const o = document.createElement("option"); o.value = value; o.textContent = label; which.appendChild(o);
}
const canvases = DATA.targets.map(t => { const c = document.createElement("canvas"); c.width = W; c.height = H; c.dataset.target = t.name; document.getElementById("panels").appendChild(c); return c; });
const xOf = cost => PAD.l + (Math.log(Math.max(cost, XMIN)) - Math.log(XMIN)) / (Math.log(XMAX) - Math.log(XMIN)) * (W - PAD.l - PAD.r);
const yOf = ret => H - PAD.b - (Math.min(Math.max(ret, YMIN), YMAX) - YMIN) / (YMAX - YMIN) * (H - PAD.t - PAD.b);
function axes(c, title) {
  c.clearRect(0,0,W,H); c.fillStyle="#222"; c.font="13px system-ui"; c.fillText(title, PAD.l, 18);
  c.strokeStyle="#ccc"; c.lineWidth=1; c.font="11px system-ui"; c.fillStyle="#555";
  for (const v of [5,10,25,50,100,200]) { const x=xOf(v); c.beginPath(); c.moveTo(x,PAD.t); c.lineTo(x,H-PAD.b); c.stroke(); c.fillText(String(v), x-6, H-PAD.b+14); }
  for (const v of [0,5,10,15,20,25]) { const y=yOf(v); c.beginPath(); c.moveTo(PAD.l,y); c.lineTo(W-PAD.r,y); c.stroke(); c.fillText(String(v), PAD.l-22, y+4); }
  c.fillStyle="rgba(128,128,128,.12)"; c.fillRect(PAD.l, yOf(5), W-PAD.l-PAD.r, yOf(0)-yOf(5));
  c.strokeStyle="#000"; c.setLineDash([5,4]); c.beginPath(); c.moveTo(xOf(DATA.budget),PAD.t); c.lineTo(xOf(DATA.budget),H-PAD.b); c.stroke(); c.setLineDash([]);
  c.fillStyle="#222"; c.fillText("cost per episode (log)", W/2-50, H-8); c.save(); c.translate(12, H/2+40); c.rotate(-Math.PI/2); c.fillText("goals per episode", 0, 0); c.restore();
}
const points = [];  // for hover: {cx, cy, text}
function draw() {
  const k = Number(stepInput.value), trace = Number(document.getElementById("trace").value);
  document.getElementById("tracen").textContent = trace;
  points.length = 0;
  canvases.forEach((canvas, panel) => {
    const c = canvas.getContext("2d"); axes(c, DATA.targets[panel].title);
    DATA.experiments.forEach((experiment, e) => {
      if (mode !== "all" && String(e) !== mode) return;
      const filled = DATA.experiments.length === 1 || e === DATA.experiments.length - 1;  // the latest experiment filled, earlier ones hollow
      for (const arm of experiment.arms) {
        if (hidden.has(arm.distribution) || hidden.has(arm.learner)) continue;
        const t = arm.targets[DATA.targets[panel].name]; const last = Math.min(k, t.steps.length - 1);
        c.strokeStyle = arm.colour; c.lineWidth = 1.2;
        for (let i = Math.max(1, last - trace + 1); i <= last; i++) {
          c.globalAlpha = 0.15 + 0.6 * (i - (last - trace)) / trace; c.beginPath(); c.moveTo(xOf(t.cost[i-1]), yOf(t.return[i-1])); c.lineTo(xOf(t.cost[i]), yOf(t.return[i])); c.stroke();
        }
        c.globalAlpha = 1; const x = xOf(t.cost[last]), y = yOf(t.return[last]);
        MARKERS[arm.marker](c, x, y, 6); c.lineWidth = 1.6;
        if (arm.marker === "cross") { c.strokeStyle = arm.colour; c.stroke(); }
        else if (filled) { c.fillStyle = arm.colour; c.fill(); c.strokeStyle = "#000"; c.stroke(); }
        else { c.strokeStyle = arm.colour; c.stroke(); }
        points.push({panel, x, y, text: `${experiment.label} · ${arm.label}\n${t.steps[last]} M: ${t.return[last]} goals, cost ${t.cost[last]}`});
      }
    });
  });
  const steps = DATA.experiments[0].arms[0].targets[DATA.targets[0].name].steps;
  document.getElementById("where").textContent = `evaluation ${k + 1} / ${numEvaluations} · ${steps[Math.min(k, steps.length - 1)]} M steps`;
}
function legend() {
  const box = document.getElementById("legend"); box.innerHTML = "";
  const seen = new Map();
  for (const arm of DATA.experiments[0].arms) { seen.set(arm.distribution, {colour: arm.colour}); seen.set(arm.learner, {marker: arm.marker}); }
  for (const [name, style] of seen) {
    const s = document.createElement("span"); s.textContent = name;
    const sw = document.createElement("i"); sw.className = "swatch";
    if (style.colour) sw.style.background = style.colour; else { sw.style.border = "2px solid #555"; sw.style.borderRadius = style.marker === "circle" ? "50%" : "0"; if (style.marker === "cross") sw.textContent = "×"; }
    s.prepend(sw); s.onclick = () => { hidden.has(name) ? hidden.delete(name) : hidden.add(name); s.classList.toggle("off"); draw(); };
    box.appendChild(s);
  }
  if (DATA.experiments.length > 1) { const n = document.createElement("span"); n.textContent = `hollow: ${DATA.experiments[0].label} · filled: ${DATA.experiments[DATA.experiments.length-1].label}`; box.appendChild(n); }
}
let timer = null;
document.getElementById("play").onclick = () => {
  if (timer) { clearInterval(timer); timer = null; document.getElementById("play").textContent = "▶ play"; return; }
  if (Number(stepInput.value) >= numEvaluations - 1) stepInput.value = 0;
  document.getElementById("play").textContent = "❚❚ pause";
  timer = setInterval(() => { if (Number(stepInput.value) >= numEvaluations - 1) { clearInterval(timer); timer = null; document.getElementById("play").textContent = "▶ play"; return; } stepInput.value = Number(stepInput.value) + 1; draw(); }, 700);
};
stepInput.oninput = draw; document.getElementById("trace").oninput = draw; which.onchange = () => { mode = which.value; draw(); };
const tip = document.getElementById("tip");
canvases.forEach((canvas, panel) => canvas.onmousemove = ev => {
  const r = canvas.getBoundingClientRect(); const mx = ev.clientX - r.left, my = ev.clientY - r.top;
  const hit = points.filter(p => p.panel === panel && Math.hypot(p.x - mx, p.y - my) < 9);
  if (!hit.length) { tip.style.display = "none"; return; }
  tip.style.display = "block"; tip.style.left = (ev.clientX + 12) + "px"; tip.style.top = (ev.clientY + 12) + "px"; tip.innerText = hit.map(p => p.text).join("\n\n");
});
legend(); draw();
</script></body></html>
"""
