// The review page: find an episode in three choices, watch it, see what the learner
// thought at every step, compare two moments (docs/acl/design/episode_review.md §1).
// The server hands over records; nothing is computed here beyond running sums.

const $ = (id) => document.getElementById(id);
const ROWS = [
  { title: "cumulated", series: [["return so far", "reward_cumulative", "#2ca02c"], ["cost so far", "cost_cumulative", "#d62728"]] },
  { title: "per step", series: [["reward", "reward", "#2ca02c"], ["cost", "cost", "#d62728"]] },
  { title: "critics", series: [["V (reward critic)", "value", "#2ca02c"], ["V_c (cost critic)", "cost_value", "#d62728"]] },
  { title: "advantages, unstandardised", series: [["A_r", "reward_advantage", "#2ca02c"], ["A_c", "cost_advantage", "#d62728"], ["A_r − λ A_c", "lagrangian_advantage", "#1f77b4"]] },
  { title: "λ", series: [["λ", "lagrange_multiplier", "#9467bd"]] },
  { title: "action", series: [] },  // filled per action dimension
];

const state = {
  library: null,
  viewers: [],      // [{ summary, episode, article, plot }]; one, or two when comparing
  step: 0,
  playing: null,    // interval handle
};

// ------------------------------------------------------------------ library and pickers

async function loadLibrary() {
  state.library = await (await fetch("/api/library")).json();
  $("group").textContent = state.library.group;
  const arms = unique(state.library.episodes.map((e) => e.arm));
  fill($("arm"), arms, arms);
  const cameras = unique(Object.values(state.library.environments).flatMap((env) => env.cameras));
  fill($("camera"), cameras, cameras, state.library.default_camera);
  refreshPickers();
}

function refreshPickers() {
  const arm = $("arm").value;
  const ofArm = state.library.episodes.filter((e) => e.arm === arm);
  const steps = unique(ofArm.map((e) => e.environment_steps)).sort((a, b) => a - b);
  fill($("evaluation"), steps, steps.map((s) => `${s.toLocaleString()} steps`), $("evaluation").value);
  fill($("compare"), ["", ...steps], ["—", ...steps.map((s) => `${s.toLocaleString()} steps`)], $("compare").value);
  const targets = unique(ofArm.map((e) => e.evaluation_name));
  fill($("target"), targets, targets, $("target").value);
  refreshEpisodeList();
}

function refreshEpisodeList() {
  const arm = $("arm").value, step = Number($("evaluation").value), target = $("target").value;
  const rows = state.library.episodes.filter((e) => e.arm === arm && e.environment_steps === step && e.evaluation_name === target);
  const body = $("episode-rows");
  body.innerHTML = "";
  for (const e of rows) {
    const tr = document.createElement("tr");
    tr.innerHTML = `<td>${e.episode_index}</td><td>${e.episode_return.toFixed(2)}</td><td>${e.episode_cost.toFixed(1)}</td>` +
      `<td>${e.length}</td><td>${e.context.map((v) => +v.toFixed(3)).join(", ")}</td>`;
    tr.onclick = () => openEpisode(e);
    if (state.viewers[0] && state.viewers[0].summary.index === e.index) tr.classList.add("selected");
    body.appendChild(tr);
  }
}

// ------------------------------------------------------------------ opening episodes

async function openEpisode(summary) {
  stop();
  const episodes = [summary];
  const compareStep = $("compare").value;
  if (compareStep !== "" && Number(compareStep) !== summary.environment_steps) {
    // The same arm, target and episode index at the other moment: the two differ only in training progress.
    const twin = state.library.episodes.find((e) =>
      e.arm === summary.arm && e.evaluation_name === summary.evaluation_name &&
      e.episode_index === summary.episode_index && e.environment_steps === Number(compareStep));
    if (twin) episodes.push(twin);
  }
  state.viewers = await Promise.all(episodes.map(async (s) => ({ summary: s, episode: await (await fetch(`/api/episode/${s.index}`)).json() })));
  $("viewers").innerHTML = "";
  for (const viewer of state.viewers) buildViewer(viewer);
  $("controls").hidden = false;
  $("timeline").max = Math.max(...state.viewers.map((v) => v.episode.length)) - 1;
  refreshEpisodeList();
  for (const tr of $("episode-rows").children) {
    const index = Number(tr.firstChild.textContent);
    if (state.viewers[1] && state.viewers[1].summary.episode_index === index) tr.classList.add("compared");
  }
  setStep(0);
}

function buildViewer(viewer) {
  const { episode } = viewer;
  const article = $("viewer-template").content.firstElementChild.cloneNode(true);
  $("viewers").appendChild(article);
  viewer.article = article;
  const lambda = episode.lagrange_multiplier[0];
  const constantLambda = episode.lagrange_multiplier.every((v) => Math.abs(v - lambda) < 1e-6);
  article.querySelector(".facts").innerHTML =
    `<b>${episode.arm}</b> · ${episode.evaluation_name} · after <b>${episode.environment_steps.toLocaleString()}</b> steps · episode ${episode.episode_index}<br>` +
    `ω = [${episode.context.map((v) => +v.toFixed(3)).join(", ")}] · λ = ${constantLambda ? lambda.toFixed(3) : "varies (see trace)"} · ` +
    `return <b>${episode.episode_return.toFixed(2)}</b> · cost <b>${episode.episode_cost.toFixed(1)}</b> · ${episode.length} steps`;
  episode.reward_cumulative = cumulative(episode.reward);
  episode.cost_cumulative = cumulative(episode.cost);
  buildTraces(viewer);
  article.querySelector(".frame").onclick = togglePlay;
}

function buildTraces(viewer) {
  const { episode, article } = viewer;
  const x = [...episode.reward.keys()];
  const traces = [], layout = { height: 160 * ROWS.length, margin: { l: 56, r: 12, t: 8, b: 32 }, showlegend: true, legend: { orientation: "h", y: -0.04 },
    grid: { rows: ROWS.length, columns: 1, pattern: "independent", roworder: "top to bottom" }, hovermode: "x unified", shapes: [] };
  ROWS.forEach((row, r) => {
    const axis = r === 0 ? "" : r + 1;
    const series = row.title === "action"
      ? episode.action[0].map((_, d) => [`action ${d}`, (t) => episode.action[t][d], null])
      : row.series.map(([name, key, colour]) => [name, (t) => episode[key][t], colour]);
    for (const [name, at, colour] of series) {
      traces.push({ x, y: x.map(at), name, type: "scatter", mode: "lines", xaxis: `x${axis}`, yaxis: `y${axis}`,
        line: colour ? { color: colour, width: 1.5 } : { width: 1.5 }, legendgroup: row.title });
    }
    layout[`xaxis${axis}`] = { matches: r === 0 ? undefined : "x", showticklabels: r === ROWS.length - 1, title: r === ROWS.length - 1 ? "step" : undefined };
    layout[`yaxis${axis}`] = { title: { text: row.title, font: { size: 11 } }, zeroline: true };
    layout.shapes.push({ type: "line", xref: `x${axis}`, yref: "paper", x0: 0, x1: 0, y0: 0, y1: 1, line: { color: "#444", width: 1, dash: "dot" } });
  });
  const div = article.querySelector(".traces");
  Plotly.newPlot(div, traces, layout, { displayModeBar: false, responsive: true });
  div.on("plotly_click", (event) => setStep(event.points[0].x));
  viewer.plot = div;
}

// ------------------------------------------------------------------ the timeline

function setStep(step) {
  const last = Number($("timeline").max);
  state.step = Math.max(0, Math.min(last, step));
  $("timeline").value = state.step;
  $("step-counter").textContent = `step ${state.step} / ${last}`;
  const camera = $("camera").value;
  for (const viewer of state.viewers) {
    const t = Math.min(state.step, viewer.episode.length - 1);  // a shorter episode holds its last frame
    viewer.article.querySelector(".frame").src = `/frame/${viewer.summary.index}/${t}?camera=${camera}`;
    viewer.article.querySelector(".running").innerHTML =
      `<span>step ${t}</span><span class="reward">return so far ${viewer.episode.reward_cumulative[t].toFixed(2)}</span>` +
      `<span class="cost">cost so far ${viewer.episode.cost_cumulative[t].toFixed(1)}</span>` +
      `<span>V ${viewer.episode.value[t].toFixed(3)}</span><span>V_c ${viewer.episode.cost_value[t].toFixed(3)}</span>` +
      `<span>A_r − λA_c ${viewer.episode.lagrangian_advantage[t].toFixed(3)}</span>`;
    Plotly.relayout(viewer.plot, Object.fromEntries(ROWS.map((_, r) => [[`shapes[${r}].x0`, t], [`shapes[${r}].x1`, t]]).flat()));
    prefetch(viewer, t, camera);
  }
}

function prefetch(viewer, from, camera) {
  for (let t = from + 1; t <= Math.min(from + 6, viewer.episode.length - 1); t++) {
    const image = new Image();
    image.src = `/frame/${viewer.summary.index}/${t}?camera=${camera}`;
  }
}

function controlDt() {
  const env = state.library.environments[state.viewers[0].episode.env_name];
  return env.control_dt;
}

function togglePlay() { state.playing ? stop() : play(); }

function play() {
  if (!state.viewers.length) return;
  const period = 1000 * controlDt() / Number($("speed").value);
  state.playing = setInterval(() => {
    if (state.step >= Number($("timeline").max)) return stop();
    setStep(state.step + 1);
  }, period);
  $("play").textContent = "⏸";
}

function stop() {
  if (state.playing) clearInterval(state.playing);
  state.playing = null;
  $("play").textContent = "▶";
}

// ------------------------------------------------------------------ wiring

function fill(select, values, labels, keep) {
  select.innerHTML = "";
  values.forEach((value, i) => {
    const option = document.createElement("option");
    option.value = value; option.textContent = labels[i];
    select.appendChild(option);
  });
  if (keep !== undefined && values.map(String).includes(String(keep))) select.value = keep;
}

const unique = (values) => [...new Set(values)];
const cumulative = (values) => values.reduce((sums, v) => (sums.push((sums.length ? sums[sums.length - 1] : 0) + v), sums), []);

// A picker releases focus after a choice, so the arrow keys step the episode instead of the picker.
const picker = (id, action) => { $(id).onchange = (event) => { action(); event.target.blur(); }; };
picker("arm", refreshPickers);
picker("evaluation", refreshEpisodeList);
picker("target", refreshEpisodeList);
picker("compare", () => state.viewers[0] && openEpisode(state.viewers[0].summary));
picker("camera", () => setStep(state.step));
picker("speed", () => { if (state.playing) { stop(); play(); } });
$("timeline").oninput = (event) => { stop(); setStep(Number(event.target.value)); };
$("play").onclick = togglePlay;
$("back").onclick = () => { stop(); setStep(state.step - 1); };
$("forward").onclick = () => { stop(); setStep(state.step + 1); };
$("to-start").onclick = () => { stop(); setStep(0); };
$("to-end").onclick = () => { stop(); setStep(Number($("timeline").max)); };
document.addEventListener("keydown", (event) => {
  if (event.target.tagName === "SELECT" || !state.viewers.length) return;
  if (event.key === "ArrowLeft") { stop(); setStep(state.step - 1); event.preventDefault(); }
  if (event.key === "ArrowRight") { stop(); setStep(state.step + 1); event.preventDefault(); }
  if (event.key === "Home") { stop(); setStep(0); event.preventDefault(); }
  if (event.key === "End") { stop(); setStep(Number($("timeline").max)); event.preventDefault(); }
  if (event.key === " ") { togglePlay(); event.preventDefault(); }
});

loadLibrary();
