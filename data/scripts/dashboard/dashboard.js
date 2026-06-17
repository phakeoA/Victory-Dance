/* Victory-Dance — Self-Play Dashboard (3c.6).
   Standalone: open dashboard.html directly (file://) and the embedded DEMO_MANIFEST
   renders immediately. "Load manifest.json" reads a real
   artifacts/self_play_archive/manifest.json (the build_manifest schema). No build step,
   no external libraries — charts are hand-drawn inline SVG so it works fully offline. */

"use strict";

// ── Embedded demo (a 12-gen run with a collapse/revert at gen 7 + recovery) ──
// Exactly the shape v_dance.selfplay.archive.build_manifest() emits.
const DEMO_MANIFEST = {
  "n_generations": 12, "best_path": "gen11.pt", "best_generation": 11,
  "best_win_rate": 0.6583333333333333, "best_elo": 1278.0, "n_promotions": 9,
  "league": ["gen0","gen1","gen2","gen3","gen5","gen6","gen9","gen10","gen11"],
  "generations": [
    {"generation":0,"checkpoint":"gen0.pt","scripted_win_rate":0.3833,"model_elo":980,"verdict":"promote","promoted":true,"n_trajectories":400,"update_stats":{"loss":0.62,"kl_to_bc":0.0018,"explained_variance":0.78,"clip_fraction":0.02,"entropy":1.30,"halted":0},"elo_delta":null,"win_rate_delta":null,"is_best":true},
    {"generation":1,"checkpoint":"gen1.pt","scripted_win_rate":0.4417,"model_elo":1030,"verdict":"promote","promoted":true,"n_trajectories":402,"update_stats":{"loss":0.55,"kl_to_bc":0.0024,"explained_variance":0.82,"clip_fraction":0.025,"entropy":1.24,"halted":0},"elo_delta":50,"win_rate_delta":0.0583,"is_best":true},
    {"generation":2,"checkpoint":"gen2.pt","scripted_win_rate":0.4917,"model_elo":1075,"verdict":"promote","promoted":true,"n_trajectories":410,"update_stats":{"loss":0.50,"kl_to_bc":0.0031,"explained_variance":0.85,"clip_fraction":0.028,"entropy":1.18,"halted":0},"elo_delta":45,"win_rate_delta":0.05,"is_best":true},
    {"generation":3,"checkpoint":"gen3.pt","scripted_win_rate":0.5167,"model_elo":1110,"verdict":"promote","promoted":true,"n_trajectories":408,"update_stats":{"loss":0.47,"kl_to_bc":0.0029,"explained_variance":0.86,"clip_fraction":0.031,"entropy":1.12,"halted":0},"elo_delta":35,"win_rate_delta":0.025,"is_best":true},
    {"generation":4,"checkpoint":"gen4.pt","scripted_win_rate":0.5083,"model_elo":1098,"verdict":"hold","promoted":false,"n_trajectories":406,"update_stats":{"loss":0.46,"kl_to_bc":0.0035,"explained_variance":0.84,"clip_fraction":0.034,"entropy":1.09,"halted":0},"elo_delta":-12,"win_rate_delta":-0.0083,"is_best":false},
    {"generation":5,"checkpoint":"gen5.pt","scripted_win_rate":0.5667,"model_elo":1155,"verdict":"promote","promoted":true,"n_trajectories":415,"update_stats":{"loss":0.42,"kl_to_bc":0.0042,"explained_variance":0.87,"clip_fraction":0.030,"entropy":1.04,"halted":0},"elo_delta":57,"win_rate_delta":0.0583,"is_best":true},
    {"generation":6,"checkpoint":"gen6.pt","scripted_win_rate":0.60,"model_elo":1190,"verdict":"promote","promoted":true,"n_trajectories":420,"update_stats":{"loss":0.40,"kl_to_bc":0.0048,"explained_variance":0.88,"clip_fraction":0.033,"entropy":0.99,"halted":0},"elo_delta":35,"win_rate_delta":0.0333,"is_best":true},
    {"generation":7,"checkpoint":"gen7.pt","scripted_win_rate":0.4917,"model_elo":1100,"verdict":"revert","promoted":false,"n_trajectories":418,"update_stats":{"loss":0.58,"kl_to_bc":0.0125,"explained_variance":0.55,"clip_fraction":0.090,"entropy":0.82,"halted":1},"elo_delta":-90,"win_rate_delta":-0.1083,"is_best":false},
    {"generation":8,"checkpoint":"gen8.pt","scripted_win_rate":0.5833,"model_elo":1165,"verdict":"hold","promoted":false,"n_trajectories":416,"update_stats":{"loss":0.41,"kl_to_bc":0.0044,"explained_variance":0.86,"clip_fraction":0.031,"entropy":0.98,"halted":0},"elo_delta":65,"win_rate_delta":0.0917,"is_best":false},
    {"generation":9,"checkpoint":"gen9.pt","scripted_win_rate":0.6167,"model_elo":1205,"verdict":"promote","promoted":true,"n_trajectories":422,"update_stats":{"loss":0.39,"kl_to_bc":0.0050,"explained_variance":0.89,"clip_fraction":0.029,"entropy":0.95,"halted":0},"elo_delta":40,"win_rate_delta":0.0333,"is_best":true},
    {"generation":10,"checkpoint":"gen10.pt","scripted_win_rate":0.6417,"model_elo":1240,"verdict":"promote","promoted":true,"n_trajectories":430,"update_stats":{"loss":0.38,"kl_to_bc":0.0052,"explained_variance":0.90,"clip_fraction":0.028,"entropy":0.93,"halted":0},"elo_delta":35,"win_rate_delta":0.025,"is_best":true},
    {"generation":11,"checkpoint":"gen11.pt","scripted_win_rate":0.6583,"model_elo":1278,"verdict":"promote","promoted":true,"n_trajectories":433,"update_stats":{"loss":0.37,"kl_to_bc":0.0055,"explained_variance":0.90,"clip_fraction":0.027,"entropy":0.91,"halted":0},"elo_delta":38,"win_rate_delta":0.0167,"is_best":true}
  ]
};

const STATE = { m: null, sel: null, tab: "overview", src: "demo data",
                mode: "static", status: null, lastKey: null, lastTags: "", poll: null,
                liveLog: null, logKey: "", viewTurn: 0, followLatest: true };

// ── tiny helpers ──────────────────────────────────────────────────────────────
const $ = (id) => document.getElementById(id);
const pct = (x) => (x == null ? "—" : (x * 100).toFixed(1) + "%");
const num = (x, d = 0) => (x == null ? "—" : Number(x).toFixed(d));
const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
function deltaSpan(d, asPct) {
  if (d == null) return '<span class="delta delta-flat">—</span>';
  const cls = d > 1e-9 ? "delta-up" : d < -1e-9 ? "delta-down" : "delta-flat";
  const arrow = d > 1e-9 ? "▲" : d < -1e-9 ? "▼" : "–";
  const v = asPct ? (Math.abs(d) * 100).toFixed(1) + "%" : Math.abs(d).toFixed(asPct === false ? 0 : 1);
  return `<span class="delta ${cls}">${arrow} ${v}</span>`;
}
function notify(msg, err) {
  const n = $("notif"); n.textContent = msg; n.className = err ? "err" : "";
  n.style.display = "block"; setTimeout(() => (n.style.display = "none"), 2600);
}

// ── SVG line chart (hand-drawn, no libs) ───────────────────────────────────────
function lineChart(opts) {
  const W = opts.width || 660, H = opts.height || 230;
  const mL = opts.mL || 48, mR = 14, mT = 12, mB = 26;
  const pw = W - mL - mR, ph = H - mT - mB;
  const data = (opts.data || []).filter((d) => d.y != null && isFinite(d.y));
  if (!data.length) return `<svg class="chart" viewBox="0 0 ${W} ${H}"><text x="${W / 2}" y="${H / 2}" text-anchor="middle" class="axlbl">no data</text></svg>`;
  const xs = data.map((d) => d.x), ys = data.map((d) => d.y);
  let xmin = Math.min(...xs), xmax = Math.max(...xs);
  let ymin = opts.yMin != null ? opts.yMin : Math.min(...ys);
  let ymax = opts.yMax != null ? opts.yMax : Math.max(...ys);
  if (ymax === ymin) { ymax += 1; ymin -= 1; }
  else { const pad = (ymax - ymin) * 0.08; ymin -= pad; ymax += pad; }
  if (opts.yMin != null) ymin = opts.yMin;
  if (opts.yMax != null) ymax = opts.yMax;
  const X = (x) => mL + (xmax === xmin ? pw / 2 : ((x - xmin) / (xmax - xmin)) * pw);
  const Y = (y) => mT + (1 - (y - ymin) / (ymax - ymin)) * ph;
  const color = opts.color || "var(--blue)";
  const fmtY = opts.fmtY || ((v) => v.toFixed(0));

  let svg = `<svg class="chart" viewBox="0 0 ${W} ${H}" preserveAspectRatio="xMidYMid meet">`;
  // y gridlines + labels
  const nTick = opts.yTicks || 4;
  for (let i = 0; i <= nTick; i++) {
    const v = ymin + (i / nTick) * (ymax - ymin), y = Y(v);
    svg += `<line class="grid" x1="${mL}" y1="${y.toFixed(1)}" x2="${W - mR}" y2="${y.toFixed(1)}" />`;
    svg += `<text class="axlbl" x="${mL - 6}" y="${(y + 3).toFixed(1)}" text-anchor="end">${esc(fmtY(v))}</text>`;
  }
  // x axis labels (gen numbers)
  data.forEach((d) => {
    svg += `<text class="axlbl" x="${X(d.x).toFixed(1)}" y="${H - 8}" text-anchor="middle">${d.x}</text>`;
  });
  // optional reference line (e.g. 50% win-rate)
  if (opts.refLine != null) {
    const y = Y(opts.refLine);
    svg += `<line class="refline" x1="${mL}" y1="${y.toFixed(1)}" x2="${W - mR}" y2="${y.toFixed(1)}" />`;
    if (opts.refLabel) svg += `<text class="reflbl" x="${W - mR}" y="${(y - 3).toFixed(1)}" text-anchor="end">${esc(opts.refLabel)}</text>`;
  }
  // line
  const path = data.map((d, i) => `${i ? "L" : "M"}${X(d.x).toFixed(1)},${Y(d.y).toFixed(1)}`).join(" ");
  svg += `<path class="lineseg" d="${path}" stroke="${color}" />`;
  // points
  data.forEach((d) => {
    const cls = "pt" + (d.isBest ? " pt-best" : "") + (d.halted ? " pt-halt" : "");
    const fill = d.halted ? "var(--red)" : d.promoted ? color : "var(--bg2)";
    svg += `<circle class="${cls}" cx="${X(d.x).toFixed(1)}" cy="${Y(d.y).toFixed(1)}" r="${d.isBest ? 5 : 4}" fill="${fill}" stroke="${color}" stroke-width="${d.isBest || d.halted ? 2 : 1}" onclick="selectGen(${d.x})" style="cursor:pointer"><title>${esc(d.tip || "")}</title></circle>`;
  });
  svg += `</svg>`;
  return svg;
}

// ── data accessors ─────────────────────────────────────────────────────────────
const gens = () => STATE.m.generations;
const stat = (g, k) => (g.update_stats && g.update_stats[k] != null ? g.update_stats[k] : null);
function series(key, getter, fmt) {
  return gens().map((g) => {
    const y = getter(g);
    return {
      x: g.generation, y, gen: g.generation, promoted: g.promoted,
      isBest: g.generation === STATE.m.best_generation, halted: stat(g, "halted") === 1,
      tip: `Gen ${g.generation}: ${y == null ? "—" : (fmt ? fmt(y) : String(y))}  ·  ${g.verdict}${g.promoted ? " ✓" : ""}`,
    };
  });
}

// ── notables (auto-generated highlights) ─────────────────────────────────────────
function computeNotables() {
  const gs = gens(), out = [];
  const last = gs[gs.length - 1], first = gs[0];
  const m = STATE.m;
  // overall improvement
  if (gs.length > 1) {
    const eloGain = (last.model_elo != null && first.model_elo != null) ? last.model_elo - first.model_elo : null;
    const wrGain = (last.scripted_win_rate != null && first.scripted_win_rate != null) ? last.scripted_win_rate - first.scripted_win_rate : null;
    const improved = (eloGain != null && eloGain > 0) || (wrGain != null && wrGain > 0);
    out.push({ kind: improved ? "good" : "warn", ico: improved ? "📈" : "📉",
      title: improved ? "Net improvement over the run" : "No net improvement yet",
      desc: `Gen 0 → Gen ${last.generation}: ${eloGain != null ? (eloGain >= 0 ? "+" : "") + eloGain.toFixed(0) + " Elo" : "—"}, ${wrGain != null ? (wrGain >= 0 ? "+" : "") + (wrGain * 100).toFixed(1) + "% scripted win-rate" : "—"}.` });
  }
  // best so far
  const best = gs.find((g) => g.generation === m.best_generation);
  if (best) out.push({ kind: "good", ico: "🏆", title: `Best generation: Gen ${best.generation}`,
    desc: `${pct(best.scripted_win_rate)} win-rate, Elo ${num(best.model_elo)} — current league anchor (${esc(m.best_path || "?")}).` });
  // biggest single-gen jump
  const jumps = gs.filter((g) => g.win_rate_delta != null);
  if (jumps.length) {
    const bj = jumps.reduce((a, b) => (b.win_rate_delta > a.win_rate_delta ? b : a));
    if (bj.win_rate_delta > 0) out.push({ kind: "info", ico: "🚀", title: `Biggest jump: Gen ${bj.generation}`,
      desc: `+${(bj.win_rate_delta * 100).toFixed(1)}% win-rate vs the previous generation${bj.elo_delta != null ? ` (${bj.elo_delta >= 0 ? "+" : ""}${bj.elo_delta.toFixed(0)} Elo)` : ""}.` });
  }
  // collapses / reverts
  const reverts = gs.filter((g) => g.verdict === "revert");
  reverts.forEach((g) => out.push({ kind: "bad", ico: "⚠️", title: `Collapse at Gen ${g.generation} — reverted`,
    desc: `Win-rate ${deltaTxt(g.win_rate_delta, true)}, Elo ${deltaTxt(g.elo_delta, false)}${stat(g, "halted") === 1 ? "; PPO update HALTED (guard tripped)" : ""}. Recovered from the previous best.` }));
  // halted updates that did NOT revert
  gs.filter((g) => stat(g, "halted") === 1 && g.verdict !== "revert").forEach((g) =>
    out.push({ kind: "warn", ico: "🛑", title: `Gen ${g.generation}: PPO update halted`, desc: "A collapse guard (KL-to-BC or explained-variance floor) stopped the update early." }));
  // promotions tally
  out.push({ kind: "info", ico: "✅", title: `${m.n_promotions} / ${m.n_generations} generations promoted`,
    desc: "Promotion = beat the accepted best on the scripted ladder beyond noise (two-proportion gate)." });
  return out;
}
function deltaTxt(d, asPct) {
  if (d == null) return "—";
  const s = d >= 0 ? "+" : "";
  return asPct ? `${s}${(d * 100).toFixed(1)}%` : `${s}${d.toFixed(0)}`;
}

// ── renderers ────────────────────────────────────────────────────────────────
function renderAll() {
  $("src-pill").textContent = STATE.src;
  renderSummary(); renderGenList();
  if (gens().length) { renderOverview(); renderHealth(); renderDetail(); }
  else { renderEmptyMain(); }
  renderSpectate(); renderLiveBar(); setLiveBadge();
  switchTab(STATE.tab);
}

function renderEmptyMain() {
  $("tab-overview").innerHTML = `<div id="empty"><div class="big">⏳</div><p>Waiting for the first generation to finish…<br/>The charts and notables fill in automatically as the self-play run produces data.</p></div>`;
  $("tab-health").innerHTML = ""; $("tab-detail").innerHTML = "";
}

function renderSummary() {
  const m = STATE.m;
  $("gen-count").textContent = `(${m.n_generations})`;
  $("run-summary").innerHTML = [
    ["Best Elo", num(m.best_elo), `gen ${m.best_generation ?? "—"}`],
    ["Best win-rate", pct(m.best_win_rate), `gen ${m.best_generation ?? "—"}`],
    ["Generations", m.n_generations, `${m.n_promotions} promoted`],
    ["League pool", (m.league || []).length, "snapshots"],
  ].map(([l, v, s]) => `<div class="mini"><div class="mv">${esc(v)}</div><div class="ml">${esc(l)}</div><div class="ms">${esc(s)}</div></div>`).join("");
}

function renderGenList() {
  $("gen-list").innerHTML = gens().slice().reverse().map((g) => {
    const isBest = g.generation === STATE.m.best_generation;
    const active = g.generation === STATE.sel ? " active" : "";
    const best = isBest ? " best" : "";
    return `<div class="gen-item${active}${best}" onclick="selectGen(${g.generation})">
      <div class="gi-num">G${g.generation}${isBest ? ' <span class="star">★</span>' : ""}</div>
      <div class="gi-mid"><div class="gi-wr">${pct(g.scripted_win_rate)}</div><div class="gi-elo">Elo ${num(g.model_elo)}</div></div>
      <div class="gi-right"><span class="vb vb-${g.verdict}">${esc(g.verdict)}</span>${deltaSpan(g.win_rate_delta, true)}</div>
    </div>`;
  }).join("");
}

function renderOverview() {
  const m = STATE.m, gs = gens(), last = gs[gs.length - 1];
  const cards = [
    ["c-yellow", num(m.best_elo), "Best Elo", `gen ${m.best_generation}`],
    ["c-green", pct(m.best_win_rate), "Best scripted win-rate", `gen ${m.best_generation}`],
    ["c-accent", `${m.n_promotions}/${m.n_generations}`, "Generations promoted", "passed the gate"],
    ["c-purple", pct(last.scripted_win_rate), `Latest (gen ${last.generation})`, `${last.verdict} ${last.win_rate_delta != null ? deltaTxt(last.win_rate_delta, true) : ""}`],
  ].map(([c, v, l, s]) => `<div class="scard ${c}"><div class="v">${esc(v)}</div><div class="l">${esc(l)}</div><div class="s">${esc(s)}</div></div>`).join("");

  const eloChart = lineChart({ data: series("Elo", (g) => g.model_elo, (v) => "Elo " + v.toFixed(0)), color: "var(--yellow)",
    fmtY: (v) => v.toFixed(0), yTicks: 4 });
  const wrChart = lineChart({ data: series("win-rate", (g) => g.scripted_win_rate, (v) => (v * 100).toFixed(1) + "% win-rate"), color: "var(--green)",
    yMin: 0, yMax: 1, refLine: 0.5, refLabel: "50% (even vs scripted)", fmtY: (v) => (v * 100).toFixed(0) + "%", yTicks: 4 });

  const notes = computeNotables().map((n) =>
    `<div class="note ${n.kind}"><div class="ico">${n.ico}</div><div><div class="nt"><b>${esc(n.title)}</b></div><div class="nd">${esc(n.desc)}</div></div></div>`).join("");

  $("tab-overview").innerHTML = `
    <div class="cards">${cards}</div>
    <div class="panel"><div class="panel-hdr"><h2>📊 Notable events &amp; improvements</h2><span class="hint">click any point or generation to inspect it</span></div><div class="panel-body"><div class="notables">${notes}</div></div></div>
    <div class="panel"><div class="panel-hdr"><h2>Elo vs scripted ladder</h2><div class="legend"><span class="lg-item"><span class="lg-dot" style="background:var(--yellow)"></span>Elo</span><span class="lg-item"><span class="lg-dot" style="background:var(--yellow);border:2px solid var(--yellow)"></span>★ best</span></div></div><div class="panel-body">${eloChart}</div></div>
    <div class="panel"><div class="panel-hdr"><h2>Scripted win-rate</h2><div class="legend"><span class="lg-item"><span class="lg-dot" style="background:var(--green)"></span>win-rate</span><span class="lg-item"><span class="lg-dot" style="background:var(--red)"></span>halted</span></div></div><div class="panel-body">${wrChart}</div></div>`;
}

const HEALTH_METRICS = [
  { key: "loss", name: "Total PPO loss", desc: "lower = converging", color: "var(--blue)", fmt: (v) => v.toFixed(3) },
  { key: "kl_to_bc", name: "KL to BC prior", desc: "spikes ⇒ drifting off the cloned policy (collapse risk)", color: "var(--purple)", fmt: (v) => v.toFixed(4) },
  { key: "explained_variance", name: "Critic explained variance", desc: "want ≥ 0.7 — value head fit", color: "var(--green)", fmt: (v) => v.toFixed(2), yMin: 0, yMax: 1 },
  { key: "clip_fraction", name: "Clip fraction", desc: "fraction of ratios clipped — high ⇒ big steps", color: "var(--orange)", fmt: (v) => v.toFixed(3) },
  { key: "entropy", name: "Policy entropy", desc: "exploration; gentle decline is normal", color: "var(--yellow)", fmt: (v) => v.toFixed(2) },
];

function renderHealth() {
  const charts = HEALTH_METRICS.map((mt) => {
    const data = series(mt.name, (g) => stat(g, mt.key), mt.fmt);
    const cur = data.length ? data[data.length - 1].y : null;
    const chart = lineChart({ data, color: mt.color, height: 170, mL: 52,
      fmtY: mt.fmt, yMin: mt.yMin, yMax: mt.yMax, yTicks: 3,
      refLine: mt.key === "explained_variance" ? 0.7 : null, refLabel: mt.key === "explained_variance" ? "0.7 floor" : null });
    return `<div class="mini-chart"><div class="mc-hdr"><span class="mc-name">${esc(mt.name)}</span><span class="mc-cur">now: ${cur == null ? "—" : esc(mt.fmt(cur))}</span></div><div class="mc-desc">${esc(mt.desc)}</div>${chart}</div>`;
  }).join("");
  const halted = gens().filter((g) => stat(g, "halted") === 1).map((g) => `Gen ${g.generation}`);
  const haltNote = halted.length
    ? `<div class="note bad" style="margin-bottom:14px"><div class="ico">🛑</div><div><div class="nt"><b>Halted updates</b></div><div class="nd">${esc(halted.join(", "))} — a collapse guard (KL-to-BC ceiling or explained-variance floor) stopped the PPO update early. Red points below mark them.</div></div></div>`
    : `<div class="note good" style="margin-bottom:14px"><div class="ico">✅</div><div><div class="nt"><b>No halted updates</b></div><div class="nd">Every generation completed its PPO update without tripping a collapse guard.</div></div></div>`;
  $("tab-health").innerHTML = `<div class="panel"><div class="panel-hdr"><h2>⚙️ PPO / critic health per generation</h2><span class="hint">§7 training diagnostics — red point = update halted</span></div><div class="panel-body">${haltNote}<div class="chart-grid">${charts}</div></div></div>`;
}

function renderDetail() {
  const g = gens().find((x) => x.generation === STATE.sel);
  if (!g) { $("tab-detail").innerHTML = ""; return; }
  const m = STATE.m;
  const idx = gens().indexOf(g);
  const isBest = g.generation === m.best_generation;
  const vsBest = (g.scripted_win_rate != null && m.best_win_rate != null) ? g.scripted_win_rate - m.best_win_rate : null;

  const kv = [
    ["Scripted win-rate", pct(g.scripted_win_rate), deltaSpan(g.win_rate_delta, true) + " vs prev"],
    ["Model Elo", num(g.model_elo), deltaSpan(g.elo_delta, false) + " vs prev"],
    ["Trajectories", num(g.n_trajectories), "collected this gen"],
    ["vs best win-rate", vsBest == null ? "—" : (vsBest >= 0 ? "+" : "") + (vsBest * 100).toFixed(1) + "%", isBest ? '<span class="star">★ this is the best</span>' : `best is gen ${m.best_generation}`],
  ].map(([k, v, d]) => `<div class="kv"><div class="k">${esc(k)}</div><div class="v">${esc(v)}</div><div class="d">${d}</div></div>`).join("");

  // health bars (normalised within each metric's own history so bars are comparable)
  const maxOf = (key) => Math.max(...gens().map((x) => stat(x, key) || 0), 1e-9);
  const bars = HEALTH_METRICS.map((mt) => {
    const v = stat(g, mt.key);
    if (v == null) return "";
    const max = mt.yMax != null ? mt.yMax : maxOf(mt.key);
    const w = Math.max(2, Math.min(100, (v / max) * 100));
    let warn = false;
    if (mt.key === "kl_to_bc" && v > 0.01) warn = true;
    if (mt.key === "explained_variance" && v < 0.7) warn = true;
    if (mt.key === "clip_fraction" && v > 0.08) warn = true;
    const flag = warn ? '<span class="hbar-flag warn">check</span>' : '<span class="hbar-flag">ok</span>';
    return `<div class="hbar-row"><div class="hbar-lbl">${esc(mt.name)}</div><div class="hbar-track"><div class="hbar-fill" style="width:${w.toFixed(0)}%;background:${mt.color}"></div></div><div class="hbar-val">${esc(mt.fmt(v))}</div>${flag}</div>`;
  }).join("");
  const halted = stat(g, "halted") === 1;

  // a per-gen notable
  let note;
  if (g.verdict === "revert") note = { kind: "bad", ico: "⚠️", t: "Collapse — reverted to the previous best", d: `Win-rate fell ${deltaTxt(g.win_rate_delta, true)} and Elo ${deltaTxt(g.elo_delta, false)}.${halted ? " The PPO update was halted by a collapse guard." : ""}` };
  else if (isBest) note = { kind: "good", ico: "🏆", t: "Best generation so far", d: "Highest scripted win-rate to date — promoted into the league as the new anchor." };
  else if (g.promoted) note = { kind: "good", ico: "✅", t: "Promoted", d: "Beat the accepted best on the scripted ladder beyond noise." };
  else note = { kind: "info", ico: "•", t: `Held (verdict: ${g.verdict})`, d: "Did not beat the accepted best beyond the noise gate; best model unchanged." };

  $("tab-detail").innerHTML = `
    <div class="detail-hdr">
      <span class="dg">Generation ${g.generation}</span>
      <span class="vb vb-${g.verdict}">${esc(g.verdict)}</span>
      ${isBest ? '<span class="star">★ best</span>' : ""}
      ${halted ? '<span class="vb vb-revert">update halted</span>' : ""}
      <div class="detail-nav">
        <button class="btn" ${idx <= 0 ? "disabled" : ""} onclick="selectGen(${g.generation - 1})">← prev</button>
        <button class="btn" ${idx >= gens().length - 1 ? "disabled" : ""} onclick="selectGen(${g.generation + 1})">next →</button>
      </div>
    </div>
    <div class="note ${note.kind}" style="margin-bottom:16px"><div class="ico">${note.ico}</div><div><div class="nt"><b>${esc(note.t)}</b></div><div class="nd">${esc(note.d)}</div></div></div>
    <div class="kv-grid">${kv}</div>
    <div class="panel"><div class="panel-hdr"><h2>PPO / critic health</h2><span class="hint">checkpoint ${esc(g.checkpoint)}</span></div><div class="panel-body">${bars || '<div class="nd">no update_stats recorded for this generation</div>'}</div></div>`;
}

// ── interactivity ──────────────────────────────────────────────────────────────
function selectGen(genNum) {
  const g = gens().find((x) => x.generation === genNum);
  if (!g) return;
  STATE.sel = genNum;
  renderGenList(); renderDetail();
  switchTab("detail");
}
function switchTab(tab) {
  STATE.tab = tab;
  document.querySelectorAll(".vtab").forEach((b) => b.classList.toggle("active", b.dataset.tab === tab));
  ["overview", "health", "spectate", "detail"].forEach((t) => $("tab-" + t).classList.toggle("hidden", t !== tab));
}

// ── live badge / status bar / spectate (status-driven) ────────────────────────
function setLiveBadge() {
  const b = $("live-badge"), s = STATE.status;
  if (STATE.mode === "static") { b.className = "live-badge demo"; b.textContent = STATE.src === "demo data" ? "demo" : "file"; return; }
  if (!s) { b.className = "live-badge off"; b.textContent = "offline"; return; }
  if (s.live) { b.className = "live-badge live"; b.innerHTML = '<span class="dot"></span>LIVE'; }
  else { b.className = "live-badge idle"; b.textContent = (s.run && s.run.phase === "done") ? "run finished" : "idle"; }
}

const PHASE_LABEL = { collecting: "Collecting self-play games", updating: "PPO / critic update",
  evaluating: "Gauntlet evaluation", starting: "Starting…", idle: "Idle", done: "Run finished" };

function renderLiveBar() {
  const bar = $("live-bar"), s = STATE.status;
  if (STATE.mode === "static" || !s || !s.live) { bar.classList.add("hidden"); return; }
  const r = s.run || {}, done = r.games_done || 0, total = r.games_total || 0;
  const hasProg = r.phase === "collecting" && total > 0;
  const w = hasProg ? Math.min(100, (done / total) * 100) : 0;
  const prog = hasProg
    ? `<div class="lbar"><div class="lbar-track"><div class="lbar-fill" style="width:${w.toFixed(0)}%"></div></div></div>`
    : `<div class="lbar"><div class="lbar-track"><div class="lbar-fill lbar-indet"></div></div></div>`;
  const wr = r.running_p1_winrate;
  bar.classList.remove("hidden");
  bar.innerHTML =
    `<div><div class="lb-phase">Gen ${r.generation ?? "?"} · ${esc(PHASE_LABEL[r.phase] || r.phase || "")}</div>` +
    `<div class="lb-sub">${hasProg ? done + "/" + total + " games" : "&nbsp;"}${r.n_generations ? " · target " + r.n_generations + " gens" : ""}${r.last_verdict ? " · last gen: " + esc(r.last_verdict) : ""}</div></div>` +
    prog +
    `<div class="lb-stat"><div class="v">${wr == null ? "—" : (wr * 100).toFixed(0) + "%"}</div><div class="l">p1 win (running)</div></div>` +
    `<div class="lb-stat"><div class="v">${(s.active_battles || []).length}</div><div class="l">live battles</div></div>`;
}

function renderSpectate() {
  const pane = $("tab-spectate"), s = STATE.status;
  const battles = (s && s.active_battles) || [];
  const url = (s && s.showdown_url) || "http://localhost:8000";
  if (STATE.mode === "static") {
    pane.innerHTML = `<div class="spec-empty"><div class="big">🎬</div><p>Live spectating is available when the dashboard is served by the dashboard server during a self-play run.</p></div>`;
    return;
  }
  if (!battles.length) {
    pane.innerHTML = `<div class="spec-empty"><div class="big">🎬</div><p>No live battles right now.<br/>Start a self-play run and its current game shows here, turn by turn.</p></div>`;
    return;
  }
  const watch = battles.map((b) =>
    `<a class="btn pri" href="${esc(url)}/${esc(b.tag)}" target="_blank" rel="noopener">▶ Watch ${esc(b.p1)} vs ${esc(b.p2)} (animated, new tab)</a>`).join(" ");
  const note = `<div class="spec-note">Live turn-by-turn view of the current self-play game, rendered from its battle log (no internet needed). <b>▶ Watch</b> opens the full animated Showdown client in a new tab (needs internet).</div>`;
  pane.innerHTML = note + `<div class="spec-actions">${watch}</div><div id="tv-container"></div>`;
  renderTurnViewerInto();
}

function renderTurnViewerInto() {
  const c = document.getElementById("tv-container");
  if (c) c.innerHTML = renderTurnViewer(STATE.liveLog);
}

function renderTurnViewer(parsed) {
  if (!parsed || !parsed.frames || !parsed.frames.length)
    return `<div class="spec-empty"><div class="big">⏳</div><p>Waiting for the first turn of the current game…</p></div>`;
  const n = parsed.frames.length;
  const idx = STATE.viewTurn = STATE.followLatest ? n - 1 : Math.max(0, Math.min(STATE.viewTurn, n - 1));
  const fr = parsed.frames[idx], st = fr.state, pl = parsed.players;
  const pills = [];
  if (st.field.weather) pills.push(`<span class="fp">${esc(st.field.weather)}</span>`);
  if (st.field.terrain) pills.push(`<span class="fp">${esc(st.field.terrain)}</span>`);
  if (st.field.trickRoom) pills.push(`<span class="fp">Trick Room</span>`);
  const fieldBar = pills.length ? `<div class="tv-field">${pills.join("")}</div>` : "";
  const panel = (side, cls) =>
    `<div class="tv-panel"><div class="tv-phdr ${cls}">${esc(pl[side] || side)}</div>` +
    `<div class="tv-field-row">${monCard(st.active[side].a, "a")}${monCard(st.active[side].b, "b")}</div></div>`;
  const logRows = (fr.events || []).length
    ? fr.events.map((e) => `<div class="al-row"><span class="al-evt al-${e.t}">${esc(e.t)}</span><span class="al-text">${esc(e.text)}</span></div>`).join("")
    : `<div class="al-empty">— leads / setup —</div>`;
  const win = (idx === n - 1 && parsed.winner) ? `<span class="tv-win">🏆 ${esc(parsed.winner)} won</span>` : "";
  const label = fr.turn === 0 ? "Leads" : "Turn " + fr.turn;
  return `
    <div class="tv-bar">
      <button class="btn" onclick="stepTurn(-1)">◀</button>
      <span class="tv-turn">${esc(label)} <span class="tv-of">${idx + 1}/${n}</span></span>
      <input id="tv-scrub" type="range" min="0" max="${n - 1}" value="${idx}" oninput="setViewTurn(this.value)" />
      <button class="btn" onclick="stepTurn(1)">▶</button>
      <button class="btn ${STATE.followLatest ? "pri" : ""}" onclick="toggleFollow()">${STATE.followLatest ? "● LIVE" : "○ follow"}</button>
      ${win}
    </div>
    ${fieldBar}
    <div class="tv-board">${panel("p1", "p1h")}${panel("p2", "p2h")}</div>
    <div class="tv-log"><h4>${esc(label)} events</h4>${logRows}</div>`;
}

function monCard(mon, slot) {
  if (!mon) return `<div class="tv-mon empty"><span class="tv-slot">${slot.toUpperCase()}</span><span class="tv-none">empty</span></div>`;
  const hp = mon.fainted ? 0 : (mon.hp == null ? 100 : mon.hp);
  const hpcls = hp <= 35 ? "hp-lo" : hp <= 70 ? "hp-mid" : "hp-hi";
  const boosts = Object.entries(mon.boosts || {}).filter(([, v]) => v)
    .map(([k, v]) => `<span class="boost-pip ${v > 0 ? "boost-pos" : "boost-neg"}">${v > 0 ? "+" : ""}${v} ${esc(k)}</span>`).join("");
  const status = mon.status ? `<span class="tv-status">${esc(mon.status)}</span>` : "";
  const tera = mon.tera ? `<span class="tv-tera">tera ${esc(mon.tera)}</span>` : "";
  return `<div class="tv-mon ${mon.fainted ? "fainted" : ""}">` +
    `<div class="tv-mon-top"><span class="tv-slot">${slot.toUpperCase()}</span><span class="tv-species">${esc(mon.species)}</span>${tera}${status}</div>` +
    `<div class="tv-hp"><div class="tv-hp-fill ${hpcls}" style="width:${Math.max(0, Math.min(100, hp))}%"></div></div>` +
    `<div class="tv-hp-num">${mon.fainted ? "fainted" : hp + "%"}</div>` +
    `<div class="tv-boosts">${boosts}</div></div>`;
}

// Parse a Showdown |-protocol log into per-turn frames (board state + events). Not a
// full engine — enough for a turn-by-turn board: active mon per slot, HP%, status, boosts.
function parseBattleLog(lines) {
  const players = { p1: "p1", p2: "p2" };
  const active = { p1: { a: null, b: null }, p2: { a: null, b: null } };
  const field = { weather: null, terrain: null, trickRoom: false };
  const frames = []; let curTurn = 0, events = [], winner = null, fmt = "";
  const snap = () => JSON.parse(JSON.stringify({ active, field }));
  const push = () => frames.push({ turn: curTurn, state: snap(), events: events.slice() });
  const who = (ref) => { const m = /^(p[12])([ab]?): ?(.*)$/.exec(ref || ""); return m ? { side: m[1], slot: m[2] || "a", nick: m[3] } : null; };
  const hpOf = (s) => { if (!s) return null; if (/fnt/.test(s)) return 0; const m = /^(\d+)(?:\/(\d+))?/.exec(String(s).trim()); return m ? (m[2] ? Math.round((+m[1] / +m[2]) * 100) : +m[1]) : null; };
  const sp = (m) => (m ? ((active[m.side][m.slot] && active[m.side][m.slot].species) || m.nick) : "?");
  for (const raw of lines) {
    const p = String(raw).split("|"); const cmd = p[1];
    if (cmd === "player") { if (p[2] && p[3]) players[p[2]] = p[3]; }
    else if (cmd === "tier") fmt = p[2] || fmt;
    else if (cmd === "turn") { push(); curTurn = parseInt(p[2], 10) || curTurn + 1; events = []; }
    else if (cmd === "switch" || cmd === "drag" || cmd === "replace") {
      const w = who(p[2]); if (!w) continue;
      const species = (p[3] || "").split(",")[0].trim();
      active[w.side][w.slot] = { species, nick: w.nick, hp: hpOf(p[4]), status: null, boosts: {}, fainted: false };
      if (cmd !== "replace") events.push({ t: "switch", side: w.side, text: `${species} switched in (${w.side})` });
    }
    else if (cmd === "move") { const w = who(p[2]), tg = who(p[4]); events.push({ t: "move", side: w && w.side, text: `${sp(w)} used ${p[3]}${tg ? " → " + sp(tg) : ""}` }); }
    else if (cmd === "-damage" || cmd === "-heal") {
      const w = who(p[2]); if (!w) continue;
      const now = hpOf(p[3]); if (active[w.side][w.slot]) active[w.side][w.slot].hp = now;
      events.push({ t: cmd === "-damage" ? "damage" : "heal", side: w.side, text: `${sp(w)} ${cmd === "-damage" ? "↓" : "↑"} HP → ${now == null ? "?" : now}%` });
    }
    else if (cmd === "faint") { const w = who(p[2]); if (w && active[w.side][w.slot]) { active[w.side][w.slot].fainted = true; active[w.side][w.slot].hp = 0; events.push({ t: "faint", side: w.side, text: `${sp(w)} fainted` }); } }
    else if (cmd === "-boost" || cmd === "-unboost") { const w = who(p[2]); if (w && active[w.side][w.slot]) { const amt = (cmd === "-boost" ? 1 : -1) * (parseInt(p[4], 10) || 0); const b = active[w.side][w.slot].boosts; b[p[3]] = (b[p[3]] || 0) + amt; events.push({ t: "boost", side: w.side, text: `${sp(w)} ${amt > 0 ? "+" : ""}${amt} ${p[3]}` }); } }
    else if (cmd === "-status") { const w = who(p[2]); if (w && active[w.side][w.slot]) active[w.side][w.slot].status = p[3]; }
    else if (cmd === "-curestatus") { const w = who(p[2]); if (w && active[w.side][w.slot]) active[w.side][w.slot].status = null; }
    else if (cmd === "-weather") field.weather = (p[2] && p[2] !== "none") ? p[2] : null;
    else if (cmd === "-fieldstart") { if (/trick room/i.test(p[2] || "")) field.trickRoom = true; else field.terrain = (p[2] || "").replace(/move: /i, ""); }
    else if (cmd === "-fieldend") { if (/trick room/i.test(p[2] || "")) field.trickRoom = false; else field.terrain = null; }
    else if (cmd === "-terastallize") { const w = who(p[2]); if (w && active[w.side][w.slot]) { active[w.side][w.slot].tera = p[3]; events.push({ t: "tera", side: w.side, text: `${sp(w)} Terastallized → ${p[3]}` }); } }
    else if (cmd === "detailschange" || cmd === "-formechange") { const w = who(p[2]); if (w && active[w.side][w.slot]) { active[w.side][w.slot].species = (p[3] || "").split(",")[0].trim(); events.push({ t: "mega", side: w.side, text: `${sp(w)} → ${active[w.side][w.slot].species}` }); } }
    else if (cmd === "-mega") { const w = who(p[2]); events.push({ t: "mega", side: w && w.side, text: `${sp(w)} Mega-Evolved` }); }
    else if (cmd === "swap") { const w = who(p[2]); if (w) { const s = active[w.side]; [s.a, s.b] = [s.b, s.a]; events.push({ t: "switch", side: w.side, text: `Ally Switch (${w.side})` }); } }   // doubles slot swap
    else if (cmd === "-ability") { const w = who(p[2]); if (w) events.push({ t: "boost", side: w.side, text: `${sp(w)} ability: ${p[3]}` }); }
    else if (cmd === "cant") { const w = who(p[2]); if (w) events.push({ t: "damage", side: w.side, text: `${sp(w)} couldn't move${p[3] ? " (" + p[3] + ")" : ""}` }); }
    else if (cmd === "win") winner = p[2] || null;
  }
  push();
  return { players, fmt, frames, winner };
}

function setViewTurn(v) { STATE.viewTurn = parseInt(v, 10) || 0; STATE.followLatest = !!(STATE.liveLog && STATE.viewTurn >= STATE.liveLog.frames.length - 1); renderTurnViewerInto(); }
function stepTurn(d) { if (!STATE.liveLog) return; STATE.viewTurn = Math.max(0, Math.min(STATE.viewTurn + d, STATE.liveLog.frames.length - 1)); STATE.followLatest = STATE.viewTurn >= STATE.liveLog.frames.length - 1; renderTurnViewerInto(); }
function toggleFollow() { STATE.followLatest = !STATE.followLatest; if (STATE.followLatest && STATE.liveLog) STATE.viewTurn = STATE.liveLog.frames.length - 1; renderTurnViewerInto(); }

// ── apply incoming data ─────────────────────────────────────────────────────────
function manifestKey(m) { const g = m.generations || []; return `${m.n_generations}|${g.length ? JSON.stringify(g[g.length - 1]) : ""}`; }

function applyManifest(m, src) {
  if (!m || !Array.isArray(m.generations)) { notify("Not a valid manifest.json", true); return false; }
  STATE.m = m; STATE.src = src;
  const ids = m.generations.map((g) => g.generation);
  if (STATE.sel == null || !ids.includes(STATE.sel)) STATE.sel = ids.length ? ids[ids.length - 1] : null;  // keep the user's selection across live updates
  renderAll();
  return true;
}

function applyStatus(s) {
  STATE.status = s;
  setLiveBadge(); renderLiveBar();
  const tags = ((s && s.active_battles) || []).map((b) => b.tag).join(",");
  if (tags !== STATE.lastTags) { STATE.lastTags = tags; renderSpectate(); }   // only rebuild iframes when the room set changes (don't reload mid-battle)
}

// ── live polling ──────────────────────────────────────────────────────────────
async function fetchJSON(url) { const r = await fetch(url, { cache: "no-store" }); if (!r.ok) throw new Error("HTTP " + r.status); return r.json(); }

async function pollOnce() {
  let s;
  try { s = await fetchJSON("status.json"); } catch (e) { return false; }     // not served (file://) or server down
  applyStatus(s);
  try {
    const m = await fetchJSON("manifest.json");
    const key = manifestKey(m);
    if (key !== STATE.lastKey) { STATE.lastKey = key; applyManifest(m, "live"); }   // re-render charts only when a generation actually changes
  } catch (e) { /* manifest may not exist until gen 0 finishes */ }
  try { applyLiveLog(await fetchJSON("live_log.json")); } catch (e) { /* no live log yet */ }
  return true;
}

function applyLiveLog(lg) {
  const key = ((lg && lg.tag) || "") + ":" + ((lg && lg.n_lines) || 0);
  if (key === STATE.logKey) return;                               // unchanged since last poll
  STATE.logKey = key;
  const newTag = lg && lg.tag;
  const sameBattle = STATE.liveLog && STATE.liveLog.tag === newTag;
  if (!sameBattle) { STATE.followLatest = true; STATE.viewTurn = 0; }   // new battle -> follow live
  STATE.liveLog = (lg && lg.log && lg.log.length) ? parseBattleLog(lg.log) : null;
  if (STATE.liveLog) STATE.liveLog.tag = newTag;
  if (STATE.tab === "spectate") renderTurnViewerInto();           // live-tail the viewer
}
function stopPolling() { if (STATE.poll) { clearInterval(STATE.poll); STATE.poll = null; } }

function loadStatic(m, src) {                       // demo / file-picker → leave live mode
  stopPolling(); STATE.mode = "static"; STATE.status = null; STATE.lastTags = "__static__";
  if (STATE.tab === "spectate") STATE.tab = "overview";
  if (applyManifest(m, src)) notify(`Loaded ${m.n_generations} generations from ${src}`);
}

async function init() {
  $("btn-demo").addEventListener("click", () => loadStatic(JSON.parse(JSON.stringify(DEMO_MANIFEST)), "demo data"));
  $("btn-load").addEventListener("click", () => $("file-in").click());
  $("file-in").addEventListener("change", (e) => {
    const f = e.target.files[0]; if (!f) return;
    const r = new FileReader();
    r.onload = () => { try { loadStatic(JSON.parse(r.result), f.name); } catch (err) { notify("Failed to parse JSON: " + err.message, true); } };
    r.readAsText(f);
  });
  document.querySelectorAll(".vtab").forEach((b) => b.addEventListener("click", () => switchTab(b.dataset.tab)));
  // Try the served live feeds; fall back to the embedded demo (e.g. opened via file://).
  STATE.mode = "live";
  const ok = await pollOnce();
  if (ok) STATE.poll = setInterval(pollOnce, 2000);
  else loadStatic(JSON.parse(JSON.stringify(DEMO_MANIFEST)), "demo data");
}
window.selectGen = selectGen;
window.setViewTurn = setViewTurn;
window.stepTurn = stepTurn;
window.toggleFollow = toggleFollow;
document.addEventListener("DOMContentLoaded", init);
