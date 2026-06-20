"use strict";
// Self-play launcher — runs as the dashboard's "Launcher" TAB (not a separate page). Wrapped in
// an IIFE so its $/state don't collide with dashboard.js's globals; all element ids are namespaced
// "lc-" to avoid clashes with the dashboard markup. Renders the run-config from /config-schema as
// labeled inputs, lets you load a config file OR type values, then POSTs the assembled config to
// /launch (generation.py is the backend). Live status from /status.json + /launch_state.
(function () {
  const $ = (id) => document.getElementById(id);
  let SCHEMA = null;                       // {sections: {sec: {field: {default, type, nullable}}}}

  function msg(text, kind) {
    const el = $("lc-msg");
    if (!el) return;
    el.textContent = text || "";
    el.className = "lc-msg" + (kind ? " " + kind : "");
    el.hidden = !text;
  }

  // ── render the form from the schema ────────────────────────────────────────
  function buildForm(schema) {
    const root = $("lc-form");
    if (!root) return;
    root.innerHTML = "";
    const sections = (schema && schema.sections) || {};
    if (schema && schema.error) msg(schema.error, "err");
    const order = ["generation", "ppo", "train", "gate"];
    const names = Object.keys(sections).sort((a, b) => order.indexOf(a) - order.indexOf(b));
    for (const sec of names) {
      const det = document.createElement("details");
      det.className = "lc-section";
      if (sec === "generation") det.open = true;
      const sum = document.createElement("summary");
      sum.textContent = sec;
      det.appendChild(sum);
      const grid = document.createElement("div");
      grid.className = "lc-fields";
      for (const key of Object.keys(sections[sec])) grid.appendChild(fieldEl(sec, key, sections[sec][key]));
      det.appendChild(grid);
      root.appendChild(det);
    }
  }

  function fieldEl(sec, key, spec) {
    const wrap = document.createElement("div");
    wrap.className = "lc-field" + (spec.type === "bool" ? " bool" : "");
    const id = `lcf_${sec}_${key}`;
    const label = document.createElement("label");
    label.htmlFor = id;
    label.textContent = key;
    let input;
    if (spec.type === "bool") {
      input = document.createElement("input");
      input.type = "checkbox";
      input.checked = !!spec.default;
    } else {
      input = document.createElement("input");
      input.type = (spec.type === "int" || spec.type === "float") ? "number" : "text";
      if (spec.type === "float") input.step = "any";
      input.placeholder = spec.nullable ? "(default: none)" : "";
      input.value = spec.default == null ? ""
        : (Array.isArray(spec.default) ? spec.default.join(", ") : String(spec.default));
    }
    input.id = id;
    input.dataset.section = sec;
    input.dataset.field = key;
    input.dataset.type = spec.type;
    if (spec.type === "bool") { wrap.appendChild(input); wrap.appendChild(label); }
    else { wrap.appendChild(label); wrap.appendChild(input); }
    return wrap;
  }

  // ── coerce one input to its JSON value (or undefined to OMIT it) ────────────
  function coerce(input) {
    const t = input.dataset.type;
    if (t === "bool") return input.checked;
    const raw = (input.value || "").trim();
    if (raw === "") return undefined;                  // empty -> omit (use the default)
    if (t === "int") { const n = parseInt(raw, 10); return Number.isNaN(n) ? undefined : n; }
    if (t === "float") { const n = parseFloat(raw); return Number.isNaN(n) ? undefined : n; }
    if (t === "list") return raw.split(",").map((s) => s.trim()).filter(Boolean);
    if (/^-?\d+$/.test(raw)) return parseInt(raw, 10);
    if (/^-?\d*\.\d+$/.test(raw)) return parseFloat(raw);
    if (raw === "true" || raw === "false") return raw === "true";
    return raw;
  }

  function collectConfig() {
    const cfg = {};
    for (const input of document.querySelectorAll("#lc-form input")) {
      const v = coerce(input);
      if (v === undefined) continue;
      const sec = input.dataset.section, key = input.dataset.field;
      (cfg[sec] = cfg[sec] || {})[key] = v;
    }
    return cfg;
  }

  function applyConfig(cfg) {
    for (const sec of Object.keys(cfg || {})) {
      if (typeof cfg[sec] !== "object" || cfg[sec] === null) continue;   // skip _comment etc.
      for (const key of Object.keys(cfg[sec])) {
        const input = $(`lcf_${sec}_${key}`);
        if (!input) continue;
        const val = cfg[sec][key];
        if (input.dataset.type === "bool") input.checked = !!val;
        else input.value = Array.isArray(val) ? val.join(", ") : (val == null ? "" : String(val));
      }
    }
  }

  async function loadSchema() {
    try { SCHEMA = await (await fetch("config-schema")).json(); buildForm(SCHEMA); }
    catch (e) { msg("failed to load /config-schema: " + e, "err"); }
  }

  // ── actions ────────────────────────────────────────────────────────────────
  async function startRun() {
    msg("starting run…");
    try {
      const r = await fetch("launch", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify(collectConfig()),
      });
      const j = await r.json();
      msg(r.ok && j.ok ? `run started (pid ${j.pid}); config: ${j.config}`
                       : "launch failed: " + (j.error || r.status), (r.ok && j.ok) ? "ok" : "err");
    } catch (e) { msg("launch request failed: " + e, "err"); }
  }

  async function stopRun() {
    try {
      const j = await (await fetch("stop", { method: "POST" })).json();
      msg(j.ok ? `stopped pid ${j.stopped_pid}` : ("stop failed: " + (j.error || "")), j.ok ? "ok" : "err");
    } catch (e) { msg("stop request failed: " + e, "err"); }
  }

  function exportConfig() {
    const blob = new Blob([JSON.stringify(collectConfig(), null, 2)], { type: "application/json" });
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = "config.json";
    a.click();
    URL.revokeObjectURL(a.href);
  }

  // ── live status poll ─────────────────────────────────────────────────────────
  async function poll() {
    try {
      const s = await (await fetch("status.json")).json();
      const run = s.run || {};
      if ($("lc-st-live")) $("lc-st-live").textContent = s.live ? "yes" : "no";
      if ($("lc-st-phase")) $("lc-st-phase").textContent = run.phase ?? "—";
      if ($("lc-st-gen")) $("lc-st-gen").textContent = run.generation ?? "—";
      const done = run.games_done ?? 0, total = run.games_total ?? 0;
      if ($("lc-st-games")) $("lc-st-games").textContent = `${done} / ${total}`;
      if ($("lc-st-verdict")) $("lc-st-verdict").textContent = run.last_verdict ?? "—";
      if ($("lc-st-bar")) $("lc-st-bar").style.width = total > 0 ? `${Math.min(100, 100 * done / total)}%` : "0%";
    } catch (e) { /* server momentarily busy */ }
    try {
      const ls = await (await fetch("launch_state")).json();
      if ($("lc-st-pid")) $("lc-st-pid").textContent = ls.pid != null
        ? `${ls.pid}${ls.running ? " (running)" : " (exited " + ls.returncode + ")"}` : "—";
      if ($("lc-st-log") && ls.log_tail && ls.log_tail.length) $("lc-st-log").textContent = ls.log_tail.join("\n");
    } catch (e) { /* no launched run */ }
  }

  // ── wire up (the markup lives in dashboard.html's #tab-launcher pane) ────────
  function init() {
    const loadBtn = $("lc-btn-load");
    if (!loadBtn) return;                              // launcher tab not present
    loadBtn.addEventListener("click", () => $("lc-config-file").click());
    $("lc-config-file").addEventListener("change", (ev) => {
      const file = ev.target.files[0];
      if (!file) return;
      const reader = new FileReader();
      reader.onload = () => {
        try { applyConfig(JSON.parse(reader.result)); msg(`loaded ${file.name}`, "ok"); }
        catch (e) { msg("invalid config JSON: " + e, "err"); }
      };
      reader.readAsText(file);
      ev.target.value = "";
    });
    $("lc-btn-reset").addEventListener("click", () => { if (SCHEMA) buildForm(SCHEMA); msg(""); });
    $("lc-btn-export").addEventListener("click", exportConfig);
    $("lc-btn-start").addEventListener("click", startRun);
    $("lc-btn-stop").addEventListener("click", stopRun);
    loadSchema();
    poll();
    setInterval(poll, 2000);
  }

  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", init);
  else init();
})();
