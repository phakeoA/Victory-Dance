"use strict";
// Console — the dashboard's "Console" TAB. Streams the launched run's stdout/stderr "like a terminal"
// (a UI-launched run goes to a HIDDEN console, so this is the only place to watch it live). Wrapped in
// an IIFE so its $/state don't collide with dashboard.js's globals; all element ids are namespaced
// "cons-". Polls /launch_log ONLY while the tab is visible: it passes the byte offset it has already
// consumed, so the server returns just the NEW bytes; it clears the buffer when the server signals a
// reset (first load / a new or rotated log file).
(function () {
  const $ = (id) => document.getElementById(id);
  const POLL_MS = 1500;
  const MAX_CHARS = 500000;           // cap the scrollback so a long run can't grow the DOM unbounded
  let offset = 0, path = null, paused = false, autoscroll = true, busy = false;

  const pane = () => $("tab-console");
  const visible = () => { const p = pane(); return p && !p.classList.contains("hidden"); };

  function setBadge(running, available) {
    const b = $("cons-badge");
    if (!b) return;
    if (!available) { b.textContent = "no run"; b.className = "cons-badge off"; }
    else if (running) { b.textContent = "running"; b.className = "cons-badge on"; }
    else { b.textContent = "exited"; b.className = "cons-badge done"; }
  }

  function atBottom() {
    const o = $("cons-out");
    return o.scrollHeight - o.scrollTop - o.clientHeight < 24;   // within ~2 lines of the end
  }
  function scrollEnd() { const o = $("cons-out"); o.scrollTop = o.scrollHeight; }

  function append(text) {
    if (!text) return;
    const o = $("cons-out");
    const stick = autoscroll && atBottom();
    o.textContent += text;
    if (o.textContent.length > MAX_CHARS) o.textContent = o.textContent.slice(-MAX_CHARS);
    if (stick) scrollEnd();
  }
  const clearOut = () => { $("cons-out").textContent = ""; };

  async function poll() {
    if (busy || paused || !visible()) return;
    busy = true;
    try {
      const q = path != null
        ? `launch_log?offset=${offset}&path=${encodeURIComponent(path)}`
        : "launch_log";
      const j = await (await fetch(q)).json();
      setBadge(j.running, j.available);
      if ($("cons-file")) $("cons-file").textContent = j.path || "(waiting for a launched run…)";
      if (!j.available) return;
      if (j.reset) clearOut();
      path = j.path; offset = j.offset;
      append(j.data || "");
    } catch (e) { /* server momentarily busy */ }
    finally { busy = false; }
  }

  function init() {
    const out = $("cons-out");
    if (!out) return;                 // console tab not present
    $("cons-btn-pause").addEventListener("click", () => {
      paused = !paused;
      const btn = $("cons-btn-pause");
      btn.textContent = paused ? "▶ Resume" : "⏸ Pause";
      btn.classList.toggle("pri", paused);
    });
    $("cons-btn-clear").addEventListener("click", clearOut);
    $("cons-btn-bottom").addEventListener("click", () => { autoscroll = true; scrollEnd(); });
    $("cons-btn-copy").addEventListener("click", async () => {
      try { await navigator.clipboard.writeText(out.textContent); } catch (e) { /* clipboard blocked */ }
    });
    // user scrolls up -> stop yanking them to the bottom; scroll back down -> re-enable follow
    out.addEventListener("scroll", () => { autoscroll = atBottom(); });
    // fetch the instant the tab is shown (don't wait up to POLL_MS for the next tick)
    new MutationObserver(() => { if (visible()) poll(); })
      .observe(pane(), { attributes: true, attributeFilter: ["class"] });
    setInterval(poll, POLL_MS);
    if (visible()) poll();
  }

  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", init);
  else init();
})();
