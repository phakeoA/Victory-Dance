// ═══════════════════════════════════════════════════════════
// tb_api.js
// All communication with the Flask backend (server.py).
// Endpoints: /health, /parse, /export
// Depends on: tb_constants.js
// ═══════════════════════════════════════════════════════════

/**
 * Ping /health and update the status dot + label in the sidebar.
 * Called on boot and every 15 s.
 */
async function checkServer() {
  const dot = document.getElementById('srv-dot');
  const lbl = document.getElementById('srv-status');
  try {
    const r = await fetch(`${SERVER}/health`, { signal: AbortSignal.timeout(2000) });
    if (r.ok) {
      dot.style.background = 'var(--green)';
      lbl.textContent = 'Server: connected';
      return true;
    }
  } catch (_) { }
  dot.style.background = 'var(--red)';
  lbl.textContent = 'Server: offline';
  return false;
}

/**
 * POST a replay .html file to /parse.
 * Returns the structured battle preview dict from vod_parser.py.
 * Note: the UI always does its own client-side parse for rendering;
 * this endpoint is exposed for future server-side enrichment (belief state, etc.).
 */
async function parseReplayViaServer(file) {
  const fd = new FormData();
  fd.append('replay_html', file);

  const r = await fetch(`${SERVER}/parse`, { method: 'POST', body: fd });
  if (!r.ok) {
    const err = await r.json().catch(() => ({ error: r.statusText }));
    throw new Error(err.error || r.statusText);
  }
  return r.json();
}

/**
 * POST the approved battle entry to /export and trigger a JSONL download.
 * Falls back to exportJSON() (client-side) if the server is unreachable.
 */
async function exportViaServer(battle) {
  const body = {
    battle_id:          battle.replay_id || ('battle-' + Date.now()),
    known_teams_entry:  buildKnownTeamsEntry(battle),
    replay_html:        battle._rawHtml || '',
  };

  const r = await fetch(`${SERVER}/export`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
  if (!r.ok) {
    const err = await r.json().catch(() => ({ error: r.statusText }));
    throw new Error(err.error || r.statusText);
  }

  const blob  = await r.blob();
  const count = r.headers.get('X-Transition-Count') || '?';
  const a     = document.createElement('a');
  a.href      = URL.createObjectURL(blob);
  a.download  = `${body.battle_id}.jsonl`;
  a.click();
  showNotif(`Exported ${count} transitions`);
}

/**
 * Build the known_teams entry shape that server.py / replay_to_transitions expects:
 * {
 *   _meta: { yourSide, winner, p1name, p2name },
 *   p1: { SpeciesName: { nature, item, ability, ev_spread, moves }, ... },
 *   p2: { ... }
 * }
 */
function buildKnownTeamsEntry(b) {
  const entry = {
    _meta: {
      yourSide: b.players.our_side,
      winner:   b.winner,
      p1name:   b.players.p1.username,
      p2name:   b.players.p2.username,
    },
  };
  for (const pid of ['p1', 'p2']) {
    entry[pid] = {};
    for (const species of (b.players[pid].roster || [])) {
      const key = `${pid}:${species}`;
      const inj = b.known_team_overrides?.[key] || {};
      entry[pid][species] = {
        nature:    inj.nature    || null,
        item:      inj.item      || null,
        ability:   inj.ability   || null,
        ev_spread: inj.ev_spread || null,
        moves:     inj.moves     || null,
      };
    }
  }
  return entry;
}
