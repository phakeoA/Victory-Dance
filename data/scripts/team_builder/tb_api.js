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
      // Late pokedex load — server may have come online after boot
      if (!POKEDEX) loadPokedex();
      return true;
    }
  } catch (_) { }
  dot.style.background = 'var(--red)';
  lbl.textContent = 'Server: offline';
  return false;
}

/**
 * Fetch data/pokedex.json (served by server.py at /data/pokedex.json) into
 * the global POKEDEX.  Enables ability dropdowns + deterministic mega-ability
 * resolution in the client-side parser.  Silently no-ops when offline —
 * everything degrades to free-text inputs and reveal-based abilities.
 */
async function loadPokedex() {
  if (POKEDEX) return true;
  try {
    const r = await fetch(`${SERVER}/data/pokedex.json`, { signal: AbortSignal.timeout(10000) });
    if (!r.ok) return false;
    POKEDEX = await r.json();
    // Re-render so ability dropdowns appear once the dex is available
    if (typeof renderMain === 'function' && activeBattle) renderMain();
    return true;
  } catch (_) {
    return false;
  }
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
 * POST the battle's rosters + revealed info to /fill-beliefs and return
 * per-species inject-panel suggestions from Pikalytics.
 * Only called from the "✨ Use Belief Integration" button — never automatic.
 */
async function fillBeliefsViaServer(battle) {
  const body = {
    vod_type:      battle.source_type || 'ranked_player_vod',
    players:       battle.players,
    revealed_info: battle.revealed_info || {},
  };
  const r = await fetch(`${SERVER}/fill-beliefs`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
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
    // Type A/B/C/D selector — without this the server stamps every
    // transition as Type B (ranked_player_vod).
    source_type:        battle.source_type || 'ranked_player_vod',
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
 * NOTE (Bug 8): `ability` is always the BASE forme's ability — the only one
 * a player chooses.  A mega forme's ability is fixed by the forme and is
 * derived from the pokedex on the Python side (transitions._inject_known_stats),
 * so it is never stored here.
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
