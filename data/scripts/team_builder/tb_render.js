// ═══════════════════════════════════════════════════════════
// tb_render.js
// All DOM render functions.
// Depends on: tb_constants.js
// ═══════════════════════════════════════════════════════════

// ── Battle list (sidebar) ────────────────────────────────
function renderBattleList() {
  const el = document.getElementById('b-list');
  if (battles.length === 0) {
    el.innerHTML = '<div style="font-size:11px;color:var(--text3);padding:4px 9px;">No battles loaded yet.</div>';
    return;
  }
  el.innerHTML = battles.map((b, i) => {
    const p1 = b.players.p1.username || 'P1';
    const p2 = b.players.p2.username || 'P2';
    const srcColor = {
      ranked_player_vod: 'var(--blue)',
      own_vod:           'var(--green)',
      bot_vod:           'var(--yellow)',
      self_play:         '#ef4179',
    }[b.source_type] || 'var(--text3)';
    const active = b === activeBattle;
    return `<div class="b-item${active ? ' active' : ''}" onclick="selectBattle(${i})">
      <div style="flex:1;min-width:0">
        <div class="b-name">${p1} vs ${p2}</div>
        <div class="b-meta">${b.turns.length} turns · <span style="color:${srcColor}">${b.source_type}</span></div>
      </div>
      <button class="b-del" onclick="event.stopPropagation();deleteBattle(${i})" title="Remove">✕</button>
    </div>`;
  }).join('');
}

// ── Main content area ────────────────────────────────────
function renderMain() {
  const main = document.getElementById('main');
  if (!activeBattle) {
    main.innerHTML = `<div id="empty-state">
      <div class="big">🎮</div>
      <p>Load a Showdown replay HTML to get started, or create a new battle manually.<br><br>You can also drag &amp; drop a replay file here.</p>
    </div>`;
    return;
  }

  const b       = activeBattle;
  const p1name  = b.players.p1.username || 'Player 1';
  const p2name  = b.players.p2.username || 'Player 2';
  const ourSide = b.players.our_side;

  main.innerHTML = `
  <div id="b-hdr">
    <div style="flex:1">
      <div class="lbl-sm">Battle</div>
      <div style="font-size:15px;font-weight:700;color:var(--text)">${p1name} <span style="color:var(--text3)">vs</span> ${p2name}</div>
      <div style="font-size:11px;color:var(--text3);margin-top:2px">${b.format || ''} · ${b.replay_id || 'no id'}</div>
    </div>
    <div>
      <div class="lbl-sm">Your side</div>
      <select class="fld" id="your-side" onchange="setOurSide(this.value)">
        <option value="p1"${ourSide === 'p1' ? ' selected' : ''}>p1 — ${p1name}</option>
        <option value="p2"${ourSide === 'p2' ? ' selected' : ''}>p2 — ${p2name}</option>
      </select>
    </div>
    <div>
      <div class="lbl-sm">Source type</div>
      <div id="src-type-row">
        ${['ranked_player_vod', 'own_vod', 'bot_vod', 'self_play'].map((t, i) => {
          const lbl = ['B: Ranked VOD', 'A: Own VOD', 'C: Bot vs Ranked', 'D: Self-Play'][i];
          const ac  = ['b', 'a', 'c', 'd'][i];
          return `<button class="src-btn${b.source_type === t ? ' active-' + ac : ''}" onclick="setSourceType('${t}')">${lbl}</button>`;
        }).join('')}
      </div>
    </div>
    <button class="btn suc" onclick="doExport()">⬇ Export JSONL</button>
    <button class="btn" onclick="exportJSON()">⬇ Export JSON</button>
  </div>

  <div id="view-tabs"></div>
  <div id="tab-content"></div>`;

  renderTabBar();
  renderTabContent();
}

function renderTabBar() {
  const el = document.getElementById('view-tabs');
  if (!el) return;
  el.innerHTML = `
    <button class="vtab${activeTab === 'turns'  ? ' active' : ''}" onclick="setTab('turns')">📽 Turn Viewer</button>
    <button class="vtab${activeTab === 'inject' ? ' active' : ''}" onclick="setTab('inject')">✏️ Inject Stats</button>
    <button class="vtab${activeTab === 'json'   ? ' active' : ''}" onclick="setTab('json')">{ } Raw JSON</button>`;
}

function renderTabContent() {
  const el = document.getElementById('tab-content');
  if (!el) return;
  if (activeTab === 'turns')  return renderTurnViewer(el);
  if (activeTab === 'inject') return renderInjectPanel(el);
  renderJsonTab(el);
}

// ── Turn Viewer ──────────────────────────────────────────
function renderTurnViewer(el) {
  const b      = activeBattle;
  const nTurns = b.turns.length;
  if (nTurns === 0) {
    el.innerHTML = '<div style="padding:20px;color:var(--text3)">No turns recorded.</div>';
    return;
  }
  const idx      = Math.min(activeTurnIdx, nTurns - 1);
  const turn     = b.turns[idx];
  const turnData = turn.state_after_actions || turn;

  el.innerHTML = `
  <div id="turn-viewer">
    <div id="tv-bar">
      <span class="tv-label">Turn</span>
      <input type="range" id="turn-scrubber" min="0" max="${nTurns - 1}" value="${idx}"
             oninput="setTurn(parseInt(this.value))">
      <span id="tv-info">${idx + 1} / ${nTurns}</span>
      <div id="tv-btns">
        <button class="btn" onclick="setTurn(${Math.max(0, idx - 1)})">◀</button>
        <button class="btn" onclick="setTurn(${Math.min(nTurns - 1, idx + 1)})">▶</button>
      </div>
    </div>
    <div id="field-bar">
      <span id="field-bar-label">Field:</span>
      ${renderFieldPills(turnData)}
    </div>
    <div id="turn-board">
      ${renderSideSlots(turn, 'p1', b)}
      ${renderSideSlots(turn, 'p2', b)}
    </div>
    <div id="action-log-wrap">
      <h4>Turn ${turn.turn} — Actions</h4>
      <div id="action-log">${renderActionLog(turn)}</div>
    </div>
  </div>`;
}

function renderFieldPills(turnData) {
  const pills = [];
  const f  = turnData.field || {};
  const sc = turnData.side_conditions || {};

  if (f.weather)  pills.push(`<span class="field-pill fp-weather">☁ ${f.weather}</span>`);
  if (f.terrain)  pills.push(`<span class="field-pill fp-terrain">🌿 ${f.terrain}</span>`);
  if (f.trick_room_turns_remaining > 0)
    pills.push(`<span class="field-pill fp-tr">⏲ Trick Room ×${f.trick_room_turns_remaining}</span>`);
  if (sc.p1?.tailwind_turns_remaining > 0)
    pills.push(`<span class="field-pill fp-tw">💨 P1 Tailwind ×${sc.p1.tailwind_turns_remaining}</span>`);
  if (sc.p2?.tailwind_turns_remaining > 0)
    pills.push(`<span class="field-pill fp-tw">💨 P2 Tailwind ×${sc.p2.tailwind_turns_remaining}</span>`);

  return pills.length
    ? pills.join('')
    : '<span style="font-size:11px;color:var(--text3)">None</span>';
}

function renderSideSlots(turn, pid, b) {
  const isP1     = pid === 'p1';
  const username = b.players[pid].username || pid.toUpperCase();
  const ourSide  = b.players.our_side;
  const yourTag  = pid === ourSide ? '<span class="your-side-tag">YOUR SIDE</span>' : '';
  const winTag   = b.winner && b.players[pid].username === b.winner ? '<span class="winner-badge">🏆</span>' : '';

  const roster         = b.players[pid].roster || [];
  const afterState     = turn.state_after_actions || turn;
  const activeForPlayer = Object.entries(afterState.active_pokemon || {})
    .filter(([k]) => k.startsWith(pid))
    .map(([k, m]) => ({ ...m, slotKey: k }));

  function monData(species) {
    const active  = activeForPlayer.find(m =>
      m.base_species === species || m.species === species || m.species.startsWith(species.split('-')[0]));
    const inj     = b.known_team_overrides?.[`${pid}:${species}`];
    const hp      = active ? active.hp_current : null;
    const fainted = active ? active.is_fainted : false;
    const isMega  = active ? active.is_mega : false;
    const onField = !!active && !fainted;
    const boosts  = active ? (active.boosts || {}) : {};
    return { active, inj, hp, fainted, isMega, onField, boosts, displaySpecies: active?.species || species, species };
  }

  // Active (on-field) slots — up to 2, shown large
  const fieldSlots = activeForPlayer
    .filter(m => !m.is_fainted)
    .sort((a, b) => a.slotKey.localeCompare(b.slotKey));

  const fieldHtml = fieldSlots.length
    ? fieldSlots.map(m => {
        const baseSp = m.base_species || m.species.replace(/-Mega(-[XY])?$/, '');
        const inj = b.known_team_overrides?.[`${pid}:${baseSp}`]
                 || b.known_team_overrides?.[`${pid}:${m.species}`];
        const boosts    = m.boosts || {};
        const boostHtml = Object.entries(boosts)
          .filter(([, v]) => v !== 0)
          .map(([s, v]) => `<span class="boost-pip ${v > 0 ? 'boost-pos' : 'boost-neg'}">${STAT_LABELS[s] || s} ${v > 0 ? '+' : ''}${v}</span>`)
          .join('');
        const slotLetter = m.slotKey.slice(-1).toUpperCase();
        const teraTag  = m.is_terastallized ? `<span class="active-badge tera-badge">TERA</span>` : '';
        const megaTag  = m.is_mega          ? `<span class="active-badge mega-badge">MEGA</span>` : '';
        const slotTag  = `<span class="active-slot-letter">${slotLetter}</span>`;
        const itemLine = m.known_item ? `<div class="active-item">🎒 ${m.known_item}</div>` : '';
        const abilityLine = m.known_ability
          ? `<div class="active-item">⚡ ${m.known_ability}${m.is_mega ? ' <span style="font-size:8px;color:var(--text3)">(mega — fixed)</span>' : ''}</div>`
          : '';
        const hp       = m.hp_current ?? 100;
        return `<div class="active-slot-card">
            <div class="active-slot-top">${slotTag}${teraTag}${megaTag}</div>
            <div class="active-species">${m.species}${inj ? ' <span class="inj-known">✓</span>' : ''}</div>
            <div class="active-hp-bar"><div class="slot-hp-fill ${hpColor(hp)}" style="width:${hp}%"></div></div>
            <div class="active-hp-num">${hp.toFixed(0)}% HP</div>
            ${itemLine}
            ${abilityLine}
            ${boostHtml ? `<div class="slot-boosts">${boostHtml}</div>` : ''}
          </div>`;
      }).join('')
    : `<div class="active-empty">— no active Pokémon —</div>`;

  // Bench row — all roster entries as compact chips
  const benchHtml = roster.map(species => {
    const d = monData(species);
    let cls = 'bench-chip';
    if (d.fainted) cls += ' bench-fainted';
    else if (d.onField) cls += ' bench-onfield';
    const hp      = d.hp !== null ? `<span class="bench-hp">${d.hp.toFixed(0)}%</span>` : '';
    const megaTag = d.isMega ? `<span class="bench-mega">M</span>` : '';
    const teraTag = d.active?.is_terastallized ? `<span class="bench-tera">T</span>` : '';
    return `<div class="${cls}" title="${d.displaySpecies}${d.hp !== null ? ' · ' + d.hp.toFixed(0) + '%' : ''}">
      <span class="bench-name">${d.displaySpecies}</span>${megaTag}${teraTag}${hp}
    </div>`;
  }).join('');

  return `<div class="tb-panel">
    <div class="tb-phdr ${isP1 ? 'p1h' : 'p2h'}">${pid.toUpperCase()} · ${username} ${yourTag} ${winTag}</div>
    <div class="tb-field-row">${fieldSlots.length
      ? fieldHtml
      : '<div class="active-empty">— no active Pokémon recorded —</div>'}</div>
    ${roster.length ? `<div class="tb-bench-row">
      <span class="bench-label">Bench</span>
      <div class="bench-chips">${benchHtml}</div>
    </div>` : ''}
  </div>`;
}

function renderActionLog(turn) {
  if (!turn.actions?.length)
    return '<div style="font-size:11px;color:var(--text3);padding:4px 0">No actions recorded.</div>';

  return turn.actions.map(a => {
    let badge = '', text = '';
    if (a.event === 'move') {
      badge = `<span class="al-evt al-move">Move</span>`;
      text  = `<b>${a.user_slot}</b> ${a.user_species || ''} → <b>${a.move}</b>${a.is_protect ? ' 🛡' : ''}${a.target_slot ? ` on ${a.target_slot}` : ''}`;
    } else if (a.event === 'switch') {
      badge = `<span class="al-evt al-switch">Switch</span>`;
      text  = `${a.player} sends out <b>${a.species}</b> to slot ${a.slot}`;
    } else if (a.event === 'damage') {
      badge = `<span class="al-evt al-damage">Dmg</span>`;
      text  = `<b>${a.slot}</b> ${a.species || ''} → ${a.hp_pct_after?.toFixed(0)}% HP`;
      if (a.source_move)   text += ` (from ${a.source_move})`;
      if (a.hp_pct_delta !== null) text += ` <span class="al-delta">(${a.hp_pct_delta?.toFixed(0)}%)</span>`;
    } else if (a.event === 'heal') {
      badge = `<span class="al-evt al-damage" style="background:rgba(76,175,125,.15);color:var(--green);border-color:var(--green)">Heal</span>`;
      text  = `<b>${a.slot}</b> → ${a.hp_pct_after?.toFixed(0)}% HP`;
      if (a.hp_pct_delta !== null) text += ` <span class="al-heal">(+${a.hp_pct_delta?.toFixed(0)}%)</span>`;
    } else if (a.event === 'faint') {
      badge = `<span class="al-evt al-faint">Faint</span>`;
      text  = `<b>${a.slot}</b> ${a.species || ''} fainted`;
    } else if (a.event === 'mega_evolution') {
      badge = `<span class="al-evt al-mega">Mega</span>`;
      text  = `${a.slot} → <b>${a.new_species}</b>`;
      if (a.mega_ability)     text += ` · ability locked to <b>${a.mega_ability}</b>`;
      if (a.pre_mega_ability) text += ` <span class="al-delta">(was ${a.pre_mega_ability})</span>`;
    } else if (a.event === 'forme_change') {
      badge = `<span class="al-evt al-mega" style="opacity:.7">Forme</span>`;
      text  = `${a.slot} → <b>${a.new_species}</b>`;
    } else if (a.event === 'ability_revealed') {
      badge = `<span class="al-evt" style="background:rgba(125,165,245,.15);color:var(--blue);border:1px solid var(--blue)">Ability</span>`;
      text  = `<b>${a.slot}</b> ${a.species || ''} revealed <b>${a.ability}</b>${a.is_mega_ability ? ' <span class="al-delta">(mega — fixed)</span>' : ''}`;
    } else if (a.event === 'stat_change') {
      badge = `<span class="al-evt al-stat">Stat</span>`;
      text  = `<b>${a.slot}</b> ${STAT_LABELS[a.stat] || a.stat} ${a.stages > 0 ? '+' : ''}${a.stages}`;
    } else if (a.event === 'item_consumed' || a.event === 'item_revealed') {
      badge = `<span class="al-evt" style="background:rgba(245,200,66,.15);color:var(--yellow);border:1px solid var(--yellow)">Item</span>`;
      text  = `<b>${a.slot}</b> ${a.event === 'item_consumed' ? 'consumed' : 'revealed'} <b>${a.item}</b>`;
    } else if (a.event === 'terastallize') {
      badge = `<span class="al-evt" style="background:rgba(239,65,121,.15);color:#ef4179;border:1px solid #ef4179">Tera</span>`;
      text  = `<b>${a.slot}</b> ${a.species || ''} terastallized → <b>${a.tera_type}</b>`;
    } else {
      badge = `<span class="al-evt" style="background:var(--surf2)">${a.event}</span>`;
      text  = JSON.stringify(a).slice(0, 80);
    }
    return `<div class="al-row">${badge}<span class="al-text">${text}</span></div>`;
  }).join('');
}

// ── Inject Stats Panel ───────────────────────────────────
function renderInjectPanel(el) {
  const b       = activeBattle;
  const allMons = [];
  for (const pid of ['p1', 'p2'])
    for (const species of (b.players[pid].roster || []))
      allMons.push({ pid, species });

  const cards = allMons.map(({ pid, species }) => {
    const key    = `${pid}:${species}`;
    const inj    = b.known_team_overrides?.[key] || {};
    const rev    = b.revealed_info?.[key] || {};
    const isOurs = pid === b.players.our_side;

    const hasManual = Object.values(inj).some(v =>
      v !== null && v !== undefined && v !== '' &&
      !(Array.isArray(v) && v.every(x => !x)) &&
      !(typeof v === 'object' && !Array.isArray(v) && Object.values(v).every(x => !x))
    );

    // Confirmed-from-replay chips
    // Bug 8: a mega mon has two ability contexts — show them separately.
    const revItem        = rev.known_item       || null;
    const revTera        = rev.known_tera_type  || null;
    const megaInVod      = !!rev.is_mega;
    const revBaseAbility = rev.pre_mega_ability || (!megaInVod ? rev.known_ability : null) || null;
    const revMegaAbility = rev.mega_ability     || null;
    const itemChip    = revItem        ? `<span class="rev-chip">🎒 ${escHtml(revItem)}</span>`                                        : '';
    const teraChip    = revTera        ? `<span class="rev-chip rev-tera">◈ Tera ${escHtml(revTera)}${rev.is_terastallized ? ' ✓' : ''}</span>` : '';
    const abilityChip = revBaseAbility ? `<span class="rev-chip rev-ability">⚡ ${escHtml(revBaseAbility)}</span>`                     : '';
    const megaChip    = revMegaAbility ? `<span class="rev-chip rev-mega-ability">Ⓜ ${escHtml(revMegaAbility)}</span>`                : '';

    const revMoves = rev.revealed_moves || [];

    // EV inputs
    const evHtml = STAT_NAMES.map(s => `
      <div class="ev-cell">
        <label>${STAT_LABELS[s]}</label>
        <input type="number" min="0" max="252" value="${inj.ev_spread?.[s] ?? ''}" placeholder="0"
          oninput="setInject('${key}','ev_${s}',this.value)" />
      </div>`).join('');

    // Move inputs — manual override wins; fall back to revealed move
    const moveHtml = [0, 1, 2, 3].map(i => {
      const manualVal   = inj.moves?.[i] ?? '';
      const revVal      = revMoves[i]     ?? '';
      const val         = manualVal || revVal;
      const isConfirmed = !manualVal && !!revVal;
      return `<div class="inj-move-slot">
        <input class="inj-move-in${isConfirmed ? ' rev-prefilled' : ''}" type="text"
          placeholder="Move ${i + 1}" value="${escHtml(val)}"
          oninput="setInject('${key}','move_${i}',this.value)" />
        ${isConfirmed ? '<span class="rev-move-badge" title="Confirmed in replay">VOD</span>' : ''}
      </div>`;
    }).join('');

    const itemVal      = inj.item    ?? revItem    ?? '';
    const itemPrefill  = !inj.item    && !!revItem;

    // ── Ability (BASE forme) ─────────────────────────────────────────────
    // The base ability is the only one the player chooses; the mega ability
    // is fixed by the mega forme (rendered read-only below).  Dropdown
    // options come from the pokedex — via the parse-time revealed_info if
    // present, else live from the loaded POKEDEX; free-text fallback offline.
    const abilityOptions = (rev.possible_abilities?.length ? rev.possible_abilities : dexAbilities(species)) || [];
    const abilityVal     = inj.ability ?? revBaseAbility ?? '';
    const abilityPrefill = !inj.ability && !!revBaseAbility;

    let abilityField;
    if (abilityOptions.length) {
      const opts = [...abilityOptions];
      if (abilityVal && !opts.includes(abilityVal)) opts.push(abilityVal);  // keep legacy/custom values selectable
      abilityField = `<select class="${abilityPrefill ? 'rev-prefilled' : ''}"
          onchange="setInject('${key}','ability',this.value)">
          <option value="">— unknown —</option>
          ${opts.map(a => `<option value="${escHtml(a)}"${abilityVal === a ? ' selected' : ''}>${escHtml(a)}</option>`).join('')}
        </select>`;
    } else {
      // Pokedex unavailable (offline) or species unknown to the dex
      abilityField = `<input type="text" placeholder="e.g. Fairy Aura"
          value="${escHtml(abilityVal)}"
          class="${abilityPrefill ? 'rev-prefilled' : ''}"
          oninput="setInject('${key}','ability',this.value)" />`;
    }

    // ── Mega ability (read-only — exactly one per mega forme) ────────────
    const megaFormes = (rev.mega_formes?.length ? rev.mega_formes : dexMegaFormes(species)) || [];
    let megaRow = '';
    if (megaInVod || megaFormes.length) {
      let megaText;
      if (megaInVod && revMegaAbility) {
        megaText = `<b>${escHtml(revMegaAbility)}</b>${rev.mega_species ? ` <span class="mega-forme-name">(${escHtml(rev.mega_species)})</span>` : ''}`;
      } else if (megaFormes.length) {
        megaText = megaFormes
          .map(f => `<b>${escHtml(f.ability || '?')}</b> <span class="mega-forme-name">(${escHtml(f.forme)})</span>`)
          .join(' · ');
      } else {
        megaText = '<span class="mega-forme-name">unknown — not in pokedex</span>';
      }
      megaRow = `<div class="inj-row mega-ability-row">
        <label>Mega ability${megaInVod ? ' <span class="rev-label-badge">MEGA’D IN VOD</span>' : ''}</label>
        <div class="mega-ability-fixed" title="Mega formes have exactly one ability — determined by the forme, not selectable">
          ${megaText} <span class="mega-lock">🔒 fixed</span>
        </div>
      </div>`;
    }

    return `<div class="inj-card">
      <div class="inj-card-hdr">
        <span class="side-dot ${pid}-dot"></span>
        <span>${species}</span>
        <span style="font-size:9px;color:var(--text3)">${pid.toUpperCase()}${isOurs ? ' · YOUR SIDE' : ''}</span>
        <span class="${hasManual ? 'inj-known' : 'inj-unknown'}">${hasManual ? '✓ Stats injected' : 'No stats yet'}</span>
      </div>

      ${(itemChip || teraChip || abilityChip || megaChip) ? `<div class="rev-chips">${itemChip}${abilityChip}${megaChip}${teraChip}</div>` : ''}

      <div class="inj-body">
        <div class="inj-row">
          <label>Nature</label>
          <select onchange="setInject('${key}','nature',this.value)">
            <option value="">— unknown —</option>
            ${NATURES.map(n => `<option value="${n}"${inj.nature === n ? ' selected' : ''}>${n}</option>`).join('')}
          </select>
        </div>
        <div class="inj-row">
          <label>Item${itemPrefill ? ' <span class="rev-label-badge">VOD</span>' : ''}</label>
          <input type="text" placeholder="e.g. Choice Specs"
            value="${escHtml(itemVal)}"
            class="${itemPrefill ? 'rev-prefilled' : ''}"
            oninput="setInject('${key}','item',this.value)" />
        </div>
        <div class="inj-row">
          <label>Ability${megaRow ? ' <span class="ability-ctx-badge">base forme</span>' : ''}${abilityPrefill ? ' <span class="rev-label-badge">VOD</span>' : ''}</label>
          ${abilityField}
        </div>
        ${megaRow}
        <div class="inj-section-label">EVs</div>
        <div class="ev-row">${evHtml}</div>
        <div class="inj-section-label">
          Moves
          ${revMoves.length ? `<span class="rev-label-badge">${revMoves.length} confirmed in VOD</span>` : ''}
        </div>
        <div class="inj-moves">${moveHtml}</div>
      </div>
    </div>`;
  }).join('');

  el.innerHTML = `<div id="inject-wrap">
    <h3>Manual Stat Injection
      <span style="color:var(--text3);font-size:10px;font-weight:400;text-transform:none;letter-spacing:0">
        Fill in known stats to upgrade Type B → Type A (your side) or Type C (bot side).
        <span class="rev-chip" style="margin-left:6px">VOD</span> = confirmed from replay log.
      </span>
    </h3>
    <div id="inject-grid">${cards}</div>
  </div>`;
}

// ── JSON Tab ─────────────────────────────────────────────
function renderJsonTab(el) {
  const json = buildExportJson();
  el.innerHTML = `<div style="position:relative">
    <button class="btn" style="position:absolute;top:8px;right:8px;z-index:2" onclick="copyJson()">📋 Copy</button>
    <pre id="json-pre" style="background:var(--bg2);border:1px solid var(--bdr);border-radius:var(--rad-lg);padding:14px;font-size:10px;line-height:1.6;overflow:auto;max-height:70vh;color:var(--text2)">${escHtml(JSON.stringify(json, null, 2))}</pre>
  </div>`;
}

// ── Notification toast ───────────────────────────────────
function showNotif(msg) {
  const n = document.getElementById('notif');
  n.textContent = msg;
  n.style.display = 'block';
  setTimeout(() => n.style.display = 'none', 2200);
}
