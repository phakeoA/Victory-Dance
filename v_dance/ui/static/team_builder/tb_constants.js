// ═══════════════════════════════════════════════════════════
// tb_constants.js
// Constants, pure helpers, and shared app state.
// Must be loaded first — all other modules depend on this.
// ═══════════════════════════════════════════════════════════

const SERVER = 'http://localhost:5174';

const NATURES = [
  'Hardy', 'Lonely', 'Brave', 'Adamant', 'Naughty',
  'Bold', 'Docile', 'Relaxed', 'Impish', 'Lax',
  'Timid', 'Hasty', 'Serious', 'Jolly', 'Naive',
  'Modest', 'Mild', 'Quiet', 'Bashful', 'Rash',
  'Calm', 'Gentle', 'Sassy', 'Careful', 'Quirky',
];
const STAT_NAMES  = ['hp', 'atk', 'def', 'spa', 'spd', 'spe'];
const STAT_LABELS = { hp: 'HP', atk: 'Atk', def: 'Def', spa: 'SpA', spd: 'SpD', spe: 'Spe' };

// ── Pure helpers ─────────────────────────────────────────
function hpColor(pct) {
  if (pct > 50) return 'hp-hi';
  if (pct > 20) return 'hp-mid';
  return 'hp-lo';
}

function escHtml(s) {
  return s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
}

// ── Pokedex helpers (Bug 8: mega ability split) ──────────
// POKEDEX is loaded asynchronously from the server (tb_api.loadPokedex).
// All helpers are null-safe so the UI degrades gracefully offline.
function dexKey(species) {
  return (species || '').toLowerCase().replace(/[^a-z0-9]/g, '');
}

function dexEntry(species) {
  return POKEDEX ? (POKEDEX[dexKey(species)] || null) : null;
}

/** All possible abilities for a forme, in dex slot order (0, 1, H, S). */
function dexAbilities(species) {
  const e = dexEntry(species);
  if (!e || !e.abilities) return [];
  const order = ['0', '1', 'H', 'S'];
  const out = order.filter(k => e.abilities[k]).map(k => e.abilities[k]);
  for (const [k, v] of Object.entries(e.abilities))
    if (!order.includes(k) && v && !out.includes(v)) out.push(v);
  return out;
}

/** True if `species` IS a mega forme. Falls back to a name check offline. */
function isMegaSpecies(species) {
  const e = dexEntry(species);
  if (e) return (e.forme || '').startsWith('Mega');
  return !!species && (species.endsWith('-Mega') || species.includes('-Mega-'));
}

/**
 * The single fixed ability of a mega forme, or null.
 * Mega formes always have exactly ONE ability — if the dex lists more
 * than one we refuse to guess.
 */
function dexMegaAbility(megaSpecies) {
  if (!isMegaSpecies(megaSpecies)) return null;
  const abilities = dexAbilities(megaSpecies);
  return abilities.length === 1 ? abilities[0] : null;
}

/**
 * All mega formes reachable from a (base or non-base) species.
 * Returns [{forme, ability}, ...] — two entries for X/Y twin megas.
 */
function dexMegaFormes(species) {
  let e = dexEntry(species);
  if (!e) return [];
  if (e.baseSpecies) e = dexEntry(e.baseSpecies) || e;  // resolve formes to base
  const out = [];
  for (const formeName of e.otherFormes || []) {
    const fe = dexEntry(formeName);
    if (fe && (fe.forme || '').startsWith('Mega'))
      out.push({ forme: formeName, ability: dexMegaAbility(formeName) });
  }
  return out;
}

// ── Belief auto-fill helpers ─────────────────────────────
// b.belief_autofill[key] tracks which inject fields were filled by the
// "Use Belief Integration" button: { nature, item, ability, ev_spread: bool,
// moves: [bool×4] }.  A manual edit revokes the flag (setInject), so the
// field stops being "auto" the moment the user touches it.

/**
 * Card-level badge state for one `pid:species` key, considering both the
 * injected values and which of them are still belief-auto-filled.
 * Used by renderInjectPanel and setInject's live badge refresh.
 */
function injBadge(key) {
  const inj  = activeBattle?.known_team_overrides?.[key] || {};
  const auto = activeBattle?.belief_autofill?.[key] || {};
  const set  = [];   // one bool per populated field group: is it auto-filled?
  if (inj.nature)  set.push(!!auto.nature);
  if (inj.item)    set.push(!!auto.item);
  if (inj.ability) set.push(!!auto.ability);
  if (inj.ev_spread && Object.values(inj.ev_spread).some(v => v)) set.push(!!auto.ev_spread);
  for (let i = 0; i < (inj.moves || []).length; i++)
    if (inj.moves[i]) set.push(!!(auto.moves && auto.moves[i]));
  if (!set.length)        return { cls: 'inj-unknown', txt: 'No stats yet' };
  if (set.every(Boolean)) return { cls: 'inj-auto',    txt: '✨ Auto-filled' };
  if (set.some(Boolean))  return { cls: 'inj-known',   txt: '✓ Injected · partly ✨' };
  return { cls: 'inj-known', txt: '✓ Stats injected' };
}

/** Tooltip text for an AUTO badge, e.g. "Auto-filled from Pikalytics (30%) — edit to override". */
function autoTitle(p) {
  const pct = (p !== null && p !== undefined) ? ` (${Math.round(p * 100)}% usage)` : '';
  return `Auto-filled from Pikalytics${pct} — editing this field makes it manual`;
}

// ── App state ─────────────────────────────────────────────
// Mutated by actions; read by render functions.
let battles      = [];
let activeBattle = null;
let activeTurnIdx = 0;
let activeTab    = 'turns';
let POKEDEX      = null;   // species-keyed dex, loaded from /data/pokedex.json
