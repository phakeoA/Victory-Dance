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

// ── App state ─────────────────────────────────────────────
// Mutated by actions; read by render functions.
let battles      = [];
let activeBattle = null;
let activeTurnIdx = 0;
let activeTab    = 'turns';
