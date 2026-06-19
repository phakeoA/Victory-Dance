/**
 * enum_champions_roster.js — enumerate the FULL legal roster of a Pokémon
 * Champions doubles format from the LOCAL Showdown clone (no Smogon dependency).
 *
 * WHY: Pikalytics' index page only renders the top ~50 mons, so the scraper used
 * Smogon's monthly stats file to enumerate the full roster. Smogon has no stats
 * for a freshly-launched reg (e.g. Reg M-B), so we enumerate from the format's
 * own mod (which ships inside the pinned pokemon-showdown checkout and updates in
 * lockstep with each reg drop — future-reg-proof).
 *
 * Roster granularity = ALL in-format FORME keys (NOT collapsed to base): the M-A
 * Pikalytics scrape is forme-inclusive (Rotom-Wash, Charizard-Mega-Y, custom
 * Champions megas like Floette-Mega), and collapsing to base would silently drop
 * distinct formes + the mega-ability belief prior. All-formes is the complete
 * superset (M-B 'champions' mod = 310 entries, covering all 272 M-A keys).
 *
 * Output: JSON {toID: displayName} on stdout. The toID is the Pikalytics URL slug
 * (lowercase, punctuation stripped); the displayName matches the M-A scrape's keys
 * EXACTLY (so downstream name-keyed code stays consistent).
 *
 * Usage:  node enum_champions_roster.js <formatId>
 *   e.g.  node enum_champions_roster.js gen9championsvgc2026regmb
 *         node enum_champions_roster.js gen9championsvgc2026regma
 */
'use strict';
const path = require('path');

const PS_DIR = path.resolve(__dirname, '..', '..', '..', 'pokemon-showdown');
const PS = require(PS_DIR);

const formatId = (process.argv[2] || 'gen9championsvgc2026regmb').trim();
const fmt = PS.Dex.formats.get(formatId);
if (!fmt || !fmt.exists) {
  process.stderr.write(`unknown format: ${formatId}\n`);
  process.exit(2);
}

// Resolve the format's MOD (champions = current/M-B, championsregma = frozen M-A),
// then read its legality table (FormatsData): an entry is in-format when it has a
// real tier that isn't Illegal/Unreleased and isn't flagged isNonstandard.
const dex = PS.Dex.mod(fmt.mod || 'gen9');
const fd = dex.data.FormatsData;
const ILLEGAL = new Set(['Illegal', 'Unreleased']);

const out = {};
for (const key of Object.keys(fd)) {
  const e = fd[key];
  if (!e || !e.tier || ILLEGAL.has(e.tier) || e.isNonstandard) continue;
  const sp = dex.species.get(key);
  if (!sp || !sp.exists) continue;
  out[sp.id] = sp.name;            // id = Pikalytics slug; name = output key (M-A-consistent)
}

const ids = Object.keys(out);
if (ids.length === 0) {
  process.stderr.write(`no in-format species found for ${formatId} (mod ${fmt.mod})\n`);
  process.exit(3);
}
process.stderr.write(`[enum] ${formatId} (mod ${fmt.mod}): ${ids.length} in-format species\n`);
process.stdout.write(JSON.stringify(out));
