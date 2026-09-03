/**
 * rlspawn.ts — server-side self-play battle spawning for Victory-Dance (the W2 throughput stack,
 * docs/ps_ppo_review_2026-09-02.md §4, step 1 — built 2026-09-03).
 *
 * A port of Nebraskinator/ps-ppo `pokemon-showdown/server/chat-plugins/rlspawn.ts` (MIT licence,
 * 2025) onto the smogon/pokemon-showdown commit this project pins (setup.sh). The SOURCE OF TRUTH is
 * `v_dance/selfplay/showdown_plugins/rlspawn.ts` in the Victory-Dance repo (the pokemon-showdown/
 * clone is gitignored); `python -m v_dance.selfplay.spawn_plugin --install` copies it into
 * `pokemon-showdown/server/chat-plugins/` and `node pokemon-showdown start` rebuilds it.
 *
 * Why: through the challenge protocol every self-play game costs a challenge/accept round-trip that
 * poke-env serialises per player, and one pairing plays one game at a time. With N battles kept
 * alive INSIDE the server between two bot accounts, collection is bounded by inference, not by
 * matchmaking (ps-ppo: 800 concurrent battles over 10 servers).
 *
 * Differences from upstream:
 *   * TEAMS — VGC formats need a team per player. Each account sets its team once with the server's
 *     own `/utm <packed>` (stored in `user.battleSettings.team`); the spawner passes that team to
 *     `Rooms.createBattle`. Upstream passed "" (random battles only).
 *   * `/rlstatus` (JSON reconciliation of schedulers + live counts) and `/rllifespan` (the GC
 *     lifespan; our VGC games can outlive his 5-minute default under inference latency).
 *   * Typed against this server version; defensive field reads kept where upstream had them.
 *
 * Commands (NO permission checks, exactly like upstream — for LOCAL `--no-security` servers only):
 *   /rlautospawn u1, u2, format, N   keep N un-ended battles alive between u1 and u2
 *                                    (50 ms tick, at most 50 new rooms per tick, N ≤ 4096)
 *   /rlautooff u1, u2, format        stop that scheduler (live rooms finish on their own)
 *   /rlactive                        → |queryresponse|rlactive|<csv of the caller's live battle rooms>
 *   /rlrescue <roomid>               expire a stalled room the caller is a player in
 *   /rlstatus                        → |queryresponse|rlstatus|<json>
 *   /rllifespan <seconds>            GC lifespan for plugin-born rooms (0 = never reap); default 600
 * GC: every 60 s, a plugin-born room older than the lifespan whose battle has not ended is expired.
 */

import type { Room } from '../rooms';
import { Rooms } from '../rooms';
import { Users } from '../users';

// ── helpers ──────────────────────────────────────────────────────────────────
function toID(text: any): string {
	if (text === null || text === undefined) return '';
	return ('' + text).toLowerCase().replace(/[^a-z0-9]+/g, '');
}

function getUser(name: string): any {
	const U: any = Users as any;
	return U.getExact?.(name) ?? U.get?.(name) ?? null;
}

function playerId(p: any): string {
	if (!p) return '';
	return toID(p.id ?? p.userid ?? p.user?.id ?? p.user?.userid ?? p.name ?? p.user?.name);
}

function roomPlayerIds(room: any): [string, string] {
	const battle = room?.battle;
	if (!battle) return ['', ''];
	const players: any[] = Array.isArray(battle.players) && battle.players.length >= 2 ?
		battle.players : [battle.p1, battle.p2];
	return [playerId(players[0]), playerId(players[1])];
}

function allRooms(): Map<string, Room> {
	const R: any = Rooms as any;
	return (R.rooms as Map<string, Room>) || new Map();
}

function liveBattleCountBetween(u1: any, u2: any): number {
	const x = toID(u1?.id ?? u1?.userid ?? u1?.name);
	const y = toID(u2?.id ?? u2?.userid ?? u2?.name);
	if (!x || !y) return 0;
	let n = 0;
	for (const room of allRooms().values()) {
		const battle: any = (room as any).battle;
		if (!battle || battle.ended) continue;
		const [a, b] = roomPlayerIds(room);
		if (!a || !b) continue;
		if ((a === x && b === y) || (a === y && b === x)) n++;
	}
	return n;
}

/** The packed team the account set with /utm (server-side `useteam`), '' = none / random. */
function teamOf(u: any): string {
	const t = u?.battleSettings?.team;
	return typeof t === 'string' ? t : '';
}

/** Create one battle between u1 and u2 in `format` and stamp its birth time (the GC key). */
function createBattleRoom(format: string, u1: any, u2: any): any {
	const R: any = Rooms as any;
	const room: any = R.createBattle({
		format: toID(format),
		rated: false,
		players: [{ user: u1, team: teamOf(u1) }, { user: u2, team: teamOf(u2) }],
	});
	if (!room) throw new Error('Rooms.createBattle returned nothing');
	room._rlBorn = Date.now();
	try { u1.joinRoom(room); } catch {}
	try { u2.joinRoom(room); } catch {}
	return room;
}

// ── schedulers ───────────────────────────────────────────────────────────────
interface Sched {
	u1: string; u2: string; format: string;
	target: number;
	timer: NodeJS.Timeout;
	created: number;
	errors: number;
	lastError: string;
}
const sched = new Map<string, Sched>();

function key(u1: string, u2: string, format: string): string {
	return `${toID(u1)}|${toID(u2)}|${toID(format)}`;
}

const TICK_MS = 50;
const BURST = 50;
const MAX_TARGET = 4096;
let RL_MAX_LIFESPAN_MS = 10 * 60 * 1000;

function tickFor(s: Sched) {
	return () => {
		const a = getUser(s.u1);
		const b = getUser(s.u2);
		if (!a || !b) return;                       // an account dropped — wait for it to come back
		const need = s.target - liveBattleCountBetween(a, b);
		if (need <= 0) return;
		const burst = Math.min(need, BURST);
		for (let i = 0; i < burst; i++) {
			try {
				createBattleRoom(s.format, a, b);
				s.created++;
			} catch (e: any) {
				s.errors++;
				s.lastError = String(e?.message ?? e);
				console.error(`[rlautospawn] create error (${s.u1} vs ${s.u2}, ${s.format}): ${s.lastError}`);
				return;
			}
		}
	};
}

function statusJSON(user: any): string {
	const schedulers = [];
	for (const s of sched.values()) {
		const a = getUser(s.u1);
		const b = getUser(s.u2);
		schedulers.push({
			u1: toID(s.u1), u2: toID(s.u2), format: s.format, target: s.target,
			live: (a && b) ? liveBattleCountBetween(a, b) : 0,
			online: !!(a && b), created: s.created, errors: s.errors, lastError: s.lastError,
		});
	}
	let born = 0, mine = 0;
	for (const room of allRooms().values()) {
		const battle: any = (room as any).battle;
		if (!battle || battle.ended) continue;
		if ((room as any)._rlBorn) born++;
		const [a, b] = roomPlayerIds(room);
		if (a === user.id || b === user.id) mine++;
	}
	return JSON.stringify({
		schedulers, live_plugin_rooms: born, my_live_rooms: mine,
		rooms: allRooms().size, lifespan_s: RL_MAX_LIFESPAN_MS / 1000, tick_ms: TICK_MS, burst: BURST,
	});
}

// ── commands ─────────────────────────────────────────────────────────────────
export const commands: Chat.ChatCommands = {
	rlautospawn(target, room, user) {
		const parts = target.split(',').map(s => s.trim()).filter(Boolean);
		if (parts.length < 4) {
			return this.errorReply('Usage: /rlautospawn user1, user2, format, N');
		}
		const [u1n, u2n, formatRaw, targetStr] = parts;
		const format = toID(formatRaw);
		const tgt = Math.max(1, Math.min(MAX_TARGET, parseInt(targetStr, 10) || 1));
		const u1 = getUser(u1n);
		const u2 = getUser(u2n);
		if (!u1 || !u2) return this.errorReply(`/rlautospawn: ${!u1 ? u1n : u2n} is not online`);
		const k = key(u1n, u2n, format);
		const existing = sched.get(k);
		if (existing) {
			existing.target = tgt;
			return this.sendReply(`[rlautospawn] ${toID(u1n)} vs ${toID(u2n)} (${format}): target -> ${tgt}`);
		}
		const s: Sched = {
			u1: u1n, u2: u2n, format, target: tgt, timer: null as any,
			created: 0, errors: 0, lastError: '',
		};
		s.timer = setInterval(tickFor(s), TICK_MS);
		sched.set(k, s);
		this.sendReply(`[rlautospawn] started ${toID(u1n)} vs ${toID(u2n)} (${format}): keep ${tgt} live`);
	},

	rlautooff(target, room, user) {
		const parts = target.split(',').map(s => s.trim()).filter(Boolean);
		if (parts.length < 3) return this.errorReply('Usage: /rlautooff user1, user2, format');
		const k = key(parts[0], parts[1], parts[2]);
		const s = sched.get(k);
		if (!s) return this.errorReply('[rlautooff] no such scheduler');
		clearInterval(s.timer);
		sched.delete(k);
		this.sendReply(`[rlautooff] stopped ${toID(parts[0])} vs ${toID(parts[1])} (${toID(parts[2])})`);
	},

	/** CSV of the caller's live battle rooms (compact: ~60 KB for 4096 ids, one frame). */
	rlactive(target, room, user) {
		const battles: string[] = [];
		for (const r of allRooms().values()) {
			const battle: any = (r as any).battle;
			if (!battle || battle.ended) continue;
			const [a, b] = roomPlayerIds(r);
			if (a === user.id || b === user.id) battles.push(r.roomid);
		}
		user.send(`|queryresponse|rlactive|${battles.join(',')}`);
	},

	/** Expire a stalled room the caller plays in (a forfeit-by-timer never comes without a timer). */
	rlrescue(target, room, user) {
		const targetRoom: any = (Rooms as any).get(toID(target));
		if (!targetRoom) return this.errorReply('[rlrescue] no such room');
		if (targetRoom.battle) {
			const [a, b] = roomPlayerIds(targetRoom);
			if (user.id !== a && user.id !== b) return this.errorReply('[rlrescue] you are not a player there');
		}
		try {
			targetRoom.expire();
		} catch (e: any) {
			console.error(`[rlrescue] expire failed for ${targetRoom.roomid}: ${e?.message}`);
			if (targetRoom.destroy) targetRoom.destroy();
		}
		this.sendReply(`[rlrescue] expired ${targetRoom.roomid}`);
	},

	rlstatus(target, room, user) {
		user.send(`|queryresponse|rlstatus|${statusJSON(user)}`);
	},

	rllifespan(target, room, user) {
		const sec = parseInt(target.trim(), 10);
		if (isNaN(sec) || sec < 0) return this.errorReply('Usage: /rllifespan <seconds> (0 = never reap)');
		RL_MAX_LIFESPAN_MS = sec * 1000;
		this.sendReply(`[rllifespan] plugin-born rooms are reaped after ${sec} s${sec ? '' : ' (never)'}`);
	},
};

// ── garbage collection of plugin-born rooms ──────────────────────────────────
setInterval(() => {
	if (!RL_MAX_LIFESPAN_MS) return;
	const now = Date.now();
	let reaped = 0;
	for (const room of allRooms().values()) {
		const born = (room as any)._rlBorn;
		if (!born || now - born <= RL_MAX_LIFESPAN_MS) continue;
		const battle: any = (room as any).battle;
		if (!battle || battle.ended) continue;
		try {
			(room as any).expire();
		} catch {
			if ((room as any).destroy) (room as any).destroy();
		}
		reaped++;
	}
	if (reaped) console.log(`[rlspawn GC] reaped ${reaped} stalled plugin room(s)`);
}, 60 * 1000);
