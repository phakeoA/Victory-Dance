"""Per-bring / per-matchup win-rate diagnostics (task 3a.5).

Over a batch of self-play Trajectories (or their EpisodeMetas), tally win-rates by:
  - BRING  (which 4 mons were brought) — sec 14: which brings the battle AI actually
    WINS with (the team-preview co-development signal; never used as a reward/gate).
  - MATCHUP (team roster vs team roster) — sec 15: which matchups are lopsided.
  - LEADS  (which 2 were led).

PURE DIAGNOSTIC. Both perspectives of a game are independent data points (the winner's
bring scores a win, the loser's a loss), so the OVERALL rate is ~50% by symmetry — the
per-bring / per-matchup slices are the informative ones. Torch-free.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional, Tuple, Union

from self_play.schema import EpisodeMeta, Trajectory

Item = Union[Trajectory, EpisodeMeta]
Key = Tuple[str, ...]


def _meta(x: Item) -> EpisodeMeta:
    return x.meta if isinstance(x, Trajectory) else x


@dataclass
class Tally:
    games: int = 0
    wins: int = 0
    losses: int = 0
    nondecisive: int = 0      # won is None (draw / horizon-cut / fallback)

    def add(self, won: Optional[bool]) -> None:
        self.games += 1
        if won is True:
            self.wins += 1
        elif won is False:
            self.losses += 1
        else:
            self.nondecisive += 1

    @property
    def decisive(self) -> int:
        return self.wins + self.losses

    @property
    def win_rate(self) -> float:
        """Wins over DECISIVE games (draws/horizon/fallback excluded from the rate but
        counted in ``games``). 0.0 when no decisive games."""
        return self.wins / self.decisive if self.decisive else 0.0

    def to_obj(self) -> dict:
        return {"games": self.games, "wins": self.wins, "losses": self.losses,
                "nondecisive": self.nondecisive, "win_rate": round(self.win_rate, 4)}


def _bring_key(m: EpisodeMeta) -> Key:
    return tuple(sorted(m.own_team[i] for i in m.tp_bring if 0 <= i < len(m.own_team)))


def _lead_key(m: EpisodeMeta) -> Key:
    return tuple(sorted(m.own_team[i] for i in m.tp_leads if 0 <= i < len(m.own_team)))


def _team_key(species: List[str]) -> Key:
    return tuple(sorted(species))


def _grouped(trajs: Iterable[Item], keyfn: Callable[[EpisodeMeta], object]) -> Dict[object, Tally]:
    out: Dict[object, Tally] = {}
    for x in trajs:
        m = _meta(x)
        out.setdefault(keyfn(m), Tally()).add(m.won)
    return out


def bring_winrates(trajs: Iterable[Item]) -> Dict[Key, Tally]:
    return _grouped(trajs, _bring_key)


def lead_winrates(trajs: Iterable[Item]) -> Dict[Key, Tally]:
    return _grouped(trajs, _lead_key)


def matchup_winrates(trajs: Iterable[Item]) -> Dict[Tuple[Key, Key], Tally]:
    return _grouped(trajs, lambda m: (_team_key(m.own_team), _team_key(m.opp_team)))


def overall(trajs: Iterable[Item]) -> Tally:
    t = Tally()
    for x in trajs:
        t.add(_meta(x).won)
    return t


def summarize(trajs: Iterable[Item], *, min_games: int = 5, top: int = 5) -> dict:
    """JSON-able summary for per-generation logging: overall, the worst/best brings by
    win-rate (only brings with >= min_games), and the most lopsided matchups (furthest
    from 50%, >= min_games decisive). Materialises ``trajs`` once."""
    items = list(trajs)
    brings = bring_winrates(items)
    ranked = sorted(((k, v) for k, v in brings.items() if v.games >= min_games),
                    key=lambda kv: kv[1].win_rate)
    matchups = matchup_winrates(items)
    lopsided = sorted(((k, v) for k, v in matchups.items() if v.decisive >= min_games),
                      key=lambda kv: abs(kv[1].win_rate - 0.5), reverse=True)
    return {
        "overall": overall(items).to_obj(),
        "n_brings": len(brings),
        "n_matchups": len(matchups),
        "worst_brings": [{"bring": list(k), **v.to_obj()} for k, v in ranked[:top]],
        "best_brings": [{"bring": list(k), **v.to_obj()} for k, v in ranked[-top:][::-1]],
        "most_lopsided_matchups": [
            {"own": list(o), "opp": list(p), **v.to_obj()} for (o, p), v in lopsided[:top]],
    }
