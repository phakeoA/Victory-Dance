"""Install the server-side spawn plugin into the pinned Showdown clone — W2 throughput step 1
(2026-09-03, docs/ps_ppo_review_2026-09-02.md §4).

    python -m v_dance.selfplay.spawn_plugin --check      # installed and current?
    python -m v_dance.selfplay.spawn_plugin --install    # copy (no-op when current)
    python -m v_dance.selfplay.spawn_plugin --install --build   # + `node build` now (the server start
                                                                #   rebuilds anyway)

The plugin's SOURCE OF TRUTH is ``v_dance/selfplay/showdown_plugins/rlspawn.ts`` (this repo);
``pokemon-showdown/`` is a gitignored clone (setup.sh), so a fresh clone needs this install once.
``node pokemon-showdown start`` runs ``node build`` on every start (the launcher), which transpiles
a new/changed chat plugin — nothing else to do after copying. Torch-/poke-env-free on purpose.
"""
from __future__ import annotations

import argparse
import logging
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)

_REPO = Path(__file__).resolve().parents[2]
SHOWDOWN_DIR = _REPO / "pokemon-showdown"
PLUGIN_NAME = "rlspawn.ts"
PLUGIN_SRC = Path(__file__).resolve().parent / "showdown_plugins" / PLUGIN_NAME
COMMANDS = ("rlautospawn", "rlautooff", "rlactive", "rlrescue", "rlstatus", "rllifespan")


def plugin_dest(showdown_dir=SHOWDOWN_DIR) -> Path:
    return Path(showdown_dir) / "server" / "chat-plugins" / PLUGIN_NAME


def compiled_path(showdown_dir=SHOWDOWN_DIR) -> Path:
    """Where the launcher's build puts the transpiled plugin (exists only after a build)."""
    return Path(showdown_dir) / "dist" / "server" / "chat-plugins" / PLUGIN_NAME.replace(".ts", ".js")


def is_installed(showdown_dir=SHOWDOWN_DIR) -> bool:
    """True when the clone carries THIS repo's plugin source byte-for-byte."""
    dest = plugin_dest(showdown_dir)
    return dest.is_file() and dest.read_bytes() == PLUGIN_SRC.read_bytes()


def install_rlspawn(showdown_dir=SHOWDOWN_DIR, *, build: bool = False, node: Optional[str] = None) -> dict:
    """Copy the plugin into the clone when missing or stale; optionally run ``node build``.
    Returns ``{"dest", "changed", "installed", "built"}``. Raises when the clone is absent."""
    showdown_dir = Path(showdown_dir)
    dest = plugin_dest(showdown_dir)
    if not (showdown_dir / "server" / "chat-plugins").is_dir():
        raise FileNotFoundError(f"pokemon-showdown clone not found at {showdown_dir} (run setup.sh)")
    changed = not is_installed(showdown_dir)
    if changed:
        shutil.copyfile(PLUGIN_SRC, dest)
        log.info("rlspawn plugin installed -> %s", dest)
    built = False
    if build:
        built = build_server(showdown_dir, node=node)
    return {"dest": str(dest), "changed": changed, "installed": True, "built": built}


def build_server(showdown_dir=SHOWDOWN_DIR, *, node: Optional[str] = None) -> bool:
    """``node build`` in the clone (incremental; the launcher does the same at every start)."""
    if node is None:
        from v_dance.play.run_local_battle import _find_node
        node = _find_node()
    r = subprocess.run([node, "build"], cwd=str(showdown_dir), capture_output=True, text=True,
                       encoding="utf-8", errors="replace")
    if r.returncode != 0:
        log.error("node build failed (%d):\n%s\n%s", r.returncode, r.stdout[-2000:], r.stderr[-2000:])
        return False
    return True


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--showdown-dir", default=str(SHOWDOWN_DIR))
    ap.add_argument("--check", action="store_true", help="report whether the clone carries the current plugin")
    ap.add_argument("--install", action="store_true", help="copy the plugin into the clone (no-op when current)")
    ap.add_argument("--build", action="store_true", help="run `node build` after installing")
    args = ap.parse_args(argv)
    sd = Path(args.showdown_dir)
    if args.install:
        res = install_rlspawn(sd, build=args.build)
        print(f"[spawn_plugin] {'copied' if res['changed'] else 'already current'} -> {res['dest']}"
              + (f"; build {'OK' if res['built'] else 'FAILED'}" if args.build else
                 "; the next `node pokemon-showdown start` rebuilds it"))
        return 0 if (not args.build or res["built"]) else 1
    inst = is_installed(sd)
    comp = compiled_path(sd)
    print(f"[spawn_plugin] source {PLUGIN_SRC}\n[spawn_plugin] clone   {plugin_dest(sd)} — "
          f"{'CURRENT' if inst else 'MISSING / STALE (run --install)'}\n"
          f"[spawn_plugin] built   {comp} — {'present' if comp.is_file() else 'not built yet'}")
    return 0 if inst else 1


if __name__ == "__main__":
    sys.exit(main())
