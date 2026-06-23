"""AUTO-GENERATED from pokemon-showdown/data/mods/champions/moves.ts — DO NOT EDIT BY HAND.

Regenerate:  python -m v_dance.encoders.gen_champions_move_overrides

v11 N1 — the custom Champions VGC format overrides these move fields; both encoders otherwise read
standard Gen-9 data. Champions-ONLY by construction (the battle encoder is hard-scoped to Champions);
revisit if a non-Champions battle format is ever encoded. Keys are move ids (norm_species(name) ==
poke-env Move.id), so the overlay is byte-identical on the offline + live encoders (parity)."""

CHAMP_BP = {  # basePower override (feeds band + Technician/-ate gates)
    "anchorshot": 90,
    "appleacid": 90,
    "astralbarrage": 110,
    "beakblast": 120,
    "bloodmoon": 130,
    "boltbeak": 80,
    "bonerush": 30,
    "dragonhammer": 100,
    "firelash": 90,
    "firstimpression": 100,
    "fishiousrend": 80,
    "geargrind": 60,
    "gravapple": 90,
    "hyperdrill": 120,
    "infernalparade": 65,
    "mountaingale": 120,
    "nightdaze": 90,
    "psyshieldbash": 90,
    "revelationdance": 100,
    "snipeshot": 85,
    "spiritshackle": 90,
    "tripledive": 35,
    "tropkick": 85,
}

CHAMP_TYPE = {  # type override (STAB + type-eff + immunity + one-hot)
    "growth": 'Grass',
    "snaptrap": 'Steel',
}

CHAMP_ACC = {  # accuracy override (percent int or True=always-hit)
    "clangoroussoul": True,
    "crabhammer": 95,
    "geargrind": 90,
    "makeitrain": 95,
    "syrupbomb": 90,
}

CHAMP_PP = {  # EMITTED, NOT consumed — pp is entangled with N5 (global PP cap) and low value; fold into N5
    "banefulbunker": 5,
    "beakblast": 5,
    "kingsshield": 5,
    "nightslash": 20,
    "nihillight": 5,
    "obstruct": 5,
    "protect": 5,
    "purify": 5,
    "sandstorm": 5,
    "shelltrap": 10,
    "snowscape": 5,
    "spikyshield": 5,
    "spinout": 10,
}
