# Delete_When_Project_Done

Archived **one-off debug / probe / verification harnesses** from completed work — kept for reference
in case a similar issue recurs, safe to delete once the project ships. None are run by pytest (they
don't match `test_*.py`) and none are imported by the real code or test suite (verified before
archiving), so removing this folder cannot break anything.

Origin (provenance preserved in subfolders):
- `tests/` — probes/audits from the encoder state-rep gaps and codec work, all DONE:
  gap #4 (`_gap4_*`), gap #5 (`_gap5_*`), gap #6 (`_gap6_*`, `_gap_all_verify`), opponent
  reconstruction (`_opp_*`), policy (`_probe_policy`), gimmick/mega (`_verify_gimmick_*`,
  `_verify_mega_*`, `_diag_mega_snapshot`). Superseded by the `test_*.py` suite.
- `local_battle/` — live-play one-offs: `_diag_final_verify`, `_diag_reject_count`,
  `_test_illusion_targeting` (superseded by `tests/test_illusion_targeting.py`), `_verify_splice`,
  `_vs_random`.

## Reusable harnesses kept OUT of here (still live, for recurring issues)
- `data/scripts/tests/_parity_harness.py` — offline↔live encoder byte-parity (needed on any
  state/encoder change; imported by `test_encoder_parity.py`).
- `local_battle/_smoke_zoroark.py` — Zoroark/illusion smoke (illusion bugs recur).
- `local_battle/_ab_headtohead.py` — checkpoint A/B head-to-head (for RL generation comparison).
- `local_battle/_diag_rejections.py` — order-rejection diagnosis (if MODEL-DRIVEN% ever regresses).
