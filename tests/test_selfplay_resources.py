"""Task 3c.8a: ResourceBudget compute caps (docs sec 20) — pure resolvers + the
enforcement side effects (torch monkeypatched, so no GPU is needed)."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[1]
from v_dance.selfplay.resources import (ResourceBudget, apply_resource_budget,  # noqa: E402
                                        physical_cpu_count, summarize)


# ── n_workers (CPU cap) ───────────────────────────────────────────────────────
def test_n_workers_fraction_floor():
    assert ResourceBudget(max_cpu_fraction=0.5).n_workers(cpu_count=12) == 6      # 5900X @ 50%
    assert ResourceBudget(max_cpu_fraction=0.5).n_workers(cpu_count=11) == 5      # floor, not round
    assert ResourceBudget(max_cpu_fraction=1.0).n_workers(cpu_count=12) == 12


def test_n_workers_max_cores_overrides_fraction():
    assert ResourceBudget(max_cpu_fraction=0.5, max_cores=8).n_workers(cpu_count=12) == 8


def test_n_workers_fails_loud_when_below_one():
    with pytest.raises(ValueError):
        ResourceBudget(max_cpu_fraction=0.0).n_workers(cpu_count=12)
    with pytest.raises(ValueError):
        ResourceBudget(max_cores=0).n_workers(cpu_count=12)


def test_physical_cpu_count_at_least_one():
    assert physical_cpu_count() >= 1


# ── device resolution ─────────────────────────────────────────────────────────
def test_resolve_device_auto_and_explicit():
    assert ResourceBudget(device="auto").resolve_device(cuda_available=True) == "cuda"
    assert ResourceBudget(device="auto").resolve_device(cuda_available=False) == "cpu"
    assert ResourceBudget(device="cpu").resolve_device(cuda_available=True) == "cpu"
    assert ResourceBudget(device="cuda").resolve_device(cuda_available=True) == "cuda"


def test_resolve_device_cuda_unavailable_fails_loud():
    with pytest.raises(ValueError):
        ResourceBudget(device="cuda").resolve_device(cuda_available=False)


def test_resolve_device_unknown_fails():
    with pytest.raises(ValueError):
        ResourceBudget(device="gpu").resolve_device(cuda_available=True)


# ── VRAM fraction ─────────────────────────────────────────────────────────────
def test_vram_fraction_basic():
    assert ResourceBudget(max_vram_gb=None).vram_fraction(8.0) is None
    assert ResourceBudget(max_vram_gb=4.0).vram_fraction(8.0) == pytest.approx(0.5)   # 4/8 GB


def test_vram_fraction_exceeding_card_fails_loud():
    with pytest.raises(ValueError):                       # typo guard (40 GB on an 8 GB card)
        ResourceBudget(max_vram_gb=40.0).vram_fraction(8.0)


# ── apply (enforcement; torch monkeypatched, no GPU required) ──────────────────
def test_apply_cpu_path_sets_threads(monkeypatch):
    pytest.importorskip("torch")
    import torch
    rec = {}
    monkeypatch.setattr(torch, "set_num_threads", lambda k: rec.setdefault("threads", k))
    out = apply_resource_budget(ResourceBudget(max_cores=4, device="cpu"),
                                set_threads=True, cuda_available=False)
    assert rec["threads"] == 4
    assert out["device"] == "cpu" and out["workers"] == 4 and out["vram"] is None


def test_apply_cuda_vram_enforced_vs_report_only(monkeypatch):
    pytest.importorskip("torch")
    from types import SimpleNamespace
    import torch
    rec = {}
    monkeypatch.setattr(torch, "set_num_threads", lambda k: None)
    monkeypatch.setattr(torch.cuda, "get_device_properties",
                        lambda i: SimpleNamespace(total_memory=8 * 1024 ** 3))   # 8 GB card
    monkeypatch.setattr(torch.cuda, "set_per_process_memory_fraction",
                        lambda f, d=0: rec.setdefault("frac", f))

    # ENFORCED (3c.8b path): the fraction is actually set.
    out = apply_resource_budget(ResourceBudget(max_cores=4, max_vram_gb=4.0, device="cuda"),
                                cuda_available=True, enforce_vram=True)
    assert rec.get("frac") == pytest.approx(0.5)
    assert out["device"] == "cuda"
    assert out["vram"]["fraction"] == pytest.approx(0.5) and out["vram"]["enforced"] is True

    # REPORT-ONLY (3c.8a path): fraction computed for the banner but NOT set on the GPU.
    rec.clear()
    out2 = apply_resource_budget(ResourceBudget(max_cores=4, max_vram_gb=4.0, device="cuda"),
                                 cuda_available=True, enforce_vram=False)
    assert "frac" not in rec
    assert out2["vram"]["fraction"] == pytest.approx(0.5) and out2["vram"]["enforced"] is False


def test_summarize_is_one_line():
    s = summarize({"workers": 6, "physical_cores": 12, "logical_cores": 24,
                   "device": "cpu", "vram": None})
    assert "workers=6" in s and "device=cpu" in s and "\n" not in s
