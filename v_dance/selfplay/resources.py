"""Resource budget — user-settable compute caps (task 3c.8 / docs sec 20).

A long, unattended training run must NEVER peg the machine. ``ResourceBudget`` is a pure
config with resolvers (testable with no torch / no CUDA); ``apply_resource_budget`` enforces
it. Per sec 20:

  * the dominant throughput lever is **CPU-parallel collection** (3c.8c) — the worker count
    is ``n_workers`` here;
  * the GPU's clean win is the **PPO update only** (3c.8b) — ``resolve_device`` /
    ``vram_fraction`` size that;
  * per-game inference stays on the CPU (tiny model + async play => GPU transfer overhead
    loses), so the device cap targets the UPDATE, not collection.

Defaults LEAVE HEADROOM (half the box) so the user can keep working. Caps that resolve
impossibly tight (0 workers, an unknown device) FAIL LOUD rather than silently.
"""
from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Optional


def physical_cpu_count() -> int:
    """Physical cores (psutil) if available, else ``os.cpu_count()`` (logical). 'Half the
    CPU' is most intuitive in PHYSICAL cores — on a 12-core/24-thread chip the user means 6,
    not 12 — so the fraction is taken against physical cores when we can detect them."""
    try:
        import psutil
        n = psutil.cpu_count(logical=False)
        if n:
            return int(n)
    except Exception:
        pass
    return int(os.cpu_count() or 1)


@dataclass
class ResourceBudget:
    """User-settable caps for a long run (sec 20).

    ``max_cpu_fraction`` (0..1) OR ``max_cores`` (explicit, wins if set) size the collection
    worker count + torch intra-op threads; ``max_vram_gb`` caps the GPU memory the update may
    use; ``device`` is ``auto`` | ``cpu`` | ``cuda``."""
    max_cpu_fraction: Optional[float] = 0.5     # half the box by default (headroom to keep working)
    max_cores: Optional[int] = None             # explicit worker/thread count (overrides the fraction)
    max_vram_gb: Optional[float] = None         # GPU memory ceiling for the update (None = uncapped)
    device: str = "auto"                        # auto | cpu | cuda

    def n_workers(self, cpu_count: Optional[int] = None) -> int:
        """Concurrent-collection workers + torch threads. ``floor(fraction * cores)`` or the
        explicit ``max_cores``, clamped >=1; FAILS LOUD if it resolves to <1."""
        cc = cpu_count if cpu_count is not None else physical_cpu_count()
        if self.max_cores is not None:
            k = int(self.max_cores)
        else:
            frac = self.max_cpu_fraction if self.max_cpu_fraction is not None else 1.0
            k = int(math.floor(frac * cc))
        if k < 1:
            raise ValueError(
                f"ResourceBudget resolves to {k} workers (cores={cc}, "
                f"fraction={self.max_cpu_fraction}, max_cores={self.max_cores}); need >=1. "
                f"Raise --max-cpu-fraction or set --max-cores.")
        return k

    def resolve_device(self, cuda_available: Optional[bool] = None) -> str:
        """``auto`` -> cuda if available else cpu; ``cuda`` FAILS LOUD if unavailable; ``cpu``
        forces CPU. ``cuda_available`` is injectable for tests."""
        if cuda_available is None:
            try:
                import torch
                cuda_available = bool(torch.cuda.is_available())
            except Exception:
                cuda_available = False
        if self.device == "cpu":
            return "cpu"
        if self.device == "cuda":
            if not cuda_available:
                raise ValueError("device='cuda' requested but CUDA is not available "
                                 "(use --device auto to fall back to CPU)")
            return "cuda"
        if self.device == "auto":
            return "cuda" if cuda_available else "cpu"
        raise ValueError(f"unknown device {self.device!r} (use auto | cpu | cuda)")

    def vram_fraction(self, total_gb: float) -> Optional[float]:
        """Fraction of total VRAM the process may use, or None (uncapped). Clamped to 1.0;
        FAILS LOUD if the cap exceeds the card (a typo like 40 GB on an 8 GB card)."""
        if self.max_vram_gb is None or total_gb <= 0:
            return None
        if self.max_vram_gb > total_gb:
            raise ValueError(f"max_vram_gb={self.max_vram_gb} exceeds the GPU's {total_gb:.1f} GB")
        return min(1.0, float(self.max_vram_gb) / float(total_gb))


def apply_resource_budget(budget: ResourceBudget, *, set_threads: bool = True,
                          enforce_vram: bool = True,
                          cuda_available: Optional[bool] = None) -> dict:
    """Enforce ``budget`` and return a resolved summary dict (for logging). Side effects:
    ``torch.set_num_threads(n_workers)`` (bounds the update's CPU threads + leaves the box
    usable) and, when the resolved device is cuda AND ``enforce_vram``,
    ``torch.cuda.set_per_process_memory_fraction`` (so allocations beyond the cap fail loudly
    instead of ballooning). ``enforce_vram=False`` still COMPUTES the VRAM plan (for reporting)
    without engaging the GPU — used by 3c.8a, which runs on CPU; 3c.8b enforces it. Lazy torch
    import so the config stays usable without torch."""
    import torch

    cores = physical_cpu_count()
    workers = budget.n_workers(cores)
    if set_threads:
        torch.set_num_threads(workers)
    device = budget.resolve_device(cuda_available)
    vram = None
    if device == "cuda":
        total_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        frac = budget.vram_fraction(total_gb)
        if frac is not None:
            if enforce_vram:
                torch.cuda.set_per_process_memory_fraction(float(frac), 0)
            vram = {"cap_gb": float(budget.max_vram_gb), "total_gb": round(total_gb, 2),
                    "fraction": round(frac, 4), "enforced": bool(enforce_vram)}
    return {"workers": workers, "physical_cores": cores, "logical_cores": os.cpu_count(),
            "device": device, "vram": vram}


def summarize(summary: dict) -> str:
    """One-line human banner for the resolved budget."""
    v = summary.get("vram")
    vram_s = (f"VRAM cap {v['cap_gb']}/{v['total_gb']}GB (frac {v['fraction']})"
              if v else "VRAM uncapped")
    return (f"workers={summary['workers']} "
            f"(of {summary['physical_cores']} physical/{summary['logical_cores']} logical cores)  "
            f"device={summary['device']}  {vram_s}")
