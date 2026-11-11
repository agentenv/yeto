"""GPU memory guard of a sampler node: high-VRAM alerts and the in-place reload check.

Measured on a K3.1 MI355X TP8 agentic sampler: the serving path grows ~20 GiB per
GPU past vLLM's ``--gpu-memory-utilization`` budget (allocator growth, collective
scratch), and an in-place expert reload stages each layer's released experts and
their kernel-format copy on the same GPUs. At 0.88 that ended at 99.8% and a
SIGKILLed sampler (exit 137). Two guards, both on per-card readings of the card
tool (``yeto.hw.card_telemetry``: amd-smi / rocm-smi / sysfs, NVML / nvidia-smi):

* :class:`VramWatch` (fed by ``telemetry.ResourceSampler`` every sample): a card
  above ``high_frac`` (default 0.97) writes ``rl_sampler_vram_high`` (again every
  ``realert_s`` while it stays high) and ``rl_sampler_vram_ok`` once every card is
  back under; the pool probe and the dashboard read both from the tape;
* :func:`reload_plan` before an in-place reload: the free memory of the fullest
  card against the staging the release needs (:func:`staging_bytes`: the largest
  per-layer block of released experts, per TP rank, times the staging + converted
  copies, plus a reserve). Short: the caller empties the engine's caching
  allocator and reads again; still short: the reload goes in smaller per-layer
  chunks that fit (``chunked``), and when not even one expert per chunk fits, by
  a restart (``restart``: the engine stops, which frees everything). It never
  stages into memory it did not see free.

Nothing here imports torch; the card readings are injected (tests use fakes).
"""

from __future__ import annotations

import json
import math
import struct
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

VRAM_HIGH_EVENT = "rl_sampler_vram_high"
VRAM_OK_EVENT = "rl_sampler_vram_ok"
DEFAULT_HIGH_FRAC = 0.97
DEFAULT_REALERT_S = 600.0
MIB = 2**20
GIB = 2**30
# the reload's GPU copies of one layer's released experts: the checkpoint-format staging
# tensors (``vllm_expert_reload._staging``) and the kernel-format conversion (``_convert``)
STAGING_COPIES = 2
DEFAULT_RESERVE_BYTES = 2 * GIB  # left free for the serving path while the reload runs

CardMemory = Callable[[], Sequence[Mapping[str, Any]]]  # -> card samples (mem_used_mb, mem_total_mb)


def card_fractions(cards: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """``[{"index", "used_mb", "total_mb", "frac", "free_bytes"}]`` of the cards that
    report both used and total memory."""
    out = []
    for i, card in enumerate(cards):
        used, total = card.get("mem_used_mb"), card.get("mem_total_mb")
        if used is None or total is None or float(total) <= 0:
            continue
        used, total = float(used), float(total)
        out.append({"index": card.get("index", i), "used_mb": used, "total_mb": total,
                    "frac": round(used / total, 4), "free_bytes": int(max(0.0, total - used) * MIB)})
    return out


def min_free_bytes(cards: Iterable[Mapping[str, Any]]) -> int | None:
    """The free memory of the fullest card (None: no card reported memory)."""
    rows = card_fractions(cards)
    return min((r["free_bytes"] for r in rows), default=None)


class VramWatch:
    """Edge-triggered high-VRAM alert over card samples (module doc)."""

    def __init__(self, emit: Callable[..., None], *, high_frac: float = DEFAULT_HIGH_FRAC,
                 realert_s: float = DEFAULT_REALERT_S, clock: Callable[[], float] = time.monotonic,
                 labels: Callable[[], Mapping[str, Any]] | None = None) -> None:
        if not 0.0 < float(high_frac) <= 1.0:
            raise ValueError(f"vram high_frac {high_frac!r} must be in (0, 1]")
        self.emit = emit
        self.high_frac = float(high_frac)
        self.realert_s = float(realert_s)
        self.clock = clock
        self.labels = labels or (lambda: {})
        self.high_since: float | None = None
        self._alerted_at: float | None = None
        self.alerts = 0
        self.peak_frac: float | None = None

    def observe(self, cards: Iterable[Mapping[str, Any]]) -> str | None:
        """One sample; returns the event written (or None)."""
        rows = card_fractions(cards)
        if not rows:
            return None
        top = max(r["frac"] for r in rows)
        self.peak_frac = top if self.peak_frac is None else max(self.peak_frac, top)
        high = [r for r in rows if r["frac"] > self.high_frac]
        now = self.clock()
        if high:
            if self.high_since is None:
                self.high_since = now
            if self._alerted_at is None or now - self._alerted_at >= self.realert_s:
                self._alerted_at = now
                self.alerts += 1
                self.emit(VRAM_HIGH_EVENT, max_frac=top, threshold=self.high_frac, alert=self.alerts,
                          high_for_s=round(now - self.high_since, 3),
                          cards=[{k: r[k] for k in ("index", "used_mb", "total_mb", "frac")} for r in high],
                          min_free_mib=min(r["free_bytes"] for r in rows) // MIB, **self.labels())
                return VRAM_HIGH_EVENT
            return None
        if self.high_since is not None:
            was = now - self.high_since
            self.high_since = self._alerted_at = None
            self.emit(VRAM_OK_EVENT, max_frac=top, threshold=self.high_frac, high_for_s=round(was, 3),
                      **self.labels())
            return VRAM_OK_EVENT
        return None


# ------------------------------------------------------------------ reload estimate
def _tensor_bytes(header: Mapping[str, Any], name: str) -> int:
    entry = header.get(name) or {}
    start, end = (entry.get("data_offsets") or (0, 0))[:2]
    return int(end) - int(start)


def read_safetensors_header(path: str | Path) -> dict[str, Any]:
    """The tensor entries of a safetensors file (no tensor data is read)."""
    with open(path, "rb") as handle:
        (n,) = struct.unpack("<Q", handle.read(8))
        header = json.loads(handle.read(n))
    header.pop("__metadata__", None)
    return header


def layer_bytes(model_dir: str | Path, plan: Mapping[int, Mapping[str, Any]],
                header: Callable[[Path], Mapping[str, Any]] = read_safetensors_header) -> dict[int, int]:
    """Checkpoint bytes of the released experts of every layer of ``plan``
    (``vllm_expert_reload.group_by_layer``: ``{layer: {"experts", "tensors": {name: file}}}``)."""
    root = Path(model_dir)
    headers: dict[str, Mapping[str, Any]] = {}
    out: dict[int, int] = {}
    for layer, entry in plan.items():
        total = 0
        for name, file in entry["tensors"].items():
            if file not in headers:
                headers[file] = header(root / file)
            total += _tensor_bytes(headers[file], name)
        out[int(layer)] = total
    return out


@dataclass(frozen=True)
class StagingEstimate:
    """The largest per-layer staging of one reload on one GPU (TP rank)."""

    layer: int | None
    layer_bytes: int  # checkpoint bytes of that layer's released experts (all ranks)
    experts: int  # released experts in that layer
    per_gpu_bytes: int  # staging + converted copies on one rank, without the reserve

    @property
    def per_expert_bytes(self) -> int:
        return math.ceil(self.per_gpu_bytes / self.experts) if self.experts else 0


def staging_bytes(sizes: Mapping[int, int], plan: Mapping[int, Mapping[str, Any]], *, tp: int,
                  copies: int = STAGING_COPIES) -> StagingEstimate:
    """The reload's peak extra GPU memory on one rank (module doc): the largest
    layer's released bytes over ``tp`` ranks (the experts' weights are narrowed by
    TP) times ``copies``."""
    if not sizes:
        return StagingEstimate(None, 0, 0, 0)
    layer = max(sizes, key=lambda k: sizes[k])
    per_gpu = math.ceil(sizes[layer] * int(copies) / max(1, int(tp)))
    return StagingEstimate(layer, int(sizes[layer]), len(plan[layer]["experts"]), per_gpu)


SAFE_IN_PLACE, SAFE_CHUNKED, SAFE_RESTART = "in_place", "chunked", "restart"


def reload_plan(free_bytes: int | None, estimate: StagingEstimate, *,
                reserve_bytes: int = DEFAULT_RESERVE_BYTES) -> dict[str, Any]:
    """How to run an in-place reload with ``free_bytes`` free on the fullest card:
    ``{"mode": in_place | chunked | restart, "max_experts": per chunk (chunked),
    "need_bytes", "free_bytes"}``. No reading (None) is taken for enough room: the
    card tool is not on every image, and the guard must not turn every reload into
    a restart there (``mode`` stays in_place, ``free_bytes`` None records it)."""
    need = int(estimate.per_gpu_bytes) + int(reserve_bytes)
    out: dict[str, Any] = {"need_bytes": need, "free_bytes": free_bytes, "reserve_bytes": int(reserve_bytes),
                           "staging_bytes": int(estimate.per_gpu_bytes), "layer": estimate.layer,
                           "experts_per_layer": estimate.experts}
    if free_bytes is None or estimate.per_gpu_bytes <= 0 or free_bytes >= need:
        return {**out, "mode": SAFE_IN_PLACE}
    room = int(free_bytes) - int(reserve_bytes)
    per_expert = estimate.per_expert_bytes
    if per_expert <= 0 or room < per_expert:
        return {**out, "mode": SAFE_RESTART}
    return {**out, "mode": SAFE_CHUNKED, "max_experts": max(1, min(estimate.experts - 1, room // per_expert))}
