"""E1 observation: where the trainer process spends one training step.

Opt-in (``--rl-profile-trainer``): around each ``train_one_step`` call it records the
wall time, the main thread's CPU time, the whole process's CPU time, garbage
collector pauses, CUDA caching-allocator counters, and stack samples of every
Python thread (every ``SAMPLE_S``), and prints one ``yeto train profile {json}``
line. It changes no value the step computes.
"""

from __future__ import annotations

import collections
import gc
import json
import os
import sys
import threading
import time
from typing import Any

SAMPLE_S = 0.005
TOP = 12
_GC = {"pause_s": 0.0, "collections": collections.Counter(), "start": None}
_GC_INSTALLED = False


def _gc_callback(phase: str, info: dict) -> None:
    if phase == "start":
        _GC["start"] = time.perf_counter()
    elif _GC["start"] is not None:
        _GC["pause_s"] += time.perf_counter() - _GC["start"]
        _GC["collections"][int(info.get("generation", -1))] += 1
        _GC["start"] = None


def _frame_label(frame: Any) -> str:
    """The innermost frame outside the standard library and threading machinery."""
    first = None
    while frame is not None:
        code = frame.f_code
        name = code.co_filename
        label = f"{os.path.basename(name)}:{code.co_name}"
        first = first or label
        if "/lib/python3" not in name or "site-packages" in name:
            return label
        frame = frame.f_back
    return first or "?"


class _Sampler(threading.Thread):
    def __init__(self, main_id: int) -> None:
        super().__init__(daemon=True, name="yeto-train-profile")
        self.main_id = main_id
        self.stop = threading.Event()
        self.main = collections.Counter()
        self.others = collections.Counter()
        self.samples = 0

    def run(self) -> None:
        own = threading.get_ident()
        names = {t.ident: t.name for t in threading.enumerate()}
        while not self.stop.wait(SAMPLE_S):
            self.samples += 1
            for ident, frame in sys._current_frames().items():
                if ident == own:
                    continue
                label = _frame_label(frame)
                if ident == self.main_id:
                    self.main[label] += 1
                elif not label.startswith(("threading.py:wait", "queue.py:get", "selectors.py")):
                    self.others[f"{names.get(ident, ident)}|{label}"] += 1


def _cuda_counters() -> dict[str, int]:
    try:
        import torch

        if not torch.cuda.is_available():
            return {}
        stats = torch.cuda.memory_stats()
    except Exception:  # noqa: BLE001 - observation only
        return {}
    keys = ("num_alloc_retries", "num_device_alloc", "num_device_free", "num_sync_all_streams")
    out = {k: int(stats.get(k, 0)) for k in keys}
    out["reserved_bytes"] = int(stats.get("reserved_bytes.all.current", 0))
    out["allocated_bytes"] = int(stats.get("allocated_bytes.all.current", 0))
    return out


def profiled(call: Any, *args: Any, **kwargs: Any) -> Any:
    """``call(*args, **kwargs)`` with the profile line printed after it."""
    global _GC_INSTALLED
    if not _GC_INSTALLED:
        gc.callbacks.append(_gc_callback)
        _GC_INSTALLED = True
    sampler = _Sampler(threading.get_ident())
    gc_before, gens_before = _GC["pause_s"], collections.Counter(_GC["collections"])
    cuda_before = _cuda_counters()
    wall, thread_cpu, process_cpu = time.perf_counter(), time.thread_time(), time.process_time()
    sampler.start()
    try:
        return call(*args, **kwargs)
    finally:
        sampler.stop.set()
        sampler.join()
        cuda_after = _cuda_counters()
        record = {
            "wall_s": round(time.perf_counter() - wall, 4),
            "main_thread_cpu_s": round(time.thread_time() - thread_cpu, 4),
            "process_cpu_s": round(time.process_time() - process_cpu, 4),
            "gc_pause_s": round(_GC["pause_s"] - gc_before, 4),
            "gc_collections": dict(_GC["collections"] - gens_before),
            "gc_objects": len(gc.get_objects()) if os.environ.get("YETO_RL_PROFILE_GC_OBJECTS") else None,
            "cuda": {k: cuda_after[k] - cuda_before.get(k, 0) if k.startswith("num_") else cuda_after[k]
                     for k in cuda_after},
            "threads": threading.active_count(),
            "samples": sampler.samples,
            "main_top": sampler.main.most_common(TOP),
            "others_top": sampler.others.most_common(TOP),
        }
        print("yeto train profile " + json.dumps(record), flush=True)
