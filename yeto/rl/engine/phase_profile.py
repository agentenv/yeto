"""E1 observation (``--rl-profile-trainer``): where the island driver's main thread
spends each driver phase.

When enabled, a daemon thread samples the stack of the thread that reports the
first phase (the driver's: it is not the thread that enables it, GPU run
e1n-yeto-miles-async-p) every ``SAMPLE_S`` and counts the samples per (phase, frame); at each
phase change the finished phase's wall time and top frames are printed as one
``yeto phase profile {json}`` line. Frames are labelled by the innermost frame in
yeto, miles or ray code (else the innermost frame). Nothing else changes.
"""

from __future__ import annotations

import collections
import json
import sys
import threading
import time
from typing import Any

SAMPLE_S = 0.005
TOP = 6
_MARKERS = ("/yeto/", "/miles/", "/ray/")
_STATE: dict[str, Any] = {"thread": None}


def _label(frame: Any, depth: int = 4) -> str:
    """The innermost ``depth`` yeto/miles/ray frames, innermost first (else the
    innermost frame): enough of the call chain to tell who waits on what."""
    inner, chain = None, []
    while frame is not None and len(chain) < depth:
        name = frame.f_code.co_filename
        here = f"{name.rsplit('/', 1)[-1]}:{frame.f_code.co_name}"
        inner = inner or here
        if any(m in name for m in _MARKERS):
            chain.append(here)
        frame = frame.f_back
    return " < ".join(chain) if chain else (inner or "?")


class _PhaseSampler(threading.Thread):
    def __init__(self) -> None:
        super().__init__(daemon=True, name="yeto-phase-profile")
        self.target: int | None = None
        self.lock = threading.Lock()
        self.phase: tuple[str, Any] | None = None
        self.started = time.perf_counter()
        self.counts: collections.Counter = collections.Counter()
        self.samples = 0

    def run(self) -> None:
        while True:
            time.sleep(SAMPLE_S)
            if self.target is None:
                continue
            frame = sys._current_frames().get(self.target)
            if frame is None:
                continue
            with self.lock:
                if self.phase is not None:
                    self.samples += 1
                    self.counts[_label(frame)] += 1

    def switch(self, phase: tuple[str, Any]) -> None:
        now = time.perf_counter()
        with self.lock:
            if self.target is None:
                self.target = threading.get_ident()  # the thread that drives the phases
            done, counts, samples = self.phase, self.counts, self.samples
            self.phase, self.counts, self.samples, started, self.started = (
                phase, collections.Counter(), 0, self.started, now)
        if done is not None:
            print("yeto phase profile " + json.dumps({
                "phase": done[0], "rollout_id": done[1], "wall_s": round(now - started, 4),
                "samples": samples, "top": counts.most_common(TOP)}), flush=True)


def enable() -> None:
    """Start the sampler; it samples the thread of the first :func:`on_phase`."""
    if _STATE["thread"] is None:
        sampler = _PhaseSampler()
        sampler.start()
        _STATE["thread"] = sampler


def on_phase(name: str, rollout_id: Any) -> None:
    sampler = _STATE["thread"]
    if sampler is not None:
        sampler.switch((name, rollout_id))
