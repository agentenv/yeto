"""fleet-dashboard 2.1/2.2: learner-island periodic telemetry threads.

``HeartbeatThread`` writes ``rl_heartbeat`` (phase, rollout_id, policy_version,
pid, uptime_s) every ``interval_s``; ``ResourceSampler`` writes
``rl_resource_sample`` (per-GPU NVML index/uuid/util/memory/power plus the
process RSS). Both run on daemon threads, only call ``emit`` (the tape writer
is lock-protected), never block the training loop, and stop cleanly on
``stop()``. When NVML cannot be loaded the sampler writes exactly one
``available: false`` record and stops. Both are opt-in (the driver starts
them only when an interval is configured), so the default tape is unchanged.

The sampler also keeps the peak GPU memory / process RSS since the last
``take_peaks()``: the driver merges these into ``rl_load_sample`` as
``peak_gpu_mem_bytes`` / ``peak_cpu_rss_bytes`` (timeline.LOAD_SAMPLE_SCHEMA),
so the two events share one probe instead of defining a second peak source.
"""

from __future__ import annotations

import os
import threading
import time
from typing import Any, Callable, Mapping

HEARTBEAT_EVENT = "rl_heartbeat"
RESOURCE_EVENT = "rl_resource_sample"
DEFAULT_HEARTBEAT_INTERVAL_S = 30.0
DEFAULT_RESOURCE_INTERVAL_S = 60.0
# Per-GPU keys of an available ``rl_resource_sample`` (None = not readable).
GPU_SAMPLE_KEYS = ("index", "uuid", "util_pct", "mem_used_mb", "mem_total_mb", "power_w")

Emit = Callable[..., None]


class _Periodic:
    name = "yeto-periodic"

    def __init__(self, interval_s: float, *, wait: Callable[[threading.Event, float], bool] | None = None):
        if not interval_s or interval_s <= 0:
            raise ValueError(f"{self.name}: interval must be > 0, got {interval_s!r}")
        self.interval_s = float(interval_s)
        self._stop = threading.Event()
        self._wait = wait or (lambda event, timeout: event.wait(timeout))
        self._thread: threading.Thread | None = None

    def tick(self) -> bool:  # pragma: no cover - overridden
        """One period's work; return False to stop the thread."""
        raise NotImplementedError

    def _loop(self) -> None:
        while not self._wait(self._stop, self.interval_s):
            try:
                if self.tick() is False:
                    return
            except Exception:  # noqa: BLE001 - telemetry must never kill the island
                continue

    def start(self) -> "_Periodic":
        if self._thread is None:
            self._thread = threading.Thread(target=self._loop, name=self.name, daemon=True)
            self._thread.start()
        return self

    def stop(self, timeout: float | None = 5.0) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout)

    @property
    def alive(self) -> bool:
        return self._thread is not None and self._thread.is_alive()


class HeartbeatThread(_Periodic):
    """2.1: ``rl_heartbeat`` every ``interval_s`` with the island's live state."""

    name = "yeto-heartbeat"

    def __init__(self, emit: Emit, state: Callable[[], Mapping[str, Any]], *,
                 interval_s: float = DEFAULT_HEARTBEAT_INTERVAL_S,
                 clock: Callable[[], float] = time.monotonic, **kw: Any):
        super().__init__(interval_s, **kw)
        self._emit = emit
        self._state = state
        self._clock = clock
        self._started = clock()
        self.beats = 0

    def tick(self) -> bool:
        self.beats += 1
        state = dict(self._state() or {})
        self._emit(
            HEARTBEAT_EVENT,
            phase=state.get("phase"),
            rollout_id=state.get("rollout_id"),
            policy_version=state.get("policy_version"),
            pid=os.getpid(),
            uptime_s=round(self._clock() - self._started, 3),
            seq=self.beats,
            interval_s=self.interval_s,
            **{k: v for k, v in state.items()
               if k not in ("phase", "rollout_id", "policy_version")},
        )
        return True


def load_nvml() -> Any:
    """The ``pynvml`` module, initialised; raises when NVML is unavailable."""
    import pynvml  # type: ignore[import-not-found]

    pynvml.nvmlInit()
    return pynvml


def _cpu_rss_bytes() -> int | None:
    try:
        with open("/proc/self/statm", encoding="ascii") as handle:
            pages = int(handle.read().split()[1])
        return pages * os.sysconf("SC_PAGE_SIZE")
    except (OSError, ValueError, IndexError):
        return None


def _safe(fn: Callable[[], Any]) -> Any:
    try:
        return fn()
    except Exception:  # noqa: BLE001 - one unreadable field is None, not a failed sample
        return None


class ResourceSampler(_Periodic):
    """2.2: ``rl_resource_sample`` every ``interval_s`` (NVML), single fallback record."""

    name = "yeto-resource-sampler"

    def __init__(self, emit: Emit, *, interval_s: float = DEFAULT_RESOURCE_INTERVAL_S,
                 nvml_loader: Callable[[], Any] = load_nvml,
                 rss: Callable[[], int | None] = _cpu_rss_bytes,
                 clock: Callable[[], float] = time.monotonic,
                 labels: Callable[[], Mapping[str, Any]] | None = None, **kw: Any):
        super().__init__(interval_s, **kw)
        self._emit = emit
        self._loader = nvml_loader
        self._rss = rss
        self._clock = clock
        self._labels = labels or (lambda: {})
        self._nvml: Any = None
        self.available: bool | None = None
        self._peak_lock = threading.Lock()
        self._peak_gpu: int | None = None
        self._peak_rss: int | None = None

    def _ensure_nvml(self) -> bool:
        if self.available is None:
            try:
                self._nvml = self._loader()
                self.available = True
            except Exception as error:  # noqa: BLE001 - ImportError, NVMLError, ...
                self.available = False
                self._emit(RESOURCE_EVENT, available=False, t=self._clock(),
                           reason=f"{type(error).__name__}: {error}"[:300], **self._labels())
        return bool(self.available)

    def start(self) -> "ResourceSampler":
        # Probe NVML once up front: an unavailable source is recorded at start
        # (single ``available: false`` record) and no thread is started.
        if self._ensure_nvml():
            super().start()
        return self

    def sample_gpus(self) -> list[dict[str, Any]]:
        nv = self._nvml
        gpus = []
        self._used_bytes: list[int] = []
        for index in range(int(nv.nvmlDeviceGetCount())):
            handle = nv.nvmlDeviceGetHandleByIndex(index)
            mem = _safe(lambda: nv.nvmlDeviceGetMemoryInfo(handle))
            util = _safe(lambda: nv.nvmlDeviceGetUtilizationRates(handle))
            power = _safe(lambda: nv.nvmlDeviceGetPowerUsage(handle))
            uuid = _safe(lambda: nv.nvmlDeviceGetUUID(handle))
            if mem is not None:
                self._used_bytes.append(int(mem.used))
            gpus.append({
                "index": index,
                "uuid": uuid.decode() if isinstance(uuid, bytes) else uuid,
                "util_pct": None if util is None else int(util.gpu),
                "mem_used_mb": None if mem is None else int(mem.used) / 2**20,
                "mem_total_mb": None if mem is None else int(mem.total) / 2**20,
                "power_w": None if power is None else float(power) / 1000.0,
            })
        return gpus

    def tick(self) -> bool:
        if not self._ensure_nvml():
            return False  # one ``available: false`` record, then no more sampling
        gpus = self.sample_gpus()
        rss = self._rss()
        used = self._used_bytes
        with self._peak_lock:
            if used:
                self._peak_gpu = max(self._peak_gpu or 0, max(used))
            if rss is not None:
                self._peak_rss = max(self._peak_rss or 0, int(rss))
        self._emit(RESOURCE_EVENT, available=True, t=self._clock(), pid=os.getpid(),
                   gpus=gpus, cpu_rss_bytes=rss, **self._labels())
        return True

    def take_peaks(self) -> dict[str, int]:
        """Peaks since the last call (LOAD_SAMPLE_SCHEMA keys; empty if none)."""
        with self._peak_lock:
            peaks = {"peak_gpu_mem_bytes": self._peak_gpu, "peak_cpu_rss_bytes": self._peak_rss}
            self._peak_gpu = self._peak_rss = None
        return {k: v for k, v in peaks.items() if v is not None}
