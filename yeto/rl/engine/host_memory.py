"""Give the memory of a released host copy back to the OS (S27 G-EY).

Run s27-elastic-c (Nemotron Nano, peer recovery): the donor's training
container went from 282 GiB to 736 GiB when every rank copied its state to
host memory for the snapshot, and stayed at about 692 GiB after the shards
were written and committed. On CPU (``cut_plugin`` host background write, the
same code) the copy is released, but two things keep it resident:

1. glibc keeps freed heap memory in the process. Tensors below the dynamic
   mmap threshold (at most 32 MiB; one Nemotron Nano expert weight is
   1856 x 2688: 9.5 MiB in bf16, 19 MiB per FP32 optimizer tensor) come from
   the heap, and ``free`` gives back only the top of a heap. ``malloc_trim(0)``
   returns every free page.
2. ``torch.save`` before 2.13 keeps the storages it saved alive in a reference
   cycle (its local pickler class) until a full garbage collection, which a
   large process runs rarely.

:func:`release_host_memory` handles both: a full collection, then
``malloc_trim(0)`` where glibc has it. A caller drops its last reference to
the copy first.
"""

from __future__ import annotations

import ctypes
import gc
import sys
import time
from typing import Any

_LIBC: list[Any] = []


def _malloc_trim() -> Any:
    """glibc's ``malloc_trim``, or None (another libc, or not Linux)."""
    if not _LIBC:
        trim = None
        if sys.platform.startswith("linux"):
            try:
                trim = ctypes.CDLL("libc.so.6").malloc_trim
                trim.argtypes = [ctypes.c_size_t]
                trim.restype = ctypes.c_int
            except (OSError, AttributeError):
                trim = None
        _LIBC.append(trim)
    return _LIBC[0]


def anonymous_rss_mib() -> int | None:
    """This process's anonymous resident memory in MiB (``RssAnon``), or None."""
    try:
        with open("/proc/self/status") as fh:
            for line in fh:
                if line.startswith("RssAnon:"):
                    return int(line.split()[1]) // 1024
    except OSError:
        pass
    return None


def release_host_memory() -> dict[str, Any]:
    """Collect reference cycles, then give free heap pages back to the OS.

    Returns what the caller logs: anonymous RSS before and after (MiB), the
    seconds of each part, and whether ``malloc_trim`` ran. ``ctypes`` drops
    the GIL during ``malloc_trim``; the collection holds it."""
    before = anonymous_rss_mib()
    started = time.monotonic()
    gc.collect()
    collected = time.monotonic()
    trim = _malloc_trim()
    if trim is not None:
        trim(0)
    return {
        "rss_anon_mib_before": before,
        "rss_anon_mib_after": anonymous_rss_mib(),
        "gc_s": round(collected - started, 3),
        "trim_s": round(time.monotonic() - collected, 3),
        "malloc_trim": trim is not None,
    }


# ---- where the host memory is (S27 G-EZ3) ----
#
# The 120B verl island on npu1 (G-EZ2) used about 150-175 GiB per node more
# than its parameters, gradients and Adam state; the node's MemAvailable alone
# cannot say which process or which kind of page holds it. These two readers
# give the split a caller logs next to each training step.

PROCESS_FIELDS = ("VmRSS", "VmHWM", "RssAnon", "RssFile", "RssShmem")
NODE_FIELDS = ("MemTotal", "MemAvailable", "MemFree", "AnonPages", "Shmem", "Cached", "Buffers",
               "Mapped", "SUnreclaim", "PageTables", "Unevictable", "Mlocked", "AnonHugePages")


def _kib_fields(path: str, fields: tuple[str, ...]) -> dict[str, int]:
    """``fields`` of a ``/proc`` file with ``Name:  N kB`` lines, in MiB (missing ones left out)."""
    out: dict[str, int] = {}
    try:
        with open(path) as fh:
            for line in fh:
                name, _, rest = line.partition(":")
                if name in fields:
                    parts = rest.split()
                    if parts and parts[0].isdigit():
                        out[name] = int(parts[0]) // 1024
    except OSError:
        pass
    return out


def process_memory(pid: int | str = "self") -> dict[str, int]:
    """A process's resident memory by kind, in MiB: total (``VmRSS``), peak
    (``VmHWM``), anonymous, file-backed and shared-memory pages."""
    return _kib_fields(f"/proc/{pid}/status", PROCESS_FIELDS)


def node_memory() -> dict[str, int]:
    """This node's ``/proc/meminfo`` fields of :data:`NODE_FIELDS`, in MiB."""
    return _kib_fields("/proc/meminfo", NODE_FIELDS)


HOST_MEMORY_ENV = "YETO_HOST_MEMORY"
HOST_MEMORY_MODES = ("report", "trim")


def host_memory_mode(environ: Any = None) -> str | None:
    """What a training island does about host memory around each step
    (``YETO_HOST_MEMORY``): None (unset, empty or ``0``: nothing), ``report``
    (log :func:`process_memory` / :func:`node_memory` before and after each
    step), or ``trim`` (also :func:`release_host_memory` after each step and
    after each export). Any other value is an error, not a silent no-op."""
    import os

    value = (os.environ if environ is None else environ).get(HOST_MEMORY_ENV, "").strip().lower()
    if value in ("", "0"):
        return None
    if value not in HOST_MEMORY_MODES:
        raise ValueError(f"{HOST_MEMORY_ENV}={value!r}: expected one of {HOST_MEMORY_MODES} (or unset)")
    return value


def summarize_by_node(reports: list[dict[str, Any]]) -> dict[str, Any]:
    """One line's worth of per-worker reports (each ``{"host", "pid", "rank",
    "process", "node"[, "release"]}``): per host the node's MemAvailable and
    Shmem (the lowest any worker saw) and the summed worker RSS by kind; per
    worker its RSS. Trim results are summed per host when present."""
    nodes: dict[str, dict[str, Any]] = {}
    for report in reports:
        host = str(report.get("host"))
        node = nodes.setdefault(host, {"workers": 0, "mem_available_mib": None, "shmem_mib": None,
                                       **{f"workers_{k.lower()}_mib": 0 for k in PROCESS_FIELDS[:1] + PROCESS_FIELDS[2:]}})
        node["workers"] += 1
        meminfo = report.get("node") or {}
        for key, field in (("mem_available_mib", "MemAvailable"), ("shmem_mib", "Shmem")):
            if field in meminfo:
                node[key] = meminfo[field] if node[key] is None else min(node[key], meminfo[field])
        if "MemTotal" in meminfo:
            node["mem_total_mib"] = meminfo["MemTotal"]
        process = report.get("process") or {}
        for field in PROCESS_FIELDS[:1] + PROCESS_FIELDS[2:]:
            node[f"workers_{field.lower()}_mib"] += int(process.get(field, 0))
        release = report.get("release")
        if release and release.get("rss_anon_mib_before") is not None and release.get("rss_anon_mib_after") is not None:
            node["trimmed_mib"] = node.get("trimmed_mib", 0) + (
                int(release["rss_anon_mib_before"]) - int(release["rss_anon_mib_after"]))
            node["trim_s_max"] = max(node.get("trim_s_max", 0.0), float(release.get("gc_s", 0)) + float(release.get("trim_s", 0)))
    workers = [{"host": r.get("host"), "rank": r.get("rank"), "pid": r.get("pid"), **(r.get("process") or {})}
               for r in sorted(reports, key=lambda r: (str(r.get("host")), r.get("rank") if r.get("rank") is not None else -1))]
    return {"nodes": nodes, "workers": workers}
