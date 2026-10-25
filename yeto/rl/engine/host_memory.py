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
