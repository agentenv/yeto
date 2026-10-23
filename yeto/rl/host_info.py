"""Host facts an island prints at start (S27 G-AG diagnostics).

Two islands with the same image and the same requested CPU and memory ran
memory-heavy boundary work 2-4x apart (s27-gr-g2), and no log said on which
hardware.  :func:`host_report` collects the facts that decide that speed:
CPU model and count, the CPUs this process may use, cgroup CPU and memory
limits, host memory, NUMA nodes, and the thread settings of OpenMP, MKL,
OpenBLAS and torch.  It reads only ``/proc``, ``/sys`` and the environment
(never a credential) and never raises: a fact it cannot read is ``None``.

:func:`rank_host_report` is the same report as a trainer-rank plugin
(``TrainGroup.run_plugin``), because the full-parameter capture, apply and
materialize run in the rank processes, whose thread settings Ray may set
differently from the driver's.
"""

from __future__ import annotations

import json
import os
import platform
import socket
import sys
from pathlib import Path
from typing import Any

THREAD_ENV = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
              "NUMEXPR_NUM_THREADS", "MILES_FULL_PARAMETER_HASH_THREADS")
RANK_HOST_REPORT = f"{__name__}.rank_host_report"
RANK_HOST_REPORT_ENV = "YETO_RANK_HOST_REPORT"


def _read(path: str) -> str | None:
    try:
        return Path(path).read_text().strip()
    except (OSError, UnicodeDecodeError):
        return None


def _cpu_model() -> str | None:
    text = _read("/proc/cpuinfo") or ""
    for line in text.splitlines():
        key, _, value = line.partition(":")
        if key.strip() in ("model name", "Model", "cpu model"):
            return value.strip()
    return platform.processor() or None


def _meminfo_gib() -> dict[str, float | None]:
    values: dict[str, float | None] = {"mem_total_gib": None, "mem_available_gib": None}
    for line in (_read("/proc/meminfo") or "").splitlines():
        key, _, rest = line.partition(":")
        name = {"MemTotal": "mem_total_gib", "MemAvailable": "mem_available_gib"}.get(key)
        if name is not None:
            try:
                values[name] = round(int(rest.split()[0]) / (1 << 20), 1)  # kB -> GiB
            except (IndexError, ValueError):
                pass
    return values


def _cgroup_cpu_limit() -> float | None:
    """CPUs the cgroup quota allows (v2 ``cpu.max``, else v1 CFS); None = no limit."""
    text = _read("/sys/fs/cgroup/cpu.max")
    if text:
        quota, _, period = text.partition(" ")
        if quota != "max":
            try:
                return round(int(quota) / int(period or 100000), 2)
            except ValueError:
                return None
        return None
    quota, period = _read("/sys/fs/cgroup/cpu/cpu.cfs_quota_us"), _read("/sys/fs/cgroup/cpu/cpu.cfs_period_us")
    try:
        if quota is not None and period is not None and int(quota) > 0:
            return round(int(quota) / int(period), 2)
    except ValueError:
        pass
    return None


def _cgroup_memory_limit_gib() -> float | None:
    for path in ("/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory/memory.limit_in_bytes"):
        text = _read(path)
        if text is None:
            continue
        if text == "max":
            return None
        try:
            value = int(text)
        except ValueError:
            return None
        return None if value >= 1 << 60 else round(value / (1 << 30), 1)
    return None


def _numa_nodes() -> int | None:
    try:
        return len([p for p in Path("/sys/devices/system/node").glob("node[0-9]*")]) or None
    except OSError:
        return None


def _torch_threads() -> dict[str, int | None]:
    """torch's intra-op and inter-op threads, only when torch is already imported
    (this report never imports torch itself)."""
    torch = sys.modules.get("torch")
    if torch is None:
        return {"torch_threads": None, "torch_interop_threads": None}
    out: dict[str, int | None] = {}
    for key, getter in (("torch_threads", "get_num_threads"), ("torch_interop_threads", "get_num_interop_threads")):
        try:
            out[key] = int(getattr(torch, getter)())
        except Exception:  # noqa: BLE001 - a report never raises
            out[key] = None
    return out


def host_report() -> dict[str, Any]:
    """Facts of this host and process; every value is JSON-serializable."""
    report: dict[str, Any] = {}
    for key, read in (
        ("hostname", socket.gethostname),
        ("kernel", platform.release),
        ("cpu_model", _cpu_model),
        ("cpu_count", os.cpu_count),
        ("cpus_usable", lambda: len(os.sched_getaffinity(0))),
        ("cgroup_cpu_limit", _cgroup_cpu_limit),
        ("cgroup_memory_limit_gib", _cgroup_memory_limit_gib),
        ("numa_nodes", _numa_nodes),
        ("pid", os.getpid),
    ):
        try:
            report[key] = read()
        except Exception:  # noqa: BLE001 - a report never raises
            report[key] = None
    try:
        report.update(_meminfo_gib())
    except Exception:  # noqa: BLE001, S110 - a report never raises
        pass
    for name in THREAD_ENV:
        report[name] = os.environ.get(name)
    try:
        report.update(_torch_threads())
    except Exception:  # noqa: BLE001, S110 - a report never raises
        pass
    return report


def print_host_report(label: str) -> dict[str, Any]:
    """Print one ``[yeto-host] <label> {...}`` line on stderr; return the report."""
    report = host_report()
    print(f"[yeto-host] {label} {json.dumps(report, sort_keys=True)}", file=sys.stderr, flush=True)
    return report


def rank_host_report(actor: Any) -> dict[str, Any]:
    """Trainer-rank plugin (``run_plugin(RANK_HOST_REPORT, {})``): this rank's
    host report.  Never raises, so a report can never fail a trainer cell."""
    del actor
    try:
        return host_report()
    except Exception as error:  # noqa: BLE001
        return {"error": f"{type(error).__name__}: {error}"}


async def report_rank_hosts(group: Any, label: str) -> None:
    """Print each trainer rank's host report once (``YETO_RANK_HOST_REPORT=0``:
    skip).  Best effort: a group without ``run_plugin`` or a failed call only
    prints why."""
    if os.environ.get(RANK_HOST_REPORT_ENV, "1") == "0":
        return
    run_plugin = getattr(group, "run_plugin", None)
    if run_plugin is None:
        return
    try:
        results = await run_plugin(RANK_HOST_REPORT, {})
    except Exception as error:  # noqa: BLE001 - diagnostics must not stop the island
        print(f"[yeto-host] {label} rank report failed: {type(error).__name__}: {error}",
              file=sys.stderr, flush=True)
        return
    for rank, result in enumerate(results or ()):
        text = json.dumps(result, sort_keys=True) if isinstance(result, dict) else repr(result)
        print(f"[yeto-host] {label} rank {rank} {text}", file=sys.stderr, flush=True)
