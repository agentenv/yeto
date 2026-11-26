"""Node health while a run goes on: one shell call per check, read by a checker of the device family.

``yeto preflight`` checks a node before a run; this checks it every minute while the run
goes on, with the same parsers (:mod:`yeto.hw.host_probe`), so that a node whose GPU or
fabric broke is taken out before NCCL times out.

Checkers are registered by device family (:func:`register`, :func:`checker_for`). The
``cuda`` checker (:class:`CudaChecker`) reads in one shell call (:meth:`Checker.script`),
in this order:

* the ``NVRM: Xid`` lines of the kernel log (``dmesg``, else ``journalctl -k``; its last
  :data:`XID_TAIL_BYTES`) and the boot id, so that only the XIDs since the previous check
  count (first: a hung ``nvidia-smi`` does not hide them, and they are judged when
  ``nvidia-smi`` does not answer);
* ``nvidia-smi --query-gpu``: the GPUs that answer, their memory and their volatile
  uncorrectable ECC count; a GPU ``nvidia-smi`` names but cannot open (``Unable to
  determine the device handle``) is a GPU gone;
* the NVSwitch ``Fabric`` state of ``nvidia-smi -q`` and ``systemctl is-active
  nvidia-fabricmanager``;
* ``ibstat`` (``ibv_devinfo -v`` when ``ibstat`` fails): the state and rate of the IB /
  RoCE ports.

The script is bounded on the node: every section runs under ``timeout -k 5``
(:data:`SECTION_TIMEOUT_S`; a section past it is killed and reads as timed out), and the
whole check runs in the background and is killed (``kill -9`` of its process group) past
:data:`HEALTH_LIMIT_S` (:func:`bounded`); its output is printed either way. A check that
was killed is a read failure, but for the sections that completed: their hardware XIDs
still count. At worst a check takes :data:`HEALTH_LIMIT_S` on the node, less than its
period (:data:`HEALTH_EVERY_S`); a caller that adds the check to another command (a
status query) waits at most that much longer.

Every family without a checker gets :class:`NullChecker` (``enabled`` False, nothing
checked): its callers run nothing and write nothing.

:meth:`Checker.parse` is a pure function of the script's output, the HCAs the job uses
(``NCCL_IB_HCA`` items) and the facts of the previous check (``prev``: baselines and the
XID cursor). It judges two jobs at once (:class:`HealthResult`):

* ``ok_single``: a single-node job: every GPU answers (as many as expected, or as the
  first check saw), none lost memory, no uncorrectable ECC error and no hardware XID
  (:data:`HARDWARE_XIDS`) since the previous check, the NVLink fabric is
  ``Completed/Success`` on every GPU that has one, the fabric manager did not stop;
* ``ok_multi``: also every port the job's ``NCCL_IB_HCA`` names exists, is ``Active``
  and runs at no less than its rate at the first check (or ``min_ib_gbps``). False
  whenever ``ok_single`` is; None when the GPUs pass but the job names HCAs and the IB
  tools did not read (a read failure of the multi-node job: ``IB tools unread``).

``checked`` False (``ok_*`` None): ``nvidia-smi`` did not answer (a timeout, missing, an
ssh that failed, the check killed past its limit) and the kernel log shows no new
hardware XID: a read failure, which says nothing about the node. ``nvidia-smi`` that
cannot talk to the driver, a new hardware XID and a node with no GPU at all are bad
(``ok_single`` False), whatever else read.

:class:`HealthTracker` turns the checks into a verdict: ``ok``, ``read_failed`` (counts
neither way: a bad streak goes on after it), ``bad_once`` (one bad check), ``lost`` (two
bad checks in a row, or more). :class:`HealthWatch` keeps a checker, a tracker and the
previous facts for one node and says which event a reading asks for: ``node_health``
when the bad streak first reaches the limit (a read failure inside it does not start it
again), ``node_health_read_failed`` when a read starts failing, and
``node_health_degraded`` once per node when part of the check cannot judge (the kernel
log is not readable, the IB tools did not read, a multi-node job names no HCA) and once
per streak of :data:`READ_FAILED_DEGRADE` read failures in a row (never a loss: a node
that stops answering is the caller's to judge).

``python -m yeto.hw.node_health [--hca mlx5_0,...] [--multi] [--dry-run]`` runs one check
on this host (``--dry-run``: prints the script only).
"""

from __future__ import annotations

import argparse
import json
import re
import shlex
import subprocess
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from .host_probe import (
    IbPort,
    hca_filter,
    natural_order,
    parse_fabric,
    parse_sections,
    ports_of,
    probe_script,
    section_ok,
)

HEALTH_EVERY_S = 60.0  # the period of the check on a node runtime and a resident ssh island
BAD_CHECKS = 2  # bad checks in a row that make a node lost
SECTION_TIMEOUT_S = 10  # each section of the check (timeout -k 5 on the node: killed 5 s later)
HEALTH_LIMIT_S = 45  # the whole check on the node: killed past this (< HEALTH_EVERY_S)
READ_FAILED_DEGRADE = 5  # read failures in a row that write a node_health_degraded event
XID_TAIL_BYTES = 1 << 20  # the kernel log's XID lines read (the last MiB)
STALE_PERIODS = 3  # a check older than this many periods reads as a read failure
MAX_REASONS = 20  # reasons / notes kept in a record or an event
MAX_PORTS = 32  # ports kept in a record's facts
TIMEOUT_MARK = "@@yeto-health-timeout"
IB_UNREAD = "IB tools unread (ibstat / ibv_devinfo)"
# XIDs that mean the GPU (or its NVLink) needs a reset or the node a reboot (NVIDIA's XID
# catalog): double-bit ECC (48), internal micro-controller (61, 62), ECC page retirement
# failure (64), NVLink (74), fallen off the bus (79), uncontained ECC (95), GSP (119, 120),
# unrecovered ECC (140). XIDs of the application (13, 31, 43, 45, ...) are kept as facts only.
HARDWARE_XIDS = frozenset({48, 61, 62, 64, 74, 79, 95, 119, 120, 140})
VERDICTS = ("ok", "read_failed", "bad_once", "lost")
EVENT_LOST = "node_health"
EVENT_READ_FAILED = "node_health_read_failed"
EVENT_DEGRADED = "node_health_degraded"

GPU_QUERY = "index,pci.bus_id,name,memory.total,ecc.errors.uncorrected.volatile.total"
_T = "__yeto_t"  # the script's section wrapper: timeout -k 5 (timeout, no -k: without it) when the node has it
TOOL_WRAPPER = (
    "if command -v timeout >/dev/null 2>&1; then "
    f'if timeout -k 1 5 true >/dev/null 2>&1; then {_T}() {{ timeout -k 5 "$@"; }}; '
    f'else {_T}() {{ timeout "$@"; }}; fi; '
    f'else {_T}() {{ shift; "$@"; }}; fi\n'
)
XID_SCRIPT = (
    "cat /proc/sys/kernel/random/boot_id 2>/dev/null | sed 's/^/boot=/'; "
    f"if dmesg >/dev/null 2>&1; then echo source=dmesg; dmesg | grep -E 'NVRM: Xid' | tail -c {XID_TAIL_BYTES}; "
    "elif journalctl -k -b -n 1 --no-pager >/dev/null 2>&1; then echo source=journal; "
    f"journalctl -k -b -o short-monotonic --no-pager | grep -E 'NVRM: Xid' | tail -c {XID_TAIL_BYTES}; "
    "else echo source=none; fi; true"
)
_SECTIONS = (
    ("xid", XID_SCRIPT),
    ("gpus", f"nvidia-smi --query-gpu={GPU_QUERY} --format=csv,noheader,nounits"),
    ("fabric", "set -o pipefail; nvidia-smi -q | grep -A3 -E '^ *Fabric *$'"),
    ("fabricmanager", "systemctl is-active nvidia-fabricmanager"),
    ("ibstat", "ibstat"),
    ("ibv_devinfo", "ibstat >/dev/null 2>&1 && exit 3; ibv_devinfo -v"),  # only when ibstat fails
)
# every section bounded on the node (the preflight's own sections stay as they are)
CUDA_SECTIONS = tuple((name, f"{_T} {SECTION_TIMEOUT_S} bash -c {shlex.quote(cmd)}") for name, cmd in _SECTIONS)
DRIVER_DOWN = "couldn't communicate with the NVIDIA driver"


def bounded(script: str, limit_s: int = HEALTH_LIMIT_S) -> str:
    """``script`` run in the background on the node, killed (``kill -9`` of its process
    group: ``setsid`` when the node has it) past ``limit_s``; its output is printed either
    way, after :data:`TIMEOUT_MARK` when it was killed."""
    return (
        f"__yeto_hs={shlex.quote(script)}\n"
        "__yeto_hf=$(mktemp /tmp/yeto-health.XXXXXX 2>/dev/null) || __yeto_hf=/tmp/yeto-health.$$\n"
        'if command -v setsid >/dev/null 2>&1; then setsid bash -c "$__yeto_hs" > "$__yeto_hf" 2>&1 < /dev/null & '
        'else bash -c "$__yeto_hs" > "$__yeto_hf" 2>&1 < /dev/null & fi\n'
        "__yeto_hp=$!; __yeto_hn=0\n"
        f'while kill -0 "$__yeto_hp" 2>/dev/null && [ "$__yeto_hn" -lt {int(limit_s) * 10} ]; do sleep 0.1; '
        "__yeto_hn=$((__yeto_hn + 1)); done\n"
        'if kill -0 "$__yeto_hp" 2>/dev/null; then kill -9 -- "-$__yeto_hp" 2>/dev/null; '
        f'kill -9 "$__yeto_hp" 2>/dev/null; echo "{TIMEOUT_MARK} {int(limit_s)}"; fi\n'
        'cat "$__yeto_hf" 2>/dev/null; rm -f "$__yeto_hf"\n'
    )
# the facts a check hands to the next one (baselines and the XID cursor); kept over a read failure
CARRIED = ("gpu_count", "bus", "mem", "ecc", "xid", "fm", "ib_rate")


@dataclass
class HealthResult:
    """One check of one node (module doc). ``reasons``: every problem found, GPU ones first
    (``facts["gpu_reasons"]`` / ``facts["ib_reasons"]`` apart); ``facts``: what was read and
    the baselines the next check compares with (JSON: string keys)."""

    checked: bool
    ok_single: bool | None = None
    ok_multi: bool | None = None
    reasons: list[str] = field(default_factory=list)
    facts: dict[str, Any] = field(default_factory=dict)

    def ok(self, multi: bool) -> bool | None:
        return self.ok_multi if multi else self.ok_single

    def job_reasons(self, multi: bool) -> list[str]:
        """The reasons that count for the job (a single-node job: the GPU ones)."""
        if not self.checked:
            return list(self.reasons)
        return list(self.reasons) if multi else list(self.facts.get("gpu_reasons") or [])


class Checker:
    """A device family's health check: one shell script, a pure parser of its output."""

    family = ""
    enabled = True

    def script(self, hcas: Sequence[str]) -> str:
        raise NotImplementedError

    def parse(self, text: str, hcas: Sequence[str], prev: Mapping[str, Any] | None) -> HealthResult:
        raise NotImplementedError


class NullChecker(Checker):
    """A family without a checker: nothing is run, nothing judged (``checked`` False)."""

    enabled = False

    def __init__(self, family: str = "", **_: Any) -> None:
        self.family = family

    def script(self, hcas: Sequence[str]) -> str:
        return ""

    def parse(self, text: str, hcas: Sequence[str], prev: Mapping[str, Any] | None) -> HealthResult:
        return HealthResult(checked=False)


# -- the cuda checker ------------------------------------------------------------------------

_PCI = re.compile(r"(?:([0-9A-Fa-f]{4,8}):)?([0-9A-Fa-f]{2}):([0-9A-Fa-f]{2})(?:\.[0-7])?")
_NO_HANDLE = re.compile(r"Unable to determine the device handle for GPU\s*([0-9A-Fa-f:.]+)\s*:?\s*(.*)")
_XID = re.compile(r"(?:\[\s*([\d.]+)\])?.*NVRM: Xid \((?:PCI:)?([0-9A-Fa-f:.]+)\): (\d+)")


def pci_key(text: str) -> str | None:
    """``domain:bus:device`` of a PCI address as nvidia-smi (``00000000:18:00.0``), the
    XID line (``PCI:0000:18:00``) or an error message (``0000:5B:00.0``) writes it."""
    m = _PCI.search(str(text or ""))
    if m is None:
        return None
    return f"{int(m.group(1) or '0', 16):04x}:{m.group(2).lower()}:{m.group(3).lower()}"


def _num(value: str) -> int | None:
    try:
        return int(float(value.strip()))
    except (ValueError, AttributeError):
        return None


def _port_label(port: IbPort) -> str:
    return port.ca if port.port == 1 else f"{port.ca}:{port.port}"


def _read_error(sec: Any) -> str:
    if sec is None:
        return "nvidia-smi: no output (the check did not run)"
    if sec.rc in (124, 137):
        return f"nvidia-smi: timed out after {SECTION_TIMEOUT_S}s"
    if sec.rc == 127:
        return "nvidia-smi: not found"
    tail = " ".join(sec.text.split())[-200:]
    return f"nvidia-smi: exit {sec.rc}" + (f": {tail}" if tail else "")


class CudaChecker(Checker):
    """The ``cuda`` family (module doc). ``expect_gpus``: the GPUs the node has (the
    inventory's ``cards_per_node``; None: as many as the first check saw); ``min_ib_gbps``:
    the least rate of a job port (None: its rate at the first check)."""

    family = "cuda"

    def __init__(self, *, expect_gpus: int | None = None, min_ib_gbps: float | None = None, **_: Any) -> None:
        self.expect_gpus = None if expect_gpus is None else int(expect_gpus)
        self.min_ib_gbps = None if min_ib_gbps is None else float(min_ib_gbps)

    def script(self, hcas: Sequence[str]) -> str:  # noqa: ARG002 - every port is read; hcas judge them
        return bounded(TOOL_WRAPPER + probe_script(CUDA_SECTIONS))

    def parse(self, text: str, hcas: Sequence[str], prev: Mapping[str, Any] | None) -> HealthResult:
        prev = dict(prev or {})
        carried = {k: prev[k] for k in CARRIED if k in prev}
        sections = parse_sections(text)
        if TIMEOUT_MARK in str(text or ""):  # killed past its limit: only its completed XID section counts
            error = ("node health check cut off: its SSH call ran past its timeout"
                     if f"{TIMEOUT_MARK} ssh" in str(text) else
                     f"node health check timed out after {HEALTH_LIMIT_S}s (killed)")
            return self._unread(error, sections, prev, carried)
        gpus = sections.get("gpus")
        if gpus is not None and DRIVER_DOWN in gpus.text:  # nvidia-smi answers, the driver does not (rc 9)
            return self._unread("", sections, prev, carried, driver_down=True)
        rows, gone = self._gpu_rows(gpus)
        if gpus is None or (not rows and not gone and not self._no_devices(gpus)):
            return self._unread(_read_error(gpus), sections, prev, carried)
        facts: dict[str, Any] = dict(carried)
        gpu_reasons: list[str] = []
        bus = {**dict(prev.get("bus") or {}), **{key: idx for idx, key, _, _ in rows if key}}
        facts["bus"] = bus
        index_of = lambda key: bus.get(key or "", key)  # noqa: E731 - a GPU by its PCI address
        if not rows and not gone:  # a cuda node with no GPU at all ("No devices were found")
            gpu_reasons.append("no GPU visible (nvidia-smi: No devices were found)")
        # count: the inventory's, else the most this node showed
        baseline = self.expect_gpus if self.expect_gpus is not None else max(
            int(prev.get("gpu_count") or 0), len(rows))
        facts["gpu_count"] = baseline
        facts["gpus"] = len(rows)
        for key, detail in gone:
            gpu_reasons.append(f"GPU {index_of(key)} unreachable ({detail or 'no device handle'})")
        if rows and len(rows) < baseline and not gone:
            gpu_reasons.append(f"{len(rows)} of {baseline} GPUs visible")
        # memory: none may shrink from what it showed (by PCI address: an index can move)
        mem = dict(prev.get("mem") or {})
        for idx, key, mib, _ in rows:
            if mib is None:
                continue
            base = mem.get(key or idx)
            if base is not None and mib < 0.99 * float(base):
                gpu_reasons.append(f"GPU {idx} memory {mib} MiB < {base} MiB")
            else:
                mem[key or idx] = max(mib, int(base or 0))
        facts["mem"] = mem
        # uncorrectable ECC since the previous check (a count that went down: the new baseline)
        ecc = dict(prev.get("ecc") or {})
        for idx, key, _, count in rows:
            if count is None:
                continue
            before = ecc.get(key or idx)
            if before is not None and "ecc" in prev and count > int(before):
                gpu_reasons.append(f"GPU {idx} uncorrectable ECC +{count - int(before)}")
            ecc[key or idx] = count
        facts["ecc"] = ecc
        gpu_reasons += self._fabric(sections, prev, facts)
        gpu_reasons += self._xids(sections.get("xid"), prev, facts, index_of)
        ib_reasons, ib_read = self._ports(sections, hcas, prev, facts)
        facts["gpu_reasons"], facts["ib_reasons"] = gpu_reasons, ib_reasons
        ok_single = not gpu_reasons
        if not ok_single:
            ok_multi: bool | None = False
        elif not hcas:
            ok_multi = True
        elif not ib_read:  # the GPUs pass, the ports the job needs did not read
            ok_multi = None
            ib_reasons = facts["ib_reasons"] = [IB_UNREAD]
        else:
            ok_multi = not ib_reasons
        return HealthResult(checked=True, ok_single=ok_single, ok_multi=ok_multi,
                            reasons=gpu_reasons + ib_reasons, facts=facts)

    def _unread(self, error: str, sections: Mapping[str, Any], prev: Mapping[str, Any],
                carried: dict[str, Any], *, driver_down: bool = False) -> HealthResult:
        """``nvidia-smi`` did not read (``error``): a read failure, unless the driver is down
        or the completed XID section shows a new hardware XID (both bad: ``ok_single`` False)."""
        facts = dict(carried)
        xid = sections.get("xid")
        bus = dict(prev.get("bus") or {})
        reasons = []
        if xid is not None and xid.rc is not None:  # the section completed: its XIDs count
            reasons = self._xids(xid, prev, facts, lambda key: bus.get(key or "", key))
        if driver_down:
            reasons = [f"nvidia-smi: {DRIVER_DOWN}"] + reasons
        if not reasons:
            return HealthResult(checked=False, reasons=[error], facts={**facts, "read_error": error})
        facts["gpu_reasons"], facts["ib_reasons"] = reasons, []
        if error:
            facts["read_error"] = error
        return HealthResult(checked=True, ok_single=False, ok_multi=False, reasons=reasons, facts=facts)

    @staticmethod
    def _no_devices(sec: Any) -> bool:
        return sec is not None and "No devices were found" in sec.text

    @staticmethod
    def _gpu_rows(sec: Any) -> tuple[list[tuple[str, str | None, int | None, int | None]], list[tuple[str, str]]]:
        """``(index, pci key, memory MiB, uncorrectable ECC)`` per GPU that answered, and
        ``(pci key, message)`` per GPU nvidia-smi could not open."""
        rows, gone = [], []
        if sec is None or sec.rc in (124, 137, 127):
            return rows, gone
        for line in sec.text.splitlines():
            m = _NO_HANDLE.search(line)
            if m:
                gone.append((pci_key(m.group(1)) or m.group(1), m.group(2).strip()))
                continue
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 5 or not parts[0].isdigit():
                continue
            rows.append((parts[0], pci_key(parts[1]), _num(parts[3]), _num(parts[4])))
        return rows, gone

    @staticmethod
    def _fabric(sections: Mapping[str, Any], prev: Mapping[str, Any], facts: dict[str, Any]) -> list[str]:
        out = []
        fm_sec = sections.get("fabricmanager")
        fm = (fm_sec.text.strip().split() or [""])[0] if fm_sec is not None and fm_sec.rc != 127 else ""
        if fm in ("active", "inactive", "failed", "activating", "deactivating"):
            facts["fm"] = fm
            if prev.get("fm") == "active" and fm in ("inactive", "failed"):
                out.append(f"nvidia-fabricmanager {fm} (was active)")
        sec = sections.get("fabric")
        if sec is None or sec.rc in (124, 137):
            return out
        real = [(s, st) for s, st in parse_fabric(sec.text) if s.upper() != "N/A"]
        bad = [(s, st) for s, st in real if s.lower() != "completed" or (st and st.lower() != "success")]
        facts["fabric"] = f"{len(real) - len(bad)}/{len(real)} completed" if real else "none"
        if bad:
            state = "/".join(x for x in bad[0] if x)
            out.append(f"NVLink fabric {state} on {len(bad)}/{len(real)} GPUs (fabric manager: {fm or '?'})")
        return out

    @staticmethod
    def _xids(sec: Any, prev: Mapping[str, Any], facts: dict[str, Any],
              index_of: Callable[[str | None], Any]) -> list[str]:
        """Hardware XIDs since the previous check (the first check sets the cursor only)."""
        if sec is None:
            return []
        lines = sec.text.splitlines()
        head = {k: v for k, v in (ln.split("=", 1) for ln in lines if ln.startswith(("boot=", "source=")))}
        source = head.get("source", "none")
        facts["xid_source"] = source
        if source == "none":
            return []
        found, texts = [], []
        for line in lines:
            m = _XID.search(line)
            if m:
                found.append((float(m.group(1)) if m.group(1) else None, pci_key(m.group(2)), int(m.group(3))))
                texts.append(line.strip())
        cursor = dict(prev.get("xid") or {})
        boot = head.get("boot", "")
        if "xid" not in prev:  # the first check: what is in the log already is history
            new = []
        elif cursor.get("boot") != boot:
            new = found
        elif all(ts is not None for ts, _, _ in found):
            last = float(cursor.get("ts") or 0.0)
            new = [x for x in found if x[0] > last]
        else:  # no timestamps: after the cursor's line (its n-th time; the log's head may have gone)
            at = [i for i, t in enumerate(texts) if t == cursor.get("last")]
            n = int(cursor.get("last_n") or 0)
            start = at[n - 1] + 1 if n and len(at) >= n else (at[-1] + 1 if at else 0)
            new = found[start:]
        stamps = [ts for ts, _, _ in found if ts is not None]
        facts["xid"] = {"boot": boot, "ts": max(stamps) if stamps else cursor.get("ts"), "count": len(found),
                        "last": texts[-1] if texts else None,
                        "last_n": texts.count(texts[-1]) if texts else 0}
        out, seen = [], set()
        other = []
        for _, key, code in new:
            label = f"GPU {index_of(key)} XID {code}"
            if code in HARDWARE_XIDS:
                if label not in seen:
                    seen.add(label)
                    out.append(label)
            else:
                other.append(label)
        if other:
            facts["xid_other"] = other[-10:]
        return out

    def _ports(self, sections: Mapping[str, Any], hcas: Sequence[str], prev: Mapping[str, Any],
               facts: dict[str, Any]) -> tuple[list[str], bool]:
        read = section_ok(sections.get("ibstat")) or section_ok(sections.get("ibv_devinfo"))
        if not read:
            facts["ib"] = "unread"
            return [], False
        ports = natural_order(ports_of(sections))
        facts["ports"] = {_port_label(p): f"{p.state}" + (f" {p.rate_gbps:g}" if p.rate_gbps else "") for p in ports}
        rates = dict(prev.get("ib_rate") or {})
        out: list[str] = []
        if hcas:
            selected, missing = hca_filter(",".join(hcas), ports)
            out += [f"{name} missing" for name in missing]
        else:
            selected = []
        for p in natural_order(selected):
            label = _port_label(p)
            if not p.active:
                out.append(f"{label} port {(p.state or 'unknown').upper()}")
                continue
            base = self.min_ib_gbps if self.min_ib_gbps is not None else rates.get(label)
            if p.rate_gbps is not None and base is not None and p.rate_gbps < float(base) * 0.999:
                out.append(f"{label} rate {p.rate_gbps:g} < {float(base):g} Gb/s")
        for p in ports:  # every active port's best rate: the baseline of a later job port
            if p.active and p.rate_gbps is not None:
                rates[_port_label(p)] = max(p.rate_gbps, float(rates.get(_port_label(p)) or 0.0))
        facts["ib_rate"] = rates
        facts["ib"] = "read"
        return out, True


# -- the registry ----------------------------------------------------------------------------

CHECKERS: dict[str, Callable[..., Checker]] = {"cuda": CudaChecker}


def register(family: str, factory: Callable[..., Checker]) -> None:
    """Register the checker of a device family (``factory(**options) -> Checker``)."""
    CHECKERS[str(family)] = factory


def checker_for(family: str, **options: Any) -> Checker:
    """The checker of ``family`` (:class:`NullChecker` for a family without one).
    ``options``: ``expect_gpus``, ``min_ib_gbps`` (a checker ignores the ones it has no use for)."""
    factory = CHECKERS.get(str(family or ""))
    return NullChecker(str(family or "")) if factory is None else factory(**options)


def hcas_of(spec: str | None) -> list[str]:
    """The items of an ``NCCL_IB_HCA`` value (its ``=`` / ``^`` prefix stays on the first):
    joined with ``,`` they are the value again."""
    return [item.strip() for item in str(spec or "").split(",") if item.strip()]


# -- the verdict -----------------------------------------------------------------------------

class HealthTracker:
    """Checks in a row -> a verdict. ``multi``: the job spans nodes (its ports count).
    ``bad_checks`` bad checks in a row make the node ``lost`` (and keep it so while they
    go on); one good check clears the streak; a read failure (``checked`` False, or the
    job's ``ok`` None) counts neither way."""

    def __init__(self, *, multi: bool, bad_checks: int = BAD_CHECKS) -> None:
        self.multi = bool(multi)
        self.bad_checks = max(1, int(bad_checks))
        self.bad_in_row = 0

    def update(self, result: HealthResult) -> str:
        ok = result.ok(self.multi) if result.checked else None
        if ok is None:
            return "read_failed"
        if ok:
            self.bad_in_row = 0
            return "ok"
        self.bad_in_row += 1
        return "lost" if self.bad_in_row >= self.bad_checks else "bad_once"


@dataclass
class Reading:
    """One check through a :class:`HealthWatch`: the verdict, the result, the event it asks
    for (``node_health`` / ``node_health_read_failed``; None: no event) and, once per node,
    what the check cannot judge (``degraded``: the ``node_health_degraded`` event; None: nothing new)."""

    verdict: str
    result: HealthResult
    event: str | None = None
    degraded: list[str] | None = None

    @property
    def lost(self) -> bool:
        return self.verdict == "lost"


def degraded_of(result: HealthResult, *, multi: bool, hcas: Sequence[str]) -> list[str]:
    """What a read check could not judge for the job: XIDs (no kernel log access), the
    ports (the IB tools did not read; a multi-node job without ``NCCL_IB_HCA``)."""
    if not result.checked:
        return []
    out = []
    if result.facts.get("xid_source") == "none":
        out.append("XID unread (no dmesg / journalctl -k access)")
    if multi and hcas and result.facts.get("ib") == "unread":
        out.append(IB_UNREAD)
    if multi and not hcas:
        out.append("no NCCL_IB_HCA: the ports are not judged")
    return out


class HealthWatch:
    """One node's health over time: its checker, its tracker and the previous facts.
    ``hcas``: the job's ``NCCL_IB_HCA`` items; ``multi``: the job spans nodes; ``every_s``:
    the period of its checks (in its record: a reader takes an older one for a read failure)."""

    def __init__(self, family: str, *, hcas: Sequence[str] = (), multi: bool = False,
                 expect_gpus: int | None = None, min_ib_gbps: float | None = None,
                 bad_checks: int = BAD_CHECKS, checker: Checker | None = None,
                 clock: Callable[[], float] = time.time, every_s: float = HEALTH_EVERY_S) -> None:
        self.checker = checker or checker_for(family, expect_gpus=expect_gpus, min_ib_gbps=min_ib_gbps)
        self.tracker = HealthTracker(multi=multi, bad_checks=bad_checks)
        self.hcas = list(hcas)
        self.multi = bool(multi)
        self.clock = clock
        self.every_s = float(every_s)
        self.prev: dict[str, Any] | None = None
        self.last: Reading | None = None
        self.at: float | None = None
        self.degraded: list[str] | None = None  # set once (the node_health_degraded event)
        self.read_failed_in_row = 0

    @property
    def enabled(self) -> bool:
        return bool(self.checker.enabled)

    def script(self) -> str:
        return self.checker.script(self.hcas)

    def reset(self) -> None:
        """A new job on the same node (a relaunch): the bad streak and the last verdict go;
        the baselines and the XID cursor of the previous checks stay."""
        self.tracker.bad_in_row = 0
        self.last = None

    def observe(self, text: str) -> Reading:
        result = self.checker.parse(text, self.hcas, self.prev)
        verdict = self.tracker.update(result)
        before = self.last.verdict if self.last is not None else None
        event = None
        if verdict == "lost" and self.tracker.bad_in_row == self.tracker.bad_checks:  # the streak reached it
            event = EVENT_LOST
        elif verdict == "read_failed" and before != "read_failed":
            event = EVENT_READ_FAILED
        degraded: list[str] | None = None
        if self.degraded is None:
            found = degraded_of(result, multi=self.multi, hcas=self.hcas)
            if found:
                self.degraded = degraded = found
        self.read_failed_in_row = self.read_failed_in_row + 1 if verdict == "read_failed" else 0
        if self.read_failed_in_row == READ_FAILED_DEGRADE:  # once per streak: it says so, it loses nothing
            why = "; ".join(result.job_reasons(self.multi)) or "unread"
            degraded = (degraded or []) + [f"read failed {READ_FAILED_DEGRADE} checks in a row: {why}"]
        self.prev = dict(result.facts)
        self.last = Reading(verdict, result, event, degraded)
        self.at = float(self.clock())
        return self.last

    def record(self) -> dict[str, Any] | None:
        """The ``health`` field of a heartbeat / an island state (None before the first check)."""
        if self.last is None:
            return None
        r = self.last.result
        reasons = r.job_reasons(self.multi)
        doc: dict[str, Any] = {"verdict": self.last.verdict, "ok": r.ok(self.multi) if r.checked else None,
                               "reasons": reasons[:MAX_REASONS], "bad_in_row": self.tracker.bad_in_row,
                               "at": round(float(self.at or 0.0), 3), "every_s": self.every_s, "multi": self.multi}
        notes = [x for x in r.reasons if x not in reasons]
        if notes:  # problems that do not count for this job (a single-node job's ports)
            doc["notes"] = notes[:MAX_REASONS]
        if self.degraded:
            doc["degraded"] = list(self.degraded)
        if self.read_failed_in_row:
            doc["read_failed_in_row"] = self.read_failed_in_row
        facts = {k: r.facts[k] for k in ("gpus", "gpu_count", "fabric", "fm", "ports", "xid_source", "xid_other")
                 if k in r.facts}
        if "ports" in facts:
            facts["ports"] = dict(list(dict(facts["ports"]).items())[:MAX_PORTS])
        if facts:
            doc["facts"] = facts
        return doc


def event_fields(reading: Reading) -> dict[str, Any]:
    """The fields of the event a reading asks for."""
    r = reading.result
    return {"verdict": reading.verdict, "reasons": list(r.reasons)[:MAX_REASONS]}


def fresh(doc: Mapping[str, Any] | None, now: float) -> Mapping[str, Any] | None:
    """A health record as a reader takes it at ``now``: one older than
    :data:`STALE_PERIODS` periods (``at``, ``every_s``) reads as a read failure (the check
    stopped: a hung node runtime); None stays None."""
    if not isinstance(doc, Mapping):
        return None
    try:
        at, every = float(doc["at"]), float(doc.get("every_s") or HEALTH_EVERY_S)
    except (KeyError, TypeError, ValueError):
        return doc
    age = float(now) - at
    if age <= STALE_PERIODS * every:
        return doc
    return {**dict(doc), "verdict": "read_failed", "stale": True,
            "reasons": [f"health check stale: the last one {age:.0f}s ago"]}


# -- run it here -----------------------------------------------------------------------------

def run_local(script: str, *, timeout: float = HEALTH_LIMIT_S + 30.0,
              run: Callable[..., Any] = subprocess.run) -> str:
    """The script's output on this host ('' when it could not run: a read failure)."""
    try:
        done = run(["bash", "-c", script], capture_output=True, text=True, timeout=timeout, check=False)
    except (OSError, subprocess.TimeoutExpired):
        return ""
    return str(done.stdout or "")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m yeto.hw.node_health",
                                     description="one node health check on this host (yeto.hw.node_health)")
    parser.add_argument("--family", default="cuda", help="the device family (default cuda)")
    parser.add_argument("--hca", default="", help="the job's NCCL_IB_HCA (its ports are judged)")
    parser.add_argument("--multi", action="store_true", help="judge for a multi-node job (the ports count)")
    parser.add_argument("--expect-gpus", type=int, default=None, help="the GPUs the node has")
    parser.add_argument("--dry-run", action="store_true", help="print the check's script; run nothing")
    args = parser.parse_args(argv)
    watch = HealthWatch(args.family, hcas=hcas_of(args.hca), multi=args.multi, expect_gpus=args.expect_gpus)
    if not watch.enabled:
        print(f"device family {args.family}: no health checker (nothing is checked)")
        return 0
    if args.dry_run:
        print(watch.script(), end="")
        return 0
    reading = watch.observe(run_local(watch.script()))
    print(json.dumps(watch.record(), indent=1, sort_keys=True))
    return 0 if reading.verdict == "ok" else 1


if __name__ == "__main__":
    sys.exit(main())
