"""Node health while a run goes on: one shell call per check, read by a checker of the device family.

``yeto preflight`` checks a node before a run; this checks it every minute while the run
goes on, with the same parsers (:mod:`yeto.hw.host_probe`), so that a node whose GPU or
fabric broke is taken out before NCCL times out.

Checkers are registered by device family (:func:`register`, :func:`checker_for`). The
``cuda`` checker (:class:`CudaChecker`) reads in one shell call (:meth:`Checker.script`),
in this order:

* the ``NVRM: Xid`` lines of the kernel log (``dmesg``, else ``journalctl -k`` when it
  holds a kernel line of this boot; its last :data:`XID_TAIL_BYTES`) and the boot id, so
  that only the XIDs since the previous check count (first: a hung ``nvidia-smi`` does
  not hide them, and they are judged when ``nvidia-smi`` does not answer);
* ``nvidia-smi --query-gpu``: the GPUs that answer, their memory and their volatile
  uncorrectable ECC count; a GPU ``nvidia-smi`` names but cannot open (``Unable to
  determine the device handle``) is a GPU gone;
* the NVSwitch ``Fabric`` state of ``nvidia-smi -q``, ``systemctl is-active
  nvidia-fabricmanager`` with its unit's ``NRestarts``, and what shows the node's own
  NVSwitch (:data:`FM_SCRIPT`, :func:`nvswitch_of`);
* ``ibstat`` (``ibv_devinfo -v`` when ``ibstat`` fails): the state and rate of the IB /
  RoCE ports.

The script is bounded on the node: every section runs under ``timeout -k 5``
(:data:`SECTION_TIMEOUT_S`; a section past it is killed and reads as timed out), and the
whole check runs in the background and is killed (``kill -9`` of its process group) past
:data:`HEALTH_LIMIT_S` (:func:`bounded`); its output is printed either way. A check that
was killed is a read failure, but for the sections that completed: their hardware XIDs
still count. A check whose kill did not take (a tool stuck in the kernel) makes the next
ones skip (a read failure) while that same process lives (pid and start time), so hung
checks do not pile up. At worst a check takes :data:`HEALTH_LIMIT_S` on the node, less than its
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
  ``Completed/Success`` on every GPU that has one, the fabric manager is active (and did
  not restart since the previous check) in a boot in which it was active, or on a node
  with NVSwitch at the first read (:meth:`CudaChecker._fabric_manager`,
  :func:`fabric_manager_absolute`);
* ``ok_multi``: also every port the job's ``NCCL_IB_HCA`` names exists, is ``Active``
  and runs at no less than its rate at the first check (or ``min_ib_gbps``). False
  whenever ``ok_single`` is; None when the GPUs pass but the job names HCAs and the IB
  tools did not read (a read failure of the multi-node job: ``IB tools unread``).

``checked`` False (``ok_*`` None): ``nvidia-smi`` did not answer (a timeout, missing, an
ssh that failed, the check killed past its limit), or the output is not whole (empty,
a section missing or cut off before its exit code), and the kernel log shows no new
hardware XID: a read failure, which says nothing about the node. An output is never
judged healthy unless every section of the check is in it, complete. ``nvidia-smi`` that
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

import json
import re
import shlex
import subprocess
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from yeto.exact_argparse import ExactArgumentParser
from .host_probe import (
    IbPort,
    hca_filter,
    link_layer_ports,
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
BUSY_MARK = "@@yeto-health-busy"  # the previous check on the node still runs (killed, not gone)
# a process's start time (field 22 of /proc/<pid>/stat; else ps lstart): with its pid, it names
# the process, so a pid reused by another process is not taken for the stuck check
START_TIME_FN = (
    '__yeto_st() { __yeto_sf="${__YETO_PROC:-/proc}/$1/stat"; if [ -r "$__yeto_sf" ]; then '
    "sed -E 's/^.*\\) //' \"$__yeto_sf\" 2>/dev/null | awk '{print $20}'; "
    "else ps -o lstart= -p \"$1\" 2>/dev/null | tr -s ' ' '_'; fi; }\n"
)
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
    # the journal counts only when it holds a kernel line of this boot: in a container
    # without the host's journal, journalctl -k exits 0 with no entry and reads no XID
    "elif [ -n \"$(journalctl -k -b -n 1 -q --no-pager 2>/dev/null)\" ]; then echo source=journal; "
    f"journalctl -k -b -o short-monotonic --no-pager | grep -E 'NVRM: Xid' | tail -c {XID_TAIL_BYTES}; "
    "else echo source=none; fi; true"
)
VPD_READ_S = 1  # one VPD read (a VPD that does not answer must not hold the section)
VPD_BUDGET_S = 3  # every VPD read of a check: past it, the rest are not read
# The fabricmanager section. First, so that a section cut off later still has them: the
# fabric manager's state (the section's exit code is systemctl's) and its unit's LoadState
# (``systemctl show -p``, its prefix cut: no ``--value``, which systemd < 230 lacks); whether
# the processes of the node can be seen (``proc1``: /proc/1/comm readable; not under
# hidepid) and, when the state is not active, the ``nv-fabricmanager`` processes (comm is cut
# at 15 characters) with their pid and start time (``fm_ids``): a fabric manager run by a GPU
# Operator driver container, or a stale host unit next to it. Then its unit's restarts, and
# what shows NVSwitch on the node (:func:`nvswitch_of`): the driver's NVSwitch devices
# (procfs, device files), NVIDIA PCI bridges (class 0680; only when lspci runs), and every
# Mellanox (15b3) PCI function with how many of their VPD read (1 s each, 3 s in all; none
# without timeout) and name the NVLink switch management port (SMDL=SW_MNG: NVSwitch behind
# a ConnectX bridge, HGX B200 / B300; root only). The __YETO_* paths are for tests.
FM_SCRIPT = START_TIME_FN + "\n".join((
    '__s=$(systemctl is-active nvidia-fabricmanager 2>/dev/null); __rc=$?; [ -n "$__s" ] && echo "$__s"',
    '__ls=$(systemctl show -p LoadState nvidia-fabricmanager 2>/dev/null); echo "loadstate=${__ls#LoadState=}"',
    '__pr="${__YETO_PROC:-/proc}"; __pv=0; [ -r "$__pr/1/comm" ] && __pv=1; echo "proc1=$__pv"',
    'if [ "$__s" != active ] && [ "$__pv" = 1 ]; then',
    '  __fp=$(grep -lsx nv-fabricmanage "$__pr"/[0-9]*/comm 2>/dev/null); __ids=',
    '  for __c in $__fp; do __q=${__c%/comm}; __q=${__q##*/}; __ids="$__ids $__q:$(__yeto_st "$__q")"; done',
    '  echo "fm_proc=$(printf \'%s\\n\' "$__fp" | grep -c .)"; echo "fm_ids=${__ids# }"',
    'fi',
    '__nr=$(systemctl show -p NRestarts nvidia-fabricmanager 2>/dev/null); echo "nrestarts=${__nr#NRestarts=}"',
    '__p="${__YETO_NVSW_PROC:-/proc/driver/nvidia-nvswitch/devices}"',
    "[ -d \"$__p\" ] && echo \"nvswitch_proc=$(ls -A \"$__p\" 2>/dev/null | wc -l | tr -d ' ')\"",
    "echo \"nvswitch_dev=$(ls -d \"${__YETO_DEV:-/dev}\"/nvidia-nvswitch[0-9]* 2>/dev/null | wc -l | tr -d ' ')\"",
    "if command -v lspci >/dev/null 2>&1; then __l=$(lspci -n -d 10de: 2>/dev/null) "
    "&& echo \"nvswitch_pci=$(printf '%s\\n' \"$__l\" | grep -c ' 0680: ')\"; fi",
    '__m=0; __r=0; __w=0; __t=; command -v timeout >/dev/null 2>&1 && __t=1; __v0=$SECONDS',
    'for __f in $(grep -lx 0x15b3 "${__YETO_SYS_PCI:-/sys/bus/pci/devices}"/*/vendor 2>/dev/null); do',
    f'  __m=$((__m + 1)); __d=${{__f%/vendor}}; [ $((SECONDS - __v0)) -lt {VPD_BUDGET_S} ] || continue',
    f"  if __v=$(set -o pipefail; timeout {VPD_READ_S} head -c 2048 \"$__d/vpd\" 2>/dev/null | tr -d '\\000'); then "
    '__r=$((__r + 1)); case "$__v" in *SW_MNG*) __w=$((__w + 1));; esac; fi',
    'done',
    'echo "mlx=$__m"; [ -n "$__t" ] && { echo "mlx_vpd=$__r"; echo "sw_mng=$__w"; }',
    'exit $__rc',
))
_SECTIONS = (
    ("xid", XID_SCRIPT),
    ("gpus", f"nvidia-smi --query-gpu={GPU_QUERY} --format=csv,noheader,nounits"),
    ("fabric", "set -o pipefail; nvidia-smi -q | grep -A3 -E '^ *Fabric *$'"),
    ("fabricmanager", FM_SCRIPT),
    ("ibstat", "ibstat"),
    ("ibv_devinfo", "ibstat >/dev/null 2>&1 && exit 3; ibv_devinfo -v"),  # only when ibstat fails
)
# every section bounded on the node (the preflight's own sections stay as they are)
CUDA_SECTIONS = tuple((name, f"{_T} {SECTION_TIMEOUT_S} bash -c {shlex.quote(cmd)}") for name, cmd in _SECTIONS)
DRIVER_DOWN = "couldn't communicate with the NVIDIA driver"


# the first commands of the background check: its pid file (pid and start time, written whole
# then renamed), then the check itself in the same process (exec keeps the pid and start time)
PID_PRELUDE = (
    START_TIME_FN
    + '{ echo "$$ $(__yeto_st $$)" > "$__yeto_pf.$$" && mv -f "$__yeto_pf.$$" "$__yeto_pf"; } 2>/dev/null '
    '|| rm -f "$__yeto_pf.$$" 2>/dev/null\n'
    'exec bash -c "$__yeto_hs"\n'
)


def bounded(script: str, limit_s: int = HEALTH_LIMIT_S, *, pid_file: str | None = None) -> str:
    """``script`` run in the background on the node, killed (``kill -9`` of its process
    group: ``setsid`` when the node has it) past ``limit_s``; its output is printed either
    way, after :data:`TIMEOUT_MARK` when it was killed.

    A check whose kill did not take (a tool stuck in the kernel, ``D`` state) leaves its
    pid and the process's start time in ``pid_file`` (default:
    ``/tmp/yeto-health-<uid>.pid``); while that same process lives, a later check runs
    nothing and prints :data:`BUSY_MARK` (a read failure), so hung checks do not pile up on
    a sick node. A pid that is gone, or now names another process (another start time: the
    pid was reused, the node rebooted), is stale: the file goes and the check runs. A
    check of another caller that runs at the same time (younger than ``limit_s``) does not
    count; nor does a pid whose start time cannot be read.

    The background process writes its own pid file (:data:`PID_PRELUDE`) before it runs
    the check (``exec``: the same pid and start time), whole and then renamed: the file
    names the check from its first command on, and a check reading it at the same time
    never sees half of it."""
    pf = shlex.quote(pid_file) if pid_file else '"/tmp/yeto-health-$(id -u 2>/dev/null || echo 0).pid"'
    return (
        f"__yeto_hs={shlex.quote(script)}\n"
        f"__yeto_pf={pf}\n"
        + START_TIME_FN +
        '__yeto_op=; __yeto_os=; read -r __yeto_op __yeto_os < "$__yeto_pf" 2>/dev/null\n'
        'if [ -n "$__yeto_op" ] && { ! kill -0 "$__yeto_op" 2>/dev/null || [ -z "$__yeto_os" ] '
        '|| [ "$(__yeto_st "$__yeto_op")" != "$__yeto_os" ]; }; then rm -f "$__yeto_pf"; __yeto_op=; fi\n'
        'if [ -n "$__yeto_op" ]; then '
        '__yeto_age=$(( $(date +%s) - $(stat -c %Y "$__yeto_pf" 2>/dev/null || stat -f %m "$__yeto_pf" 2>/dev/null '
        '|| date +%s) )); '
        f'if [ "$__yeto_age" -gt {int(limit_s) + 5} ]; then echo "{BUSY_MARK} $__yeto_op"; exit 0; fi; fi\n'
        "__yeto_hf=$(mktemp /tmp/yeto-health.XXXXXX 2>/dev/null) || __yeto_hf=/tmp/yeto-health.$$\n"
        f"__yeto_hc={shlex.quote(PID_PRELUDE)}; export __yeto_hs __yeto_pf\n"
        'if command -v setsid >/dev/null 2>&1; then setsid bash -c "$__yeto_hc" > "$__yeto_hf" 2>&1 < /dev/null & '
        'else bash -c "$__yeto_hc" > "$__yeto_hf" 2>&1 < /dev/null & fi\n'
        "__yeto_hp=$!; __yeto_hn=0\n"
        f'while kill -0 "$__yeto_hp" 2>/dev/null && [ "$__yeto_hn" -lt {int(limit_s) * 10} ]; do sleep 0.1; '
        "__yeto_hn=$((__yeto_hn + 1)); done\n"
        'if kill -0 "$__yeto_hp" 2>/dev/null; then kill -9 -- "-$__yeto_hp" 2>/dev/null; '
        f'kill -9 "$__yeto_hp" 2>/dev/null; echo "{TIMEOUT_MARK} {int(limit_s)}"; sleep 0.2; fi\n'
        # gone: the next check may run; still there (stuck in the kernel): its pid stays
        'if ! kill -0 "$__yeto_hp" 2>/dev/null && [ "$(cut -d" " -f1 "$__yeto_pf" 2>/dev/null)" = "$__yeto_hp" ]; then '
        'rm -f "$__yeto_pf"; fi\n'
        'cat "$__yeto_hf" 2>/dev/null; rm -f "$__yeto_hf"\n'
    )
# the facts a check hands to the next one (baselines and the XID cursor); kept over a read failure
CARRIED = ("gpu_count", "bus", "mem", "ecc", "xid", "fm", "fm_boot", "fm_why", "fm_judged", "fm_restarts",
           "fm_ids", "ib_rate")
# the states of ``systemctl is-active nvidia-fabricmanager`` that are judged (anything else,
# an empty or timed-out answer, says nothing)
FM_STATES = ("active", "inactive", "failed", "activating", "deactivating")


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


def _boot_of(sec: Any) -> str:
    """The boot id the XID section printed first ('' when it did not)."""
    if sec is None:
        return ""
    for line in sec.text.splitlines():
        if line.startswith("boot="):
            return line[len("boot="):].strip()
    return ""


def _prev_boot(prev: Mapping[str, Any]) -> str:
    """The boot id of the previous check's XID cursor ('' when it has none)."""
    cursor = prev.get("xid")
    return str(cursor.get("boot") or "") if isinstance(cursor, Mapping) else ""


def read_fabric_manager(sec: Any) -> tuple[str, int | None, dict[str, Any]]:
    """``(state, NRestarts, probe)`` of the ``fabricmanager`` section (:data:`FM_SCRIPT`).

    The state, its process first: a state that is not ``active`` while an
    ``nv-fabricmanager`` process runs reads ``active`` (a GPU Operator driver container runs
    it, maybe next to a stale host unit that is disabled, inactive or failed). Else a unit
    that is ``not-found`` reads :data:`FM_ABSENT` (no unit, no process); ``inactive`` /
    ``failed`` of a unit that is not ``loaded`` (``masked``, unknown) reads :data:`FM_ABSENT`
    too when no process runs; of a ``loaded`` unit it stays (it stopped). The processes not
    seen (``proc1`` 0: hidepid; not counted): the state of an unloaded unit is ''. '' too when
    the state did not read (systemctl missing, no state printed).

    NRestarts is None when it did not read; ``probe`` holds the counts the section printed
    (``nvswitch_proc``, ``nvswitch_dev``, ``nvswitch_pci``, ``mlx``, ``mlx_vpd``, ``sw_mng``,
    ``fm_proc``; a count it did not print is missing: not read), ``fm_ids`` (the
    ``pid:starttime`` of each ``nv-fabricmanager`` process, when there is one), and
    ``timed_out`` when the section was cut off by its timeout (2: after systemctl's state,
    1: before)."""
    if sec is None:
        return "", None, {}
    kv: dict[str, str] = {}
    state = ""
    for line in sec.text.splitlines():
        line = line.strip()
        if "=" in line:
            key, _, value = line.partition("=")
            kv.setdefault(key.strip(), value.strip())
        elif line and not state:
            state = line.split()[0]
    if sec.rc == 127 or state not in FM_STATES:
        state = ""
    printed = bool(state)  # systemctl's state line came (the probe after it may not have)
    load = kv.get("loadstate")
    procs = None if kv.get("proc1") == "0" else _num(kv.get("fm_proc", ""))
    if state != "active" and procs:  # its process runs: the fabric manager is up
        state = "active"
    elif load == "not-found":
        state = "" if procs is None else FM_ABSENT
    elif load != "loaded" and state in ("inactive", "failed"):  # no loaded unit to have stopped
        state = FM_ABSENT if procs == 0 else ""
    probe: dict[str, Any] = {k: n for k in PROBE_KEYS if (n := _num(kv.get(k, ""))) is not None}
    if procs is not None:
        probe["fm_proc"] = procs
        if procs and kv.get("fm_ids"):
            probe["fm_ids"] = " ".join(sorted(kv["fm_ids"].split()))
    if sec.rc in (124, 137):
        probe["timed_out"] = 2 if printed else 1
    return state, _num(kv.get("nrestarts", "")), probe


FM_ABSENT = "not running"  # no unit and no nv-fabricmanager process


PROBE_KEYS = ("nvswitch_proc", "nvswitch_dev", "nvswitch_pci", "mlx", "mlx_vpd", "sw_mng")
NVSWITCH_YES, NVSWITCH_NO, NVSWITCH_UNREAD = "yes", "no", "unread"
_B_SXM = re.compile(r"^B\d{3}\b")  # B200 / B300 (HGX: SXM only)


def nvswitch_of(probe: Mapping[str, int], gpu_names: Sequence[str], fabric: bool | None) -> tuple[str, str]:
    """Whether the node has NVSwitch of its own: ``(yes | no | unread, basis)``.

    ``probe``: :func:`read_fabric_manager`'s counts; ``gpu_names``: nvidia-smi's names of
    the GPUs; ``fabric``: the GPUs have a real NVLink fabric state (``nvidia-smi -q``: not
    ``N/A``), False when none has, None when it did not read. In order:

    * yes: the NVIDIA driver shows NVSwitch devices (``/proc/driver/nvidia-nvswitch/devices``,
      ``/dev/nvidia-nvswitch<N>``) or lspci an NVIDIA PCI bridge (class 0680): Hopper HGX;
    * no: a GPU named ``GB...`` (an NVL72-class compute tray: its NVSwitches are in the
      switch trays), or no real fabric state (PCIe);
    * yes: a Mellanox function whose VPD names the NVLink switch management port
      (``SW_MNG``): HGX B200 / B300, whose NVSwitches sit behind a ConnectX bridge, not on
      the host's PCIe (no procfs entry, no device file, no class 0680); the VPD reads as
      root only;
    * yes: every GPU a B200 / B300 (not GB, not PCIe) with a real fabric state (the rule
      without root; a VPD without the port does not overrule it);
    * no: the VPD of every Mellanox function read, none names it;
    * unread: anything else (the fabric unread, another GPU with a fabric and no NVSwitch
      device shown): never taken for no."""
    names = [re.sub(r"^NVIDIA\s+", "", str(n).strip()).upper() for n in gpu_names if str(n).strip()]
    for key, basis in (("nvswitch_proc", "procfs"), ("nvswitch_dev", "device files"), ("nvswitch_pci", "lspci")):
        if probe.get(key, 0) > 0:
            return NVSWITCH_YES, basis
    if any(n.startswith("GB") for n in names):
        return NVSWITCH_NO, "compute tray (GB GPU)"
    if fabric is False:
        return NVSWITCH_NO, "no NVLink fabric"
    if probe.get("sw_mng", 0) > 0:
        return NVSWITCH_YES, "ConnectX VPD SW_MNG"
    if fabric and names and all(_B_SXM.match(n) and "PCIE" not in n for n in names):
        return NVSWITCH_YES, "B200/B300 GPU with an NVLink fabric"
    mlx, vpd = probe.get("mlx"), probe.get("mlx_vpd")
    if mlx and vpd == mlx and probe.get("sw_mng") == 0:
        return NVSWITCH_NO, "ConnectX VPD without SW_MNG"
    if probe.get("timed_out"):
        return NVSWITCH_UNREAD, ("FM section timed out after reading state" if probe["timed_out"] == 2
                                 else "FM section timed out")
    return NVSWITCH_UNREAD, "NVSwitch not determined"


FM_OK, FM_BAD, FM_UNREAD = "ok", "bad", "unread"


def nvswitch_hard(probe: Mapping[str, Any]) -> bool:
    """The node shows NVSwitch of its own: a device (procfs, device file), a PCI bridge, or a
    VPD naming the switch management port. The GPU names alone are no such evidence."""
    return any(int(probe.get(k) or 0) > 0 for k in ("nvswitch_proc", "nvswitch_dev", "nvswitch_pci", "sw_mng"))


OFF_HOST = "fabric managed off-host (no local NVSwitch device, fabric Completed)"


def fabric_manager_absolute(facts: Mapping[str, Any]) -> tuple[str, str]:
    """The fabric manager judged from one check alone (no earlier check): ``(verdict,
    detail)``, the verdict :data:`FM_OK`, :data:`FM_BAD` or :data:`FM_UNREAD`.

    ``facts``: a check's facts (:meth:`CudaChecker.parse`; its own read, never carried:
    ``nvswitch`` / ``nvswitch_by`` (:func:`nvswitch_of`), ``nvswitch_hard``
    (:func:`nvswitch_hard`), ``fabric_complete`` and ``fm_state``). In order:

    * a node known to have no NVSwitch of its own (PCIe, a GB compute tray, ConnectX VPD
      without the switch port): ok, whatever its state;
    * no loaded unit and no ``nv-fabricmanager`` process (:data:`FM_ABSENT`): bad when the
      node shows NVSwitch of its own (the processes are seen: the fabric manager is gone);
      else ok when every GPU's fabric is ``Completed/Success`` (:data:`OFF_HOST`: a shared
      NVSwitch guest VM, whose fabric manager runs on the service VM); else unread (the
      fabric state judges the node);
    * NVSwitch not determined, or the state not read (a check whose ``nvidia-smi`` did not
      answer has neither): unread;
    * a node with NVSwitch: ``active`` ok, any other state bad (``detail``: the reason).

    Unread is never ok nor bad."""
    nvswitch, state = facts.get("nvswitch"), facts.get("fm_state")
    if nvswitch == NVSWITCH_NO:
        return FM_OK, f"no NVSwitch on the node ({facts.get('nvswitch_by') or '?'})"
    if state == FM_ABSENT:
        if facts.get("nvswitch_hard") is True:
            return FM_BAD, f"nv-fabricmanager {FM_ABSENT} (NVSwitch present)"
        if facts.get("fabric_complete") is True:
            return FM_OK, OFF_HOST
        return FM_UNREAD, f"nv-fabricmanager {FM_ABSENT}, the NVLink fabric not Completed"
    if nvswitch != NVSWITCH_YES:
        return FM_UNREAD, str(facts.get("nvswitch_by") or "NVSwitch not determined")
    if state not in FM_STATES:
        return FM_UNREAD, "nvidia-fabricmanager state unread"
    if state == "active":
        return FM_OK, "nvidia-fabricmanager active"
    return FM_BAD, f"nvidia-fabricmanager {state} (NVSwitch present)"


def _gpu_names(sec: Any) -> list[str]:
    """nvidia-smi's GPU names (the gpus section's third column)."""
    if sec is None or sec.rc in (124, 137, 127):
        return []
    out = []
    for line in sec.text.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 5 and parts[0].isdigit():
            out.append(parts[2])
    return out


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
    the least rate of a job port (None: its rate at the first check); ``fabric``: the
    node's inventory fabric: a job without ``NCCL_IB_HCA`` is judged on the ports NCCL
    picks then (the preflight's rule, :func:`~yeto.hw.host_probe.link_layer_ports`)."""

    family = "cuda"

    def __init__(self, *, expect_gpus: int | None = None, min_ib_gbps: float | None = None,
                 fabric: str | None = None, **_: Any) -> None:
        self.expect_gpus = None if expect_gpus is None else int(expect_gpus)
        self.min_ib_gbps = None if min_ib_gbps is None else float(min_ib_gbps)
        self.fabric = fabric or None

    def script(self, hcas: Sequence[str]) -> str:  # noqa: ARG002 - every port is read; hcas judge them
        return bounded(TOOL_WRAPPER + probe_script(CUDA_SECTIONS))

    def parse(self, text: str, hcas: Sequence[str], prev: Mapping[str, Any] | None) -> HealthResult:
        prev = dict(prev or {})
        carried = {k: prev[k] for k in CARRIED if k in prev}
        sections = parse_sections(text)
        if BUSY_MARK in str(text or ""):  # the previous check still runs: this one ran nothing
            error = "node health check skipped: the previous one still runs (a tool stuck on the node)"
            return HealthResult(checked=False, reasons=[error], facts={**carried, "read_error": error})
        if TIMEOUT_MARK in str(text or ""):  # killed past its limit: only its completed XID section counts
            error = ("node health check cut off: its SSH call ran past its timeout"
                     if f"{TIMEOUT_MARK} ssh" in str(text) else
                     f"node health check timed out after {HEALTH_LIMIT_S}s (killed)")
            return self._unread(error, sections, prev, carried)
        gpus = sections.get("gpus")
        if gpus is None:  # nothing (an SSH call that failed, a check that did not run)
            return self._unread(_read_error(None), sections, prev, carried)
        cut = [name for name, _ in CUDA_SECTIONS if name not in sections or sections[name].rc is None]
        if cut:  # an output cut off: what it holds is not the whole answer
            return self._unread(f"node health output incomplete (no {', '.join(cut)})", sections, prev, carried)
        if DRIVER_DOWN in gpus.text:  # nvidia-smi answers, the driver does not (rc 9)
            return self._unread("", sections, prev, carried, driver_down=True)
        rows, gone = self._gpu_rows(gpus)
        if not rows and not gone and not self._no_devices(gpus):
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
        elif not hcas and (self.fabric is None or not ib_read):  # no port to judge: none named, none read
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
        fm, restarts, probe = read_fabric_manager(sections.get("fabricmanager"))
        sec = sections.get("fabric")
        out: list[str] = []
        real_fabric: bool | None = None  # a real NVLink fabric state (None: unread)
        facts["fabric_complete"] = False  # every GPU's fabric Completed/Success
        if sec is not None and sec.rc not in (124, 137):
            real = [(s, st) for s, st in parse_fabric(sec.text) if s.upper() != "N/A"]
            bad = [(s, st) for s, st in real if s.lower() != "completed" or (st and st.lower() != "success")]
            facts["fabric"] = f"{len(real) - len(bad)}/{len(real)} completed" if real else "none"
            if bad:
                state = "/".join(x for x in bad[0] if x)
                out.append(f"NVLink fabric {state} on {len(bad)}/{len(real)} GPUs (fabric manager: {fm or '?'})")
            if real or sec.rc in (0, 1):  # 1: no Fabric block (grep); another exit: nvidia-smi -q failed
                real_fabric = bool(real)
            facts["fabric_complete"] = bool(real) and not bad
        nvswitch, basis = nvswitch_of(probe, _gpu_names(sections.get("gpus")), real_fabric)
        facts["nvswitch"], facts["nvswitch_by"] = nvswitch, basis
        facts["nvswitch_hard"] = nvswitch_hard(probe)
        return CudaChecker._fabric_manager(fm, restarts, nvswitch, _boot_of(sections.get("xid")), prev, facts,
                                           ids=probe.get("fm_ids")) + out

    @staticmethod
    def _fabric_manager(fm: str, restarts: int | None, nvswitch: str, boot: str,
                        prev: Mapping[str, Any], facts: dict[str, Any], *, ids: str | None = None) -> list[str]:
        """The fabric manager's reasons over the checks of a watch.

        ``fm`` is this read's state ('' unread), ``restarts`` its unit's ``NRestarts`` (None
        unread), ``nvswitch`` whether the node has NVSwitch of its own (:func:`nvswitch_of`:
        yes, no, unread). A node with none (``no``: PCIe, a GB compute tray) is never judged
        on its fabric manager. ``fm_boot`` (carried) is the boot in which the fabric manager
        must be active: it was active in it (an active fabric manager means there is a
        fabric to manage: it judges a node whose NVSwitch was not determined), or (``fm_why``
        ``nvswitch``) the first judged read of the watch found it down on a node with
        NVSwitch (:func:`fabric_manager_absolute`). In that boot anything but ``active`` is a
        reason at every check, so a fabric manager that stays down reaches ``lost``. A new
        boot forgets it: a fabric manager not active since the reboot is judged as before
        (one reason when it was active before the reboot, on a node with NVSwitch; then the
        fabric state judges the node).

        The first judged read (``fm_judged`` not yet true: a new launcher) is the one where
        the state read and NVSwitch was determined; a read where either did not leaves the
        next read first. An older ``prev`` without ``fm_boot`` whose fabric manager was
        active: the boot of its XID cursor (else this one); without ``fm_judged``: judged
        when it has ``fm``.

        The state is the process's first (:func:`read_fabric_manager`): an
        ``nv-fabricmanager`` process reads ``active`` (and is remembered so); none and no
        loaded unit reads :data:`FM_ABSENT`, a reason in a boot in which it must run, and at
        the first judged read when :func:`fabric_manager_absolute` says bad. Without a unit
        (no ``NRestarts``), another ``nv-fabricmanager`` process (pid and start time) in the
        same boot is a restart, a reason for that check.

        On a node that must run it (``fm_boot`` this boot, or NVSwitch), ``NRestarts`` that
        grew since the previous check of the same boot is a reason for that check: a unit
        that restarts between two checks (systemd's own ``Restart=``) reads ``active`` at
        both."""
        if "fm_boot" in prev:
            need, why = prev.get("fm_boot") or None, prev.get("fm_why") or None
        elif prev.get("fm") == "active":
            need, why = _prev_boot(prev) or boot or None, None
        else:
            need = why = None
        if need and boot and need != boot:  # a new boot: forgotten
            need = why = None
        was_judged = bool(prev.get("fm_judged", "fm" in prev))
        facts["fm_state"] = fm if fm in FM_STATES or fm == FM_ABSENT else ""
        absolute, detail = fabric_manager_absolute(facts)
        judged = absolute != FM_UNREAD
        if fm == "active" and boot:
            need, why = boot, None
        elif not was_judged and absolute == FM_BAD and boot and not need:
            need, why = boot, "nvswitch"
        facts["fm_judged"] = was_judged or judged
        for key, value in (("fm_boot", need), ("fm_why", why)):
            if value:
                facts[key] = value
            else:
                facts.pop(key, None)
        if fm in FM_STATES or fm == FM_ABSENT:
            facts["fm"] = fm
        if nvswitch == NVSWITCH_NO:  # no NVSwitch of its own: nothing to manage here
            if restarts is not None and boot:
                facts["fm_restarts"] = {"boot": boot, "n": restarts}
            if ids and boot:
                facts["fm_ids"] = {"boot": boot, "ids": ids}
            return []
        must = bool(need) and need == boot  # it must be active in this boot
        out = []
        if fm == FM_ABSENT and must:  # no unit: its process ran in this boot and is gone
            out.append(f"nv-fabricmanager {fm} ({'NVSwitch present' if why == 'nvswitch' else 'was active'})")
        if fm in FM_STATES and fm != "active":
            if must:
                out.append(f"nvidia-fabricmanager {fm} ({'NVSwitch present' if why == 'nvswitch' else 'was active'})")
            elif not was_judged and absolute == FM_BAD:  # the first judged read, the boot unknown
                out.append(detail)
            elif prev.get("fm") == "active" and fm in ("inactive", "failed") and nvswitch == NVSWITCH_YES:
                out.append(f"nvidia-fabricmanager {fm} (was active)")  # it stopped across a reboot (or boot unknown)
        base = prev.get("fm_restarts")
        if restarts is not None and boot:
            before = base.get("n") if isinstance(base, Mapping) and base.get("boot") == boot else None
            if (must or nvswitch == NVSWITCH_YES) and before is not None and restarts > int(before):
                out.append(f"nvidia-fabricmanager restarted {restarts - int(before)}x since the previous check")
            facts["fm_restarts"] = {"boot": boot, "n": restarts}
        # a fabric manager without its unit (no NRestarts): its process changed in the boot
        base = prev.get("fm_ids")
        if ids and boot:
            before = base.get("ids") if isinstance(base, Mapping) and base.get("boot") == boot else None
            if (must or nvswitch == NVSWITCH_YES) and before and before != ids:
                out.append("nv-fabricmanager restarted (another process since the previous check)")
            facts["fm_ids"] = {"boot": boot, "ids": ids}
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
        else:  # no timestamps: after the cursor's place in the log (its head may have rotated out)
            new = found[CudaChecker._no_ts_start(texts, cursor):]
        stamps = [ts for ts, _, _ in found if ts is not None]
        run = next((i for i, t in enumerate(reversed(texts)) if t != texts[-1]), len(texts)) if texts else 0
        facts["xid"] = {"boot": boot, "ts": max(stamps) if stamps else cursor.get("ts"), "count": len(found),
                        "last": texts[-1] if texts else None,
                        "last_n": texts.count(texts[-1]) if texts else 0,
                        "run": run, "before": texts[-run - 1] if len(texts) > run else None}
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

    @staticmethod
    def _no_ts_start(texts: list[str], cursor: Mapping[str, Any]) -> int:
        """Where the XID lines after the cursor start, in a log without timestamps.

        The cursor's anchor is its last line, how many times it ends the log (``run``) and the
        line before that run (``before``): the newest place of ``before`` and the run is the
        cursor (a head of the log that rotated out does not move it). Without an anchor (the
        log held one repeated line, or its anchor rotated out): after the ``last_n``-th time
        of the last line, else after its last time. Limit: a log that holds a single repeated
        XID line and rotates one copy out while the same XID repeats misses that repeat; a
        kernel log with timestamps (dmesg's default) has none of this."""
        last, before, run = cursor.get("last"), cursor.get("before"), int(cursor.get("run") or 0)
        if last is not None and before is not None and run:
            want = [before] + [last] * run
            for i in range(len(texts) - len(want), -1, -1):
                if texts[i:i + len(want)] == want:
                    return i + len(want)
        at = [i for i, t in enumerate(texts) if t == last]
        n = int(cursor.get("last_n") or 0)
        return at[n - 1] + 1 if n and len(at) >= n else (at[-1] + 1 if at else 0)

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
        elif self.fabric is not None:  # NCCL's own pick without NCCL_IB_HCA: by the fabric's link layer
            # as the preflight: bad when none of them is active (one port down: NCCL uses the others)
            picked = link_layer_ports(ports, self.fabric)
            if not any(p.active for p in picked):
                down = ", ".join(f"{_port_label(p)} {(p.state or 'unknown').upper()}" for p in picked) or "none found"
                out.append(f"no active {self.fabric} port ({down})")
            selected = []
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


def degraded_of(result: HealthResult, *, multi: bool, hcas: Sequence[str], fabric: str | None = None) -> list[str]:
    """What a read check could not judge for the job: XIDs (no kernel log access), the
    ports (the IB tools did not read; a multi-node job without ``NCCL_IB_HCA`` nor a
    fabric to pick its ports by)."""
    if not result.checked:
        return []
    out = []
    if result.facts.get("xid_source") == "none":
        out.append("XID unread (no dmesg / journalctl -k access)")
    if multi and hcas and result.facts.get("ib") == "unread":
        out.append(IB_UNREAD)
    if multi and not hcas and fabric is None:
        out.append("no NCCL_IB_HCA: the ports are not judged")
    elif multi and not hcas and result.facts.get("ib") == "unread":
        out.append(IB_UNREAD)
    return out


class HealthWatch:
    """One node's health over time: its checker, its tracker and the previous facts.
    ``hcas``: the job's ``NCCL_IB_HCA`` items; ``multi``: the job spans nodes; ``every_s``:
    the period of its checks (in its record: a reader takes an older one for a read failure)."""

    def __init__(self, family: str, *, hcas: Sequence[str] = (), multi: bool = False,
                 expect_gpus: int | None = None, min_ib_gbps: float | None = None,
                 bad_checks: int = BAD_CHECKS, checker: Checker | None = None,
                 clock: Callable[[], float] = time.time, every_s: float = HEALTH_EVERY_S,
                 fabric: str | None = None) -> None:
        self.fabric = fabric or None
        self.checker = checker or checker_for(family, expect_gpus=expect_gpus, min_ib_gbps=min_ib_gbps,
                                              fabric=self.fabric)
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
        self.kept: dict[str, Any] | None = None  # the record before a reset, until the next check

    @property
    def enabled(self) -> bool:
        return bool(self.checker.enabled)

    def script(self) -> str:
        return self.checker.script(self.hcas)

    def reset(self) -> None:
        """A new job on the same node (a relaunch): the bad streak and the last verdict go;
        the baselines and the XID cursor of the previous checks stay, and so does the record
        (``reset: true``) until the next check replaces it."""
        kept = self.record()
        self.kept = None if kept is None else {**kept, "reset": True}
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
            found = degraded_of(result, multi=self.multi, hcas=self.hcas, fabric=self.fabric)
            if found:
                self.degraded = degraded = found
        self.read_failed_in_row = self.read_failed_in_row + 1 if verdict == "read_failed" else 0
        if self.read_failed_in_row == READ_FAILED_DEGRADE:  # once per streak: it says so, it loses nothing
            why = "; ".join(result.job_reasons(self.multi)) or "unread"
            degraded = (degraded or []) + [f"read failed {READ_FAILED_DEGRADE} checks in a row: {why}"]
        self.prev = dict(result.facts)
        self.last, self.kept = Reading(verdict, result, event, degraded), None
        self.at = float(self.clock())
        return self.last

    def record(self) -> dict[str, Any] | None:
        """The ``health`` field of a heartbeat / an island state (None before the first check)."""
        if self.last is None:
            return None if self.kept is None else dict(self.kept)
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
    parser = ExactArgumentParser(prog="python -m yeto.hw.node_health",
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
