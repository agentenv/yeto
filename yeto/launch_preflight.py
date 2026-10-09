"""Launch preflight checks (openspec change launch-preflight-guards).

Two checks run BEFORE the launcher creates any cloud resource:

* thread preflight: the number of threads of this user (the same number as
  ``ps -L -u <user> | wc -l`` without the header line).  The s1run thread
  guard (THREAD_MAX 3300, ulimit -u 4096) tears down a live run when the user
  goes over its limit, so a launch must not start near that limit.
* memory preflight: an upper-bound estimate of the per-GPU memory peak of each
  island (``yeto.memory_estimate``); a launch that is known to run out of GPU
  memory stops here, at zero cost.

Both checks have an explicit off switch; using it prints a warning and the
launcher records it in the run manifest (``preflight`` key).

The launcher calls :func:`thread_preflight` and :func:`memory_preflight`; the
rest of this module is pure Python and reads /proc only, so the unit tests can
give it a fake /proc directory and a fake clock.
"""

from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable

# ---------------------------------------------------------------------------
# defaults (design decision 1)
# ---------------------------------------------------------------------------
THREAD_MODES = ("wait", "error", "off")
DEFAULT_THREAD_MODE = "wait"
DEFAULT_THREAD_START = 2800
DEFAULT_THREAD_HARD = 3000
DEFAULT_THREAD_WAIT_S = 1800.0
THREAD_POLL_S = 30.0

MEMORY_MODES = ("error", "warn", "off")
DEFAULT_MEMORY_MODE = "error"
DEFAULT_MEMORY_MARGIN = 0.9

SKY_API_RESTART = "sky api stop && sky api start"


class PreflightError(ValueError):
    """A preflight check refused the launch (no cloud resource was created)."""


# ---------------------------------------------------------------------------
# thread counting (task 1.1)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ThreadCount:
    total: int
    sky_api: int
    sky_api_found: bool
    processes: int

    @property
    def sky_api_share(self) -> float:
        return self.sky_api / self.total if self.total else 0.0

    def to_dict(self) -> dict:
        return {"total": self.total, "sky_api_server": self.sky_api,
                "sky_api_server_found": self.sky_api_found, "processes": self.processes}


def _status_fields(text: str) -> dict[str, str]:
    out = {}
    for line in text.splitlines():
        key, sep, value = line.partition(":")
        if sep:
            out[key.strip()] = value.strip()
    return out


def _is_sky_api(cmdline: str) -> bool:
    # The API server runs as ``python -m sky.server.server``; its request
    # executors rename themselves ``SkyPilot:executor:<kind>:<pid>``.
    return "sky.server" in cmdline or cmdline.startswith("SkyPilot:")


def count_threads(proc_root: str | os.PathLike[str] = "/proc", uid: int | None = None) -> ThreadCount:
    """Sum ``Threads:`` of every process of ``uid`` under ``proc_root``.

    Processes of the SkyPilot API server (command line contains
    ``sky.server``, its executors and every descendant) are summed apart.
    A process that ends while it is read is skipped."""

    root = Path(proc_root)
    uid = os.getuid() if uid is None else int(uid)
    procs: dict[int, tuple[int, int, str]] = {}  # pid -> (ppid, threads, cmdline)
    try:
        entries = list(root.iterdir())
    except OSError as e:
        raise PreflightError(f"thread preflight: cannot read {root}: {e}") from e
    for entry in entries:
        if not entry.name.isdigit():
            continue
        try:
            fields = _status_fields((entry / "status").read_text(encoding="utf-8", errors="replace"))
            uids = fields.get("Uid", "").split()
            if not uids or int(uids[0]) != uid:
                continue
            threads = int(fields.get("Threads", "0") or 0)
            ppid = int(fields.get("PPid", "0") or 0)
            try:
                raw = (entry / "cmdline").read_bytes()
            except OSError:
                raw = b""
            cmd = raw.replace(b"\0", b" ").decode("utf-8", errors="replace").strip()
        except (OSError, ValueError):
            continue
        procs[int(entry.name)] = (ppid, threads, cmd)
    sky_roots = {pid for pid, (_, _, cmd) in procs.items() if _is_sky_api(cmd)}
    sky = set()
    for pid in procs:
        seen = set()
        cur = pid
        while cur in procs and cur not in seen:
            if cur in sky_roots:
                sky.add(pid)
                break
            seen.add(cur)
            cur = procs[cur][0]
    total = sum(t for _, t, _ in procs.values())
    return ThreadCount(total=total, sky_api=sum(procs[p][1] for p in sky),
                       sky_api_found=bool(sky_roots), processes=len(procs))


# ---------------------------------------------------------------------------
# thread preflight (tasks 1.3, 1.4)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ThreadConfig:
    mode: str = DEFAULT_THREAD_MODE
    start: int = DEFAULT_THREAD_START
    hard: int = DEFAULT_THREAD_HARD
    wait_s: float = DEFAULT_THREAD_WAIT_S
    poll_s: float = THREAD_POLL_S

    @classmethod
    def from_args(cls, args) -> "ThreadConfig":
        cfg = cls(
            mode=getattr(args, "preflight_threads", None) or DEFAULT_THREAD_MODE,
            start=int(getattr(args, "preflight_thread_start", None) or DEFAULT_THREAD_START),
            hard=int(getattr(args, "preflight_thread_hard", None) or DEFAULT_THREAD_HARD),
            wait_s=float(DEFAULT_THREAD_WAIT_S if getattr(args, "preflight_thread_wait_s", None) is None
                         else args.preflight_thread_wait_s),
        )
        cfg.validate()
        return cfg

    def validate(self) -> None:
        if self.mode not in THREAD_MODES:
            raise PreflightError(f"--preflight-threads must be one of {THREAD_MODES}, got {self.mode!r}")
        if self.start < 1 or self.hard < 1:
            raise PreflightError("--preflight-thread-start and --preflight-thread-hard must be >= 1")
        if self.start > self.hard:
            raise PreflightError(f"--preflight-thread-start ({self.start}) must not be above "
                                 f"--preflight-thread-hard ({self.hard})")
        if self.wait_s < 0:
            raise PreflightError("--preflight-thread-wait-s must be >= 0")


def thread_message(count: ThreadCount, cfg: ThreadConfig, headline: str) -> str:
    """The text of a thread preflight error or wait line (task 1.4)."""

    if count.sky_api_found:
        sky = (f"SkyPilot API server: {count.sky_api} threads "
               f"({count.sky_api_share:.0%} of {count.total})")
    else:
        sky = "SkyPilot API server: not found (未找到 SkyPilot API 服务)"
    return "\n".join([
        f"[preflight] {headline}",
        f"[preflight]   user threads {count.total}, start threshold {cfg.start}, hard limit {cfg.hard}",
        f"[preflight]   {sky}",
        f"[preflight]   to free threads: `{SKY_API_RESTART}` -- first make sure no launch is in "
        "progress (a restart cancels a running sky.launch)",
        "[preflight]   do not run local tests that start Ray (gcs_server/raylet) while a GPU "
        "launch is live",
        "[preflight]   switch: --preflight-threads {wait,error,off}",
    ])


def thread_preflight(args, *, counter: Callable[[], ThreadCount] | None = None,
                     clock: Callable[[], float] = time.monotonic,
                     sleep: Callable[[float], None] = time.sleep,
                     out=None) -> dict:
    """Run the thread preflight; return the manifest record.

    Below ``start``: continue.  ``start`` <= n < ``hard``: ``wait`` re-reads every
    ``poll_s`` until below ``start`` (error after ``wait_s``), ``error`` stops now.
    n >= ``hard``: stop now in every mode.  ``off``: no read, a warning."""

    out = sys.stderr if out is None else out
    cfg = ThreadConfig.from_args(args)
    if cfg.mode == "off":
        print("[preflight] WARNING: thread preflight is OFF (--preflight-threads off); the "
              "s1run thread guard may tear down this run", file=out, flush=True)
        return {"mode": "off", "disabled": True}
    counter = counter or count_threads
    started = clock()
    count = counter()
    readings = [count.total]
    waited = False
    while True:
        if count.total >= cfg.hard:
            raise PreflightError(thread_message(
                count, cfg, f"STOP: user threads {count.total} >= hard limit {cfg.hard}; "
                "no cloud resource was created"))
        if count.total < cfg.start:
            break
        if cfg.mode == "error":
            raise PreflightError(thread_message(
                count, cfg, f"STOP: user threads {count.total} >= start threshold {cfg.start} "
                "(--preflight-threads error); no cloud resource was created"))
        elapsed = clock() - started
        if elapsed >= cfg.wait_s:
            raise PreflightError(thread_message(
                count, cfg, f"STOP: user threads still {count.total} >= {cfg.start} after waiting "
                f"{elapsed:.0f} s (limit {cfg.wait_s:.0f} s); no cloud resource was created"))
        print(thread_message(count, cfg, f"WAIT: user threads {count.total} >= start threshold "
                             f"{cfg.start}; re-reading in {cfg.poll_s:.0f} s "
                             f"({elapsed:.0f}/{cfg.wait_s:.0f} s waited)"), file=out, flush=True)
        waited = True
        sleep(cfg.poll_s)
        count = counter()
        readings.append(count.total)
    print(f"[preflight] threads OK: {count.total} < {cfg.start} "
          f"(SkyPilot API server {count.sky_api if count.sky_api_found else 'not found'})",
          file=out, flush=True)
    return {"mode": cfg.mode, "start": cfg.start, "hard": cfg.hard, "wait_s": cfg.wait_s,
            "waited": waited, "readings": readings, **count.to_dict()}


# ---------------------------------------------------------------------------
# memory preflight (tasks 2.3, 2.4, 2.5)
# ---------------------------------------------------------------------------
@dataclass
class MemoryReport:
    mode: str
    margin: float
    islands: list[dict] = field(default_factory=list)
    disabled: bool = False

    def to_dict(self) -> dict:
        return {"mode": self.mode, "margin": self.margin, "disabled": self.disabled,
                "islands": self.islands}


def memory_config(args) -> tuple[str, float]:
    mode = getattr(args, "preflight_memory", None) or DEFAULT_MEMORY_MODE
    if mode not in MEMORY_MODES:
        raise PreflightError(f"--preflight-memory must be one of {MEMORY_MODES}, got {mode!r}")
    margin = getattr(args, "preflight_memory_margin", None)
    margin = DEFAULT_MEMORY_MARGIN if margin is None else float(margin)
    if not 0 < margin <= 1:
        raise PreflightError("--preflight-memory-margin must be in (0, 1]")
    return mode, margin


def island_memory_estimates(args, specs: Iterable, *, margin: float) -> list[dict]:
    """Per-island estimate records (dry-run ``memory_estimate`` and manifest)."""

    from .memory_estimate import estimate_for_launch

    out = []
    for learner_id, spec in enumerate(specs):
        est = estimate_for_launch(args, spec, margin=margin)
        out.append({"learner_id": learner_id, **est})
    return out


def memory_preflight(args, specs, *, out=None) -> dict:
    """Estimate every island's per-GPU peak; refuse (mode ``error``) when one
    is above ``margin`` x GPU memory.  Unknown model: warn, record, continue."""

    out = sys.stderr if out is None else out
    mode, margin = memory_config(args)
    if mode == "off":
        print("[preflight] WARNING: GPU memory preflight is OFF (--preflight-memory off); "
              "an out-of-memory island is found only after it is paid for", file=out, flush=True)
        return MemoryReport(mode, margin, disabled=True).to_dict()
    report = MemoryReport(mode, margin, island_memory_estimates(args, specs, margin=margin))
    over = []
    for rec in report.islands:
        if not rec.get("estimated"):
            print(f"[preflight] WARNING: GPU memory not estimated for island {rec['learner_id']} "
                  f"(显存未估算): {rec.get('reason')}", file=out, flush=True)
            continue
        print(f"[preflight] island {rec['learner_id']} {rec['gpu']}: estimated peak "
              f"{rec['peak_gib']:.1f} GiB / limit {rec['limit_gib']:.1f} GiB "
              f"({'OK' if rec['fits'] else 'OVER'})", file=out, flush=True)
        if not rec["fits"]:
            if rec.get("calibrated"):
                over.append(rec)
            else:
                # design Risks: constants are calibrated on Qwen3.5-4B only; for
                # other models an over-limit estimate warns and never refuses.
                print(format_memory_refusal(rec).replace("STOP:", "WARNING (model not calibrated, "
                                                         "advisory only):", 1), file=out, flush=True)
    if over:
        text = "\n".join(format_memory_refusal(rec) for rec in over)
        if mode == "error":
            raise PreflightError(text + "\n[preflight]   no cloud resource was created; switch: "
                                 "--preflight-memory {error,warn,off}")
        print(text + "\n[preflight]   --preflight-memory warn: continuing anyway", file=out, flush=True)
    return report.to_dict()


def format_memory_refusal(rec: dict) -> str:
    lines = [f"[preflight] STOP: island {rec['learner_id']} ({rec['gpu']}, {rec['gpu_mem_gib']} GiB) "
             f"estimated peak {rec['peak_gib']:.1f} GiB > {rec['margin']:.0%} limit "
             f"{rec['limit_gib']:.1f} GiB per GPU",
             "[preflight]   parts (GiB):"]
    for key, value in rec["parts_gib"].items():
        lines.append(f"[preflight]     {key:<22} {value:8.2f}")
    if rec.get("suggestions"):
        lines.append("[preflight]   suggestions that fit (estimated peak after the change):")
        for s in rec["suggestions"]:
            lines.append(f"[preflight]     - {s['change']}: {s['peak_gib']:.1f} GiB")
    else:
        lines.append("[preflight]   no tried suggestion fits (shorter context, 2x GPUs, larger GPU)")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI (tasks 1.2, 2.4, 3.1): one call from cli._add_launch_args
# ---------------------------------------------------------------------------
def add_cli_args(p) -> None:
    g = p.add_argument_group("launch preflight (before any cloud resource)")
    g.add_argument("--preflight-threads", choices=THREAD_MODES, default=DEFAULT_THREAD_MODE,
                   help="user thread count check before launch: wait (default) re-reads every "
                   f"{THREAD_POLL_S:.0f} s while start <= threads < hard; error stops at once; off "
                   "skips it (recorded in the run manifest). threads >= hard always stops")
    g.add_argument("--preflight-thread-start", type=int, default=DEFAULT_THREAD_START, metavar="N",
                   help=f"launch only below N user threads (default {DEFAULT_THREAD_START})")
    g.add_argument("--preflight-thread-hard", type=int, default=DEFAULT_THREAD_HARD, metavar="N",
                   help=f"stop at once at N user threads or more (default {DEFAULT_THREAD_HARD})")
    g.add_argument("--preflight-thread-wait-s", type=float, default=DEFAULT_THREAD_WAIT_S,
                   metavar="S", help=f"wait limit of --preflight-threads wait (default "
                   f"{DEFAULT_THREAD_WAIT_S:.0f})")
    g.add_argument("--preflight-memory", choices=MEMORY_MODES, default=DEFAULT_MEMORY_MODE,
                   help="per-GPU memory peak estimate of each RL island: error (default) stops "
                   "when above the margin, warn prints and continues, off skips it (recorded)")
    g.add_argument("--preflight-memory-margin", type=float, default=DEFAULT_MEMORY_MARGIN,
                   metavar="F", help="allowed peak as a fraction of GPU memory (default 0.9)")
    g.add_argument("--rl-island-override", action="append", default=None,
                   metavar="ISLAND:KEY=VALUE",
                   help="NEGATIVE TEST ONLY: give one island another value of rl_lr_schedule, "
                   "rl_max_policy_age or identity_test_salt (repeatable); needs "
                   "--rl-negative-test-run")
    g.add_argument("--rl-negative-join-delay-s", type=float, default=None, metavar="S",
                   help="negative-test run: an overridden island waits S seconds before it starts "
                   "(default 180), so the unchanged island sets the syncer's session contract "
                   "and the overridden one is refused")
    g.add_argument("--rl-negative-test-run", action="store_true",
                   help="mark this run as a negative-test run (required by --rl-island-override); "
                   "a normal run refuses to resume its checkpoint store and `yeto merge` refuses "
                   "its adapters")


def launch_preflight(args, specs, *, counter=None, sleep=time.sleep, clock=time.monotonic,
                     out=None) -> dict:
    """Thread then memory preflight; the ``preflight`` record of the run manifest.
    Raises PreflightError (no cloud resource created)."""
    threads = thread_preflight(args, counter=counter, sleep=sleep, clock=clock, out=out)
    memory = memory_preflight(args, specs, out=out)
    return {"threads": threads, "memory": memory}


def pre_cloud_checks(args, *, counter=None, sleep=time.sleep, clock=time.monotonic, out=None,
                     threads: bool = True) -> dict:
    """Every launch-preflight-guards check that must pass before any cloud
    resource, on PREPARED args (``prepare_launch_args`` already ran), except the
    thread check when ``threads`` is False (the launcher runs it earlier, before
    ``sky_patches.install``).  Stores the manifest record on
    ``args._launch_preflight_manifest`` and returns it."""
    from .gpu_spec import parse_gpu_spec
    from . import island_overrides as io

    specs = parse_gpu_spec(args.gpu)
    record = dict(getattr(args, "_launch_preflight_manifest", None) or {})
    preflight = dict(record.get("preflight") or {})
    if threads:
        preflight["threads"] = thread_preflight(args, counter=counter, sleep=sleep, clock=clock, out=out)
    preflight["memory"] = memory_preflight(args, specs, out=out)
    check_policy_age_spec(args)  # global flag (island copies: island_overrides.check_island_args)
    num_islands = len(specs) + max(0, getattr(args, "external_learners", 0) or 0)
    overrides = io.overrides_of(args, len(specs))
    for island in overrides:
        io.island_args(args, island, overrides)  # re-run the per-parameter checks on the copy
    io.check_resume_before_launch(args, out=out)
    if overrides or getattr(args, "rl_negative_test_run", False):
        io.print_warning(args, overrides, out=out)
    record["preflight"] = preflight
    record.update(io.manifest_fields(args, overrides))
    record["islands_checked"] = num_islands
    args._launch_preflight_manifest = record
    return record


def dry_run_memory(args, specs) -> list[dict] | None:
    """Task 2.5: per-island ``memory_estimate`` of ``yeto launch --dry-run``
    (None when ``--preflight-memory off``)."""
    mode, margin = memory_config(args)
    if mode == "off":
        return None
    return island_memory_estimates(args, specs, margin=margin)


def check_policy_age_spec(args) -> None:
    """``--rl-max-policy-age N`` (global or a per-island override) needs an
    algorithm spec with ``execution.max_policy_staleness >= N`` -- the same rule
    the Miles island applies in its config translation (adapters/miles/config.py)
    and its A1 preflight, now checked before any cloud resource (10-09 ruling of
    the main agent: S18 LPG run s18-lpg-age-20261009a failed on the island).
    Needs prepared args (``rl_algorithm_spec_json`` resolved)."""
    if getattr(args, "training_mode", "sft") != "rl":
        return
    age = int(getattr(args, "rl_max_policy_age", 0) or 0)
    if age == 0:
        return
    import json as _json

    from .rl.engine.algorithm import AlgorithmSpec
    from .rl.engine.execution_profile import algorithm_max_policy_staleness

    raw = getattr(args, "rl_algorithm_spec_json", None)
    tolerated = (algorithm_max_policy_staleness(AlgorithmSpec.from_dict(_json.loads(raw)))
                 if raw else 0)
    if tolerated < age:
        raise PreflightError(
            f"--rl-max-policy-age {age} needs an algorithm spec with "
            f"execution.max_policy_staleness >= {age} (and a TIS correction); the run's spec "
            f"tolerates {tolerated} (--rl-algorithm-spec)")
