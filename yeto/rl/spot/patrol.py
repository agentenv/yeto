"""Patrol (and stop-loss) for a spot RL run (trainer + spot rollout islands).

Reads, all optional:

* ``--tape``: the trainer's event tape (JSONL or a log with ``YETO_RL_EVENT <json>`` lines):
  ``rl_local_round`` (train/rollout seconds, grad_norm, nonzero_advantage_count),
  ``rl_verl_fa_round`` (pulled/kept/discarded, uniform_reward_groups), ``rl_host_memory``
  (per node MemAvailable and worker RssAnon);
* ``--consumed``: ``verl-buffer-consumed-<learner>.jsonl`` (kept / dropped per group);
* ``--sampler-tape`` (repeatable): sampler island tapes (``rl_agentic_group``,
  ``rl_sampler_engine_round``, ``rl_sampler_leave``) -> heartbeat, output, policy lag,
  release delay;
* Nebius instances (``nebius compute instance list --format json``) and boat usage
  (``boat limits``, JSON): run with ``--nebius-cmd`` / ``--boat-cmd`` or from files
  (``--nebius-json`` / ``--boat-json``);
* ``--spent-usd`` / ``--budget-usd``.

It prints one JSON object (``alerts`` with ``key``, ``severity`` warn|stop, ``message``,
``value``, ``threshold``) and exits 0 (quiet), 1 (warnings) or 2 (a stop-loss alert).
Thresholds: :class:`Thresholds`.

Read-only by default. ``--act`` runs the stop-loss for ``stop`` alerts: stop the
trainer (``--kill-script`` over ssh), delete the run's Nebius VMs and sweep the run's
boat sandboxes (``boat_labels sweep``); a VM stuck in STARTING is deleted on its own.
Every external call goes through an injectable runner, so the logic is unit tested
offline (tests/test_spot_patrol.py).

    python -m yeto.rl.spot.patrol --tape TRAINER.log --sampler-tape ISLAND.jsonl \\
        --nebius-cmd 'nebius compute instance list --parent-id P --format json' --boat-cmd 'boat limits'
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import os
import shlex
import subprocess
import sys
import time
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path
from typing import Any

EVENT_PREFIX = "YETO_RL_EVENT "
GIB_MIB = 1024.0
STARTING_STATES = frozenset({"STARTING", "CREATING", "PROVISIONING"})


@dataclasses.dataclass
class Thresholds:
    """Alert thresholds of the patrol."""

    train_s: float = 220.0  # one step, the first excluded
    wait_samples_s: float = 45 * 60.0
    policy_lag: int = 3  # >=
    window: int = 10  # steps of the drop / advantage windows
    drop_rate: float = 0.20  # >
    nonzero_adv_groups: float = 0.30  # <
    mem_available_gib: float = 200.0  # <
    rssanon_slope_gib: float = 2.0  # > per step
    island_silent_s: float = 30 * 60.0
    vm_starting_s: float = 10 * 60.0
    boat_quota: float = 0.80  # >=
    spend: float = 0.80  # >= of the budget
    release_delay_s: float = 60 * 60.0
    grad_bad_steps: int = 1  # consecutive zero / non-finite grad_norm steps to stop


# which alerts stop the run under --act (the rest only warn)
STOP_KEYS = frozenset({"grad_norm_bad", "mem_available_low", "rssanon_slope", "spend_budget", "boat_quota"})


@dataclasses.dataclass
class Alert:
    key: str
    severity: str
    message: str
    value: Any = None
    threshold: Any = None
    target: str | None = None  # e.g. the VM id an action is for

    def to_dict(self) -> dict[str, Any]:
        return {k: v for k, v in dataclasses.asdict(self).items() if v is not None}


# ------------------------------------------------------------------ reading
def read_events(path: str | Path | None) -> list[dict[str, Any]]:
    """Event records of a JSONL tape or a log with ``YETO_RL_EVENT`` lines; torn or
    foreign lines are skipped. A missing file reads as no events."""
    if path is None:
        return []
    try:
        text = Path(path).expanduser().read_text(encoding="utf-8", errors="replace")
    except OSError:
        return []
    out = []
    for line in text.splitlines():
        line = line.strip()
        if EVENT_PREFIX in line:
            line = line.split(EVENT_PREFIX, 1)[1]
        if not line.startswith("{"):
            continue
        try:
            record = json.loads(line)
        except ValueError:
            continue
        if isinstance(record, dict):
            out.append(record)
    return out


def _of(events: Iterable[dict[str, Any]], name: str) -> list[dict[str, Any]]:
    return [e for e in events if e.get("event") == name]


def _num(value: Any) -> float | None:
    try:
        return None if value is None or isinstance(value, bool) else float(value)
    except (TypeError, ValueError):
        return None


def slope(points: Sequence[tuple[float, float]]) -> float | None:
    """Least-squares slope of ``(x, y)`` points (None with fewer than 3 or no x spread)."""
    if len(points) < 3:
        return None
    n = len(points)
    mx = sum(x for x, _ in points) / n
    my = sum(y for _, y in points) / n
    sxx = sum((x - mx) ** 2 for x, _ in points)
    if sxx == 0:
        return None
    return sum((x - mx) * (y - my) for x, y in points) / sxx


# ------------------------------------------------------------------ checks
def trainer_alerts(events: list[dict[str, Any]], now: float, th: Thresholds) -> list[Alert]:
    alerts: list[Alert] = []
    rounds = _of(events, "rl_local_round")
    for r in rounds[1:][-th.window:]:  # the first step compiles: excluded
        t = _num(r.get("train_seconds"))
        if t is not None and t > th.train_s:
            alerts.append(Alert("train_s_high", "warn", f"step {r.get('local_round_id')} trained {t:.0f} s",
                                round(t, 1), th.train_s))
    if rounds:
        bad = 0
        for r in reversed(rounds):
            g = _num(r.get("grad_norm"))
            if g is None or g == 0.0 or not math.isfinite(g):
                bad += 1
            else:
                break
        if bad >= th.grad_bad_steps:
            alerts.append(Alert("grad_norm_bad", "stop", f"last {bad} step(s) with grad_norm 0 / NaN / missing "
                                f"(last {rounds[-1].get('grad_norm')})", bad, th.grad_bad_steps))
    last = rounds[-1] if rounds else None
    last_t = _num((last or {}).get("time_unix"))
    if last_t is not None and now - last_t > th.wait_samples_s:
        alerts.append(Alert("waiting_for_samples", "warn", f"no trained step for {(now - last_t) / 60:.0f} min",
                            round(now - last_t), th.wait_samples_s))
    elif last is not None and (_num(last.get("rollout_seconds")) or 0.0) > th.wait_samples_s:
        alerts.append(Alert("waiting_for_samples", "warn",
                            f"step {last.get('local_round_id')} waited {last['rollout_seconds'] / 60:.0f} min "
                            "for samples", round(last["rollout_seconds"]), th.wait_samples_s))
    fa = _of(events, "rl_verl_fa_round")[-th.window:]
    pulled = sum(int(e.get("pulled") or 0) for e in fa)
    if pulled:
        dropped = sum(int(e.get("discarded") or 0) for e in fa)
        rate = dropped / pulled
        if rate > th.drop_rate:
            alerts.append(Alert("drop_rate_high", "warn", f"{dropped}/{pulled} pulled groups discarded over "
                                f"{len(fa)} steps", round(rate, 3), th.drop_rate))
    kept = sum(int(e.get("kept") or 0) for e in fa if e.get("uniform_reward_groups") is not None)
    if kept:
        uniform = sum(int(e["uniform_reward_groups"]) for e in fa if e.get("uniform_reward_groups") is not None)
        share = (kept - uniform) / kept
        if share < th.nonzero_adv_groups:
            alerts.append(Alert("nonzero_adv_groups_low", "warn", f"{kept - uniform}/{kept} trained groups with a "
                                "nonzero advantage", round(share, 3), th.nonzero_adv_groups))
    elif rounds:  # older tapes: the round's zero-variance ratio
        ratios = [_num(r.get("zero_variance_group_ratio")) for r in rounds[-th.window:]]
        ratios = [x for x in ratios if x is not None]
        if ratios and 1.0 - sum(ratios) / len(ratios) < th.nonzero_adv_groups:
            share = 1.0 - sum(ratios) / len(ratios)
            alerts.append(Alert("nonzero_adv_groups_low", "warn", "share of groups with reward variance",
                                round(share, 3), th.nonzero_adv_groups))
    alerts += memory_alerts(_of(events, "rl_host_memory"), th)
    return alerts


def memory_alerts(records: list[dict[str, Any]], th: Thresholds) -> list[Alert]:
    alerts: list[Alert] = []
    if not records:
        return alerts
    low: dict[str, float] = {}
    for host, node in (records[-1].get("nodes") or {}).items():
        mib = _num((node or {}).get("mem_available_mib"))
        if mib is not None and mib / GIB_MIB < th.mem_available_gib:
            low[host] = round(mib / GIB_MIB, 1)
    if low:
        alerts.append(Alert("mem_available_low", "stop", f"MemAvailable below {th.mem_available_gib:g} GiB on "
                            f"{sorted(low)}", low, th.mem_available_gib))
    per_host: dict[str, list[tuple[float, float]]] = {}
    for r in records:
        rid = _num(r.get("rollout_id"))
        if rid is None or r.get("phase") not in (None, "after", "after_step"):
            continue
        for host, node in (r.get("nodes") or {}).items():
            rss = _num((node or {}).get("workers_rssanon_mib"))
            if rss is not None:
                per_host.setdefault(host, []).append((rid, rss / GIB_MIB))
    steep = {}
    for host, points in per_host.items():
        s = slope(points[-max(th.window, 3):])
        if s is not None and s > th.rssanon_slope_gib:
            steep[host] = round(s, 2)
    if steep:
        alerts.append(Alert("rssanon_slope", "stop", f"worker RssAnon grows > {th.rssanon_slope_gib:g} GiB/step on "
                            f"{sorted(steep)}", steep, th.rssanon_slope_gib))
    return alerts


def consumed_alerts(records: list[dict[str, Any]], th: Thresholds, *, group_window: int = 40) -> list[Alert]:
    """The buffer feed's own drops (over length, short group, stale incarnation) over the
    last ``group_window`` pulled groups."""
    recent = records[-group_window:]
    if not recent:
        return []
    dropped = [r for r in recent if r.get("dropped")]
    alerts = []
    rate = len(dropped) / len(recent)
    if rate > th.drop_rate:
        reasons: dict[str, int] = {}
        for r in dropped:
            reasons[str(r["dropped"])] = reasons.get(str(r["dropped"]), 0) + 1
        alerts.append(Alert("feed_drop_rate_high", "warn", f"buffer feed dropped {len(dropped)}/{len(recent)} "
                            f"groups {reasons}", round(rate, 3), th.drop_rate))
    stale = [r for r in records if r.get("dropped") == "stale_incarnation"]
    if stale:
        alerts.append(Alert("stale_incarnation_groups", "warn", f"{len(stale)} group(s) of a superseded sampler "
                            "incarnation refused (a fenced island still writes?)", len(stale)))
    return alerts


def sampler_alerts(tapes: dict[str, list[dict[str, Any]]], trainer: list[dict[str, Any]], now: float,
                   th: Thresholds) -> list[Alert]:
    alerts: list[Alert] = []
    trained = _of(trainer, "rl_local_round")
    for name, events in tapes.items():
        if not events or _of(events, "rl_sampler_leave"):
            continue  # an incarnation that left is not expected to beat
        last_t = max((_num(e.get("time_unix")) or 0.0) for e in events)
        written = [e for e in _of(events, "rl_agentic_group") if e.get("written")] \
            or [e for e in _of(events, "rl_sampler_engine_round") if e.get("groups")]
        out_t = max(((_num(e.get("time_unix")) or 0.0) for e in written), default=None)
        if now - last_t > th.island_silent_s:
            alerts.append(Alert("island_silent", "warn", f"{name}: no event for {(now - last_t) / 60:.0f} min",
                                round(now - last_t), th.island_silent_s, target=name))
        elif out_t is not None and now - out_t > th.island_silent_s:
            alerts.append(Alert("island_no_output", "warn", f"{name}: no group written for "
                                f"{(now - out_t) / 60:.0f} min", round(now - out_t), th.island_silent_s, target=name))
        lags = [int(e["version_lag"]) for e in _of(events, "rl_agentic_group")[-20:]
                if e.get("version_lag") is not None]
        if lags and max(lags) >= th.policy_lag:
            alerts.append(Alert("policy_lag", "warn", f"{name}: groups up to {max(lags)} versions behind",
                                max(lags), th.policy_lag, target=name))
        served = [int(e["policy_version"]) for e in _of(events, "rl_sampler_engine_round")
                  if e.get("policy_version") is not None]
        if served and trained:
            version = served[-1]
            # the first trained step the island has not loaded (its release, roughly) and when
            newer = [r for r in trained if (_num(r.get("local_round_id")) or 0) > version]
            if newer:
                since = _num(newer[0].get("time_unix"))
                if since is not None and now - since > th.release_delay_s:
                    alerts.append(Alert("release_delay", "warn", f"{name}: serves v{version}; v{version + 1} "
                                        f"trained {(now - since) / 60:.0f} min ago", round(now - since),
                                        th.release_delay_s, target=name))
    return alerts


def _parse_ts(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    from datetime import datetime

    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00")).timestamp()
    except ValueError:
        return None


def nebius_instances(doc: Any, run_filter: str | None = None) -> list[dict[str, Any]]:
    """``[{id, name, state, since}]`` of ``nebius compute instance list --format json``."""
    items = doc.get("items", []) if isinstance(doc, dict) else (doc or [])
    out = []
    for item in items:
        meta, status = item.get("metadata") or {}, item.get("status") or {}
        name = str(meta.get("name", ""))
        if run_filter and run_filter not in name:
            continue
        out.append({"id": meta.get("id"), "name": name, "state": str(status.get("state", "")).upper(),
                    "since": _parse_ts(meta.get("updated_at") or meta.get("created_at"))})
    return out


def vm_alerts(instances: list[dict[str, Any]], now: float, th: Thresholds) -> list[Alert]:
    alerts = []
    for vm in instances:
        if vm["state"] in STARTING_STATES and vm["since"] is not None and now - vm["since"] > th.vm_starting_s:
            alerts.append(Alert("vm_starting_stuck", "warn", f"VM {vm['name']} {vm['state']} for "
                                f"{(now - vm['since']) / 60:.0f} min (delete + recreate, never start)",
                                round(now - vm["since"]), th.vm_starting_s, target=vm["id"]))
    return alerts


def boat_alerts(doc: dict[str, Any] | None, th: Thresholds) -> list[Alert]:
    if not doc:
        return []
    alerts = []
    day = ((doc.get("starts") or {}).get("day") or {})
    used, limit = _num(day.get("used")), _num(day.get("limit"))
    if used is not None and limit:
        if used / limit >= th.boat_quota:
            alerts.append(Alert("boat_quota", "stop", f"boat starts today {used:.0f}/{limit:.0f}",
                                round(used / limit, 3), th.boat_quota))
    active, cap = _num(doc.get("activeSandboxes")), _num(doc.get("maxActiveSandboxes"))
    if active is not None and cap and active / cap >= th.boat_quota:
        alerts.append(Alert("boat_active", "warn", f"boat sandboxes active {active:.0f}/{cap:.0f}",
                            round(active / cap, 3), th.boat_quota))
    keep = doc.get("keepGoing") or {}
    spent, extra_cap = _num(keep.get("usedDollars")), _num(keep.get("capDollars"))
    if spent is not None and extra_cap and spent / extra_cap >= th.boat_quota:
        alerts.append(Alert("boat_quota", "stop", f"boat extra starts ${spent:.2f} of the ${extra_cap:.0f} cap",
                            round(spent / extra_cap, 3), th.boat_quota))
    return alerts


def spend_alerts(spent: float | None, budget: float | None, th: Thresholds) -> list[Alert]:
    if spent is None or not budget:
        return []
    share = spent / budget
    return [Alert("spend_budget", "stop", f"spent ${spent:.0f} of ${budget:.0f}", round(share, 3), th.spend)] \
        if share >= th.spend else []


# ------------------------------------------------------------------ actions
Runner = Callable[[Sequence[str], str | None], tuple[int, str]]


def run_command(argv: Sequence[str], stdin_path: str | None = None) -> tuple[int, str]:
    """The default runner: ``(exit code, output)``."""
    stdin = open(stdin_path, "rb") if stdin_path else None  # noqa: SIM115 - closed below
    try:
        done = subprocess.run(list(argv), stdin=stdin, capture_output=True, text=True, timeout=600)
        return done.returncode, (done.stdout + done.stderr)[-2000:]
    except (OSError, subprocess.TimeoutExpired) as error:
        return 127, f"{type(error).__name__}: {error}"
    finally:
        if stdin is not None:
            stdin.close()


@dataclasses.dataclass
class StopPlan:
    """How to stop the run (``--act``)."""

    run: str | None = None  # the trainer run id (kill script argument)
    trainer_host: str | None = None  # ssh host of the trainer (--trainer-host / TRAINER_HOST)
    kill_script: str | None = None  # local path of the kill script, sent over ssh stdin
    nebius: str = "nebius"
    boat_run: str | None = None  # boat_labels sweep --run
    python: str = sys.executable


def stop_actions(alerts: list[Alert], instances: list[dict[str, Any]],
                 plan: StopPlan) -> list[tuple[list[str], str | None]]:
    """The commands ``--act`` runs for ``alerts`` (``(argv, stdin path)`` pairs): a stop
    alert -> stop the trainer, delete every listed VM, sweep the run's boat sandboxes; a
    VM stuck in STARTING -> delete that VM."""
    commands: list[tuple[list[str], str | None]] = []
    deleted: set[str] = set()
    if any(a.severity == "stop" for a in alerts):
        if plan.run and plan.kill_script and plan.trainer_host:
            commands.append((["ssh", plan.trainer_host, "bash", "-s", "--", plan.run], plan.kill_script))
        for vm in instances:
            if vm.get("id") and vm["id"] not in deleted:
                deleted.add(vm["id"])
                commands.append(([plan.nebius, "compute", "instance", "delete", "--id", str(vm["id"])], None))
        if plan.boat_run:
            commands.append(([plan.python, "-m", "yeto.rl.harness.harbor.boat_labels", "sweep",
                              "--run", plan.boat_run], None))
    for a in alerts:
        if a.key == "vm_starting_stuck" and a.target and a.target not in deleted:
            deleted.add(a.target)
            commands.append(([plan.nebius, "compute", "instance", "delete", "--id", str(a.target)], None))
    return commands


# ------------------------------------------------------------------ patrol
def _json_source(path: str | None, cmd: str | None, runner: Runner) -> tuple[Any, str | None]:
    if path:
        try:
            return json.loads(Path(path).expanduser().read_text()), None
        except (OSError, ValueError) as error:
            return None, f"{path}: {error}"
    if cmd:
        code, out = runner(shlex.split(cmd), None)
        if code != 0:
            return None, f"{cmd}: exit {code}: {out[-300:]}"
        try:
            return json.loads(out[out.index("{") if "{" in out else 0:]), None
        except ValueError as error:
            return None, f"{cmd}: {error}"
    return None, None


def patrol(args: argparse.Namespace, *, now: float | None = None, runner: Runner = run_command,
           th: Thresholds | None = None) -> dict[str, Any]:
    now = time.time() if now is None else now
    th = th or Thresholds()
    trainer = read_events(args.tape)
    consumed = read_events(args.consumed)
    samplers = {Path(p).name: read_events(p) for p in args.sampler_tape or ()}
    errors: list[str] = []
    nebius_doc, err = _json_source(args.nebius_json, args.nebius_cmd, runner)
    errors += [err] if err else []
    boat_doc, err = _json_source(args.boat_json, args.boat_cmd, runner)
    errors += [err] if err else []
    instances = nebius_instances(nebius_doc, args.vm_filter) if nebius_doc is not None else []
    alerts = (trainer_alerts(trainer, now, th) + consumed_alerts(consumed, th)
              + sampler_alerts(samplers, trainer, now, th) + vm_alerts(instances, now, th)
              + boat_alerts(boat_doc, th) + spend_alerts(args.spent_usd, args.budget_usd, th))
    for a in alerts:
        if a.severity == "stop" and a.key not in args.stop_on:
            a.severity = "warn"
    report: dict[str, Any] = {
        "time_unix": round(now, 1), "alerts": [a.to_dict() for a in alerts], "errors": errors,
        "steps": len(_of(trainer, "rl_local_round")), "consumed": len(consumed),
        "vms": [{k: v for k, v in vm.items() if k != "since"} for vm in instances],
        "boat_starts_day": ((boat_doc or {}).get("starts") or {}).get("day"),
    }
    commands = stop_actions(alerts, instances, StopPlan(
        run=args.run, trainer_host=args.trainer_host, kill_script=args.kill_script, nebius=args.nebius,
        boat_run=args.boat_run)) if alerts else []
    report["planned_actions"] = [shlex.join(argv) + (f" < {stdin}" if stdin else "") for argv, stdin in commands]
    if args.act:
        report["actions"] = []
        for argv, stdin in commands:
            code, out = runner(argv, stdin)
            report["actions"].append({"cmd": shlex.join(argv), "exit": code, "output": out[-500:]})
    return report


def exit_code(report: dict[str, Any]) -> int:
    severities = {a["severity"] for a in report["alerts"]}
    return 2 if "stop" in severities else 1 if severities else 0


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--tape", help="trainer event tape / log")
    p.add_argument("--consumed", help="verl-buffer-consumed-<learner>.jsonl")
    p.add_argument("--sampler-tape", action="append", help="a sampler island tape (repeatable)")
    p.add_argument("--nebius-json"), p.add_argument("--nebius-cmd",
                                                    help="e.g. 'nebius compute instance list --parent-id P "
                                                         "--format json'")
    p.add_argument("--vm-filter", help="only VMs whose name contains this (required with --act and a Nebius "
                                       "source: the stop-loss deletes every listed VM)")
    p.add_argument("--boat-json"), p.add_argument("--boat-cmd", help="e.g. 'boat limits'")
    p.add_argument("--spent-usd", type=float), p.add_argument("--budget-usd", type=float)
    p.add_argument("--stop-on", type=lambda s: frozenset(x for x in s.split(",") if x),
                   default=STOP_KEYS, help=f"alerts that stop the run (default {','.join(sorted(STOP_KEYS))})")
    p.add_argument("--act", action="store_true", help="run the stop-loss (default: report only)")
    p.add_argument("--run", help="trainer run id for the kill script")
    p.add_argument("--trainer-host", default=os.environ.get("TRAINER_HOST"),
                   help="ssh host of the trainer (default: $TRAINER_HOST); no stop of the trainer without it")
    p.add_argument("--kill-script", help="local kill script, run on the trainer host over ssh (bash -s -- RUN)")
    p.add_argument("--nebius", default="nebius", help="the nebius CLI")
    p.add_argument("--boat-run", help="boat run label to sweep (boat_labels sweep --run)")
    return p


def main(argv: Sequence[str] | None = None) -> int:
    p = parser()
    args = p.parse_args(argv)
    if args.act and (args.nebius_json or args.nebius_cmd) and not args.vm_filter:
        p.error("--act with --nebius-json/--nebius-cmd needs --vm-filter (else every VM in the project is deleted)")
    if args.boat_cmd:
        args.boat_cmd = str(Path(args.boat_cmd.split()[0]).expanduser()) + args.boat_cmd[len(args.boat_cmd.split()[0]):]
    report = patrol(args)
    print(json.dumps(report, indent=1, sort_keys=True))
    return exit_code(report)


if __name__ == "__main__":
    sys.exit(main())
