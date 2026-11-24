"""``yeto island {status,stop,start}``: one trainer island of a fleet, by its record in the run store.

The records (``fleet/trainers/<k>/state.json``) and the per-island requests
(``fleet/trainers/<k>/STOP`` and ``START``) are those of :mod:`yeto.launch.island_lifecycle`;
``k`` is the island's learner id (its position in ``--gpu``). The run store is named by
``--store`` (an ``s3://`` / ``r2://`` / ``oss://`` URI, or a local directory) or by a
``yeto run up`` config (``--config``: its ``store`` and its ``env`` for the credentials).

* ``status``: every island's state, source and machine, incarnation, relaunches, last
  decision and any pending STOP / START;
* ``stop K``: writes island K's STOP (and drops a pending START). The launch's fleet
  controller drains island K, releases its machines and records it ``done``; the other
  islands go on. A ``yeto run up`` of the run then stops island K's sampler pools too;
* ``start K``: removes island K's STOP and writes its START. The launch's fleet controller
  orders an island it stopped on its STOP again (its next incarnation). The sampler pools
  ``yeto run up`` stopped for that island stay stopped in that process: restart
  ``yeto run up`` with the island's ``fleet_island_stops`` entry removed from its state.

``--dry-run`` prints what would be written and changes nothing. Exit codes: 0 done, 1 the
store did not take the request, 2 bad input (no store, no such island).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any

from .launch import island_lifecycle as lc


def add_parser(sub: Any) -> argparse.ArgumentParser:
    p = sub.add_parser(
        "island", help="one trainer island of a fleet: status, stop, start (by its record in the run store)",
        description="Inspect, stop or start one trainer island of a fleet through its record and requests in "
                    "the run store (fleet/trainers/<k>/). See docs/CLI_REFERENCE.md.")
    group = p.add_subparsers(dest="island_command", metavar="{status,stop,start}")
    status = group.add_parser("status", help="every island's state, machine, incarnation and pending requests")
    _store_args(status)
    status.add_argument("--json", action="store_true", help="print the records as one JSON object")
    for verb, help_text in (("stop", "stop island K only (its STOP); the other islands go on"),
                            ("start", "order island K again after its STOP (its START)")):
        q = group.add_parser(verb, help=help_text)
        q.add_argument("island", type=int, help="the island's learner id k (its position in --gpu)")
        _store_args(q)
        q.add_argument("--reason", default=None, help="recorded with the request")
        q.add_argument("--dry-run", action="store_true", help="print what would be written; change nothing")
    return p


def _store_args(q: argparse.ArgumentParser) -> None:
    src = q.add_mutually_exclusive_group(required=True)
    src.add_argument("--store", help="the run store: s3:// / r2:// / oss:// URI or a local directory")
    src.add_argument("--config", help="a `yeto run up` config: its store and env")


def _open(args: Any) -> tuple[lc.IslandStateStore | None, str]:
    environ = dict(os.environ)
    store = getattr(args, "store", None)
    if getattr(args, "config", None):
        with open(os.path.expanduser(args.config), encoding="utf-8") as fh:
            cfg = json.load(fh)
        store = cfg.get("store")
        environ.update({k: os.path.expanduser(str(v)) for k, v in (cfg.get("env") or {}).items()})
    if not store:
        return None, "no run store (--store, or a run config with a store)"
    states = lc.open_state_store(str(store), environ, log=lambda m: print(m, file=sys.stderr))
    if states is None:
        return None, f"the store {store} has no object API here (s3:// / r2:// / oss:// or a local directory)"
    return states, str(store)


def _age(at: Any) -> str:
    try:
        return f"{max(0.0, time.time() - float(at)) / 60:.0f} min ago"
    except (TypeError, ValueError):
        return "-"


def cmd_status(args: Any, states: lc.IslandStateStore, store: str) -> int:
    records = states.read_all()
    view = {}
    for k in sorted(records):
        view[k] = dict(records[k], stop_request=states.stop_requested(k), start_request=states.start_requested(k))
    if getattr(args, "json", False):
        print(json.dumps({str(k): v for k, v in view.items()}, indent=1, sort_keys=True, default=str))
        return 0
    if not view:
        print(f"no island records in {store} (fleet/trainers/<k>/state.json): a launch with the island lifecycle "
              "off keeps none")
        return 0
    print(f"islands of {store}:")
    for k, r in view.items():
        decision = r.get("decision") or {}
        handle = r.get("handle") or {}
        machine = handle.get("call_id") or handle.get("job_id") or handle.get("incarnation")
        line = (f"  island {k}: {r.get('state')} ({_age(r.get('since'))})  {r.get('name')} on {r.get('source')}"
                + (f" [{machine}]" if machine is not None else "")
                + f"  incarnation {r.get('incarnation', 0)}, relaunches {r.get('relaunches', 0)}")
        if decision.get("cause"):
            line += f"  last decision: {decision['cause']} -> {decision.get('action')}"
        if r.get("stop_request"):
            line += f"  STOP pending ({r['stop_request'].get('reason')}, by {r['stop_request'].get('by')})"
        if r.get("start_request"):
            line += f"  START pending ({r['start_request'].get('reason')}, by {r['start_request'].get('by')})"
        print(line)
    return 0


def cmd_request(args: Any, states: lc.IslandStateStore, store: str) -> int:
    k, verb = int(args.island), args.island_command
    record = states.read(k)
    if record is None and states.listed() and k not in states.listed():
        print(f"[yeto] island {k}: no record in {store} (islands with a record: {states.listed()})", file=sys.stderr)
        return 2
    reason = args.reason or f"yeto island {verb}"
    key = (lc.stop_key if verb == "stop" else lc.start_key)(k)
    if args.dry_run:
        print(f"[yeto] dry run: would write {store.rstrip('/')}/{key} {{reason: {reason!r}, by: 'yeto island {verb}'}}"
              + (f" and remove {lc.start_key(k)}" if verb == "stop" else f" after removing {lc.stop_key(k)}")
              + f"; island {k} is {record.get('state') if record else 'not recorded yet'}")
        print("[yeto] " + ("the launch's fleet controller then drains and releases this island only; a `yeto run up` "
                           "of the run stops its sampler pools" if verb == "stop" else
                           "the launch's fleet controller then orders this island again (its next incarnation)"))
        return 0
    if verb == "stop":
        states.clear_start(k)
        ok = states.request_stop(k, reason, by="yeto island stop")
    else:
        ok = states.clear_stop(k) and states.request_start(k, reason, by="yeto island start")
    if not ok:
        print(f"[yeto] island {k}: the store {store} did not take the {verb.upper()} request", file=sys.stderr)
        return 1
    print(f"[yeto] island {k}: {verb.upper()} written ({store.rstrip('/')}/{key})")
    return 0


def main(args: Any) -> int:
    verb = getattr(args, "island_command", None)
    if verb not in ("status", "stop", "start"):
        print("usage: yeto island {status,stop,start} ...", file=sys.stderr)
        return 2
    try:
        states, store = _open(args)
    except (OSError, ValueError) as error:
        print(f"[yeto] island: {error}", file=sys.stderr)
        return 2
    if states is None:
        print(f"[yeto] island: {store}", file=sys.stderr)
        return 2
    return cmd_status(args, states, store) if verb == "status" else cmd_request(args, states, store)
