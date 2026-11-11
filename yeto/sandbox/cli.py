"""``yeto sandbox {prepare,status,validate,teardown}`` (see :mod:`yeto.sandbox.ops`)."""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any


def add_parser(sub: Any) -> None:
    p = sub.add_parser("sandbox", help="task environments on sandbox hosts (agentic.envs): "
                       "prepare / status / validate / teardown")
    ssub = p.add_subparsers(dest="sandbox_command", metavar="{prepare,status,validate,teardown,assign}")
    ssub.required = True

    def common(q: argparse.ArgumentParser) -> None:
        q.add_argument("--envs", required=True, help="sampler spec JSON (its agentic.envs) or a standalone envs JSON")
        q.add_argument("--store", required=True,
                       help="the run store (directory or s3:// URI) that holds the readiness manifest")
        q.add_argument("--run", default=None, help="run label (overrides agentic.envs.run)")
        q.add_argument("--host", action="append", default=None, help="only this host (repeatable)")
        q.add_argument("--json", action="store_true", help="print JSON")

    pr = ssub.add_parser("prepare", help="pull/build images, bake the agent runtime, start sidecars (idempotent)")
    common(pr)
    pr.add_argument("--task", action="append", default=None, help="only this task id (repeatable)")
    pr.add_argument("--stage-dir", default=None, help="rendered per-host task directories "
                    "(default ~/.yeto/sandbox/<run>/stage)")
    st = ssub.add_parser("status", help="containers, disk, image readiness, sidecar health per host")
    common(st)
    va = ssub.add_parser("validate", help="oracle must score 1, nop 0; writes the result to the manifest")
    common(va)
    va.add_argument("--task", action="append", default=None)
    va.add_argument("--force", action="store_true", help="check tasks already validated again")
    va.add_argument("--trials-dir", default="/tmp/yeto_sandbox_trials")
    va.add_argument("--out", default=None, help="write the ids of the tasks that passed here (one per line)")
    td = ssub.add_parser("teardown", help="remove this run's labelled sidecars and trial containers (and their networks); nothing else")
    common(td)
    td.add_argument("--images", action="store_true", help="also remove images this run built")
    td.add_argument("--dry-run", action="store_true")
    asg = ssub.add_parser("assign", help="which sandbox host each sampler drives (sandbox.docker_host \"envs\") "
                          "and whether every host has room; no host is contacted")
    asg.add_argument("--envs", required=True, help="sampler spec JSON (its agentic.envs) or a standalone envs JSON")
    asg.add_argument("--run", default=None, help="run label (overrides agentic.envs.run)")
    asg.add_argument("--islands", type=int, required=True, help="sampler shards 0 .. N-1")
    asg.add_argument("--shard-base", default="smp", help="sampler id prefix (<base>-<k>; the pool's shard_base)")
    asg.add_argument("--per-sampler", type=int, default=None,
                     help="trial containers one sampler runs at once (its HARBOR_MAX_SANDBOXES)")
    asg.add_argument("--json", action="store_true", help="print JSON")


def _objects(store: str) -> Any:
    from yeto.rl.engine.store_dispense import _objects_for  # noqa: PLC0415

    return _objects_for(store)


def main(args: argparse.Namespace) -> int:
    from pathlib import Path  # noqa: PLC0415

    from . import ops  # noqa: PLC0415
    from .envs import EnvsError, load_envs  # noqa: PLC0415

    try:
        cfg = load_envs(args.envs, run=args.run)
    except (EnvsError, OSError, ValueError) as exc:
        print(f"[yeto sandbox] {exc}", file=sys.stderr)
        return 2
    cmd = args.sandbox_command
    if cmd == "assign":
        from .envs import assignment_report  # noqa: PLC0415

        try:
            report = assignment_report(cfg, [f"{args.shard_base}-{k}" for k in range(int(args.islands))],
                                       per_sampler=args.per_sampler)
        except EnvsError as exc:
            print(f"[yeto sandbox] {exc}", file=sys.stderr)
            return 2
        print(json.dumps(report, indent=1, sort_keys=True) if args.json else _brief(cmd, report))
        return 1 if any("> max_sandboxes" in w for w in report["warnings"]) else 0
    objects = _objects(args.store)
    try:
        if cmd == "prepare":
            stage = args.stage_dir or str(Path("~/.yeto/sandbox").expanduser() / cfg.run / "stage")
            report = ops.prepare(cfg, objects, hosts=args.host, ids=args.task, stage_dir=stage)
            code = 0 if report["ok"] else 1
        elif cmd == "status":
            report, code = ops.status(cfg, objects, hosts=args.host), 0
        elif cmd == "validate":
            report = ops.validate(cfg, objects, trial=ops.harbor_trial(args.trials_dir), hosts=args.host,
                                  ids=args.task, force=args.force)
            if args.out:
                Path(args.out).write_text("".join(f"{t}\n" for t in report["passed"]))
            code = 0 if all(r.get("passed") is True for r in report["results"]) else 1
        else:
            report, code = ops.teardown(cfg, objects, hosts=args.host, images=args.images, dry_run=args.dry_run), 0
    except EnvsError as exc:
        print(f"[yeto sandbox] {exc}", file=sys.stderr)
        return 2
    print(json.dumps(report, indent=1, sort_keys=True, default=str) if args.json else _brief(cmd, report))
    return code


def _brief(cmd: str, report: dict[str, Any]) -> str:
    lines = []
    if cmd == "prepare":
        for h, c in report["hosts"].items():
            facts = report["host_facts"].get(h, {})
            lines += [f"  WARNING {w}" for w in (facts.get("room") or {}).get("warnings") or []]
            images = facts.get("images") or {}
            if images:
                lines.append(f"  images: {images['images']} needed, {images['missing']} were missing "
                             f"(about {images['missing_gib_est']} GiB)")
                if images.get("warning"):
                    lines.append(f"  WARNING {images['warning']}")
            lines.append(f"{h}: prepared {c['prepared']} failed {c['failed']} bridge={facts.get('bridge_ip')} "
                         f"data_root={facts.get('data_root')}" + (f" ERROR {facts['error']}" if facts.get("error") else ""))
        lines += [f"  FAILED {f['host']}/{f['task_id']}: {f['error']}" for f in report["failed"]]
    elif cmd == "status":
        for h, r in report["hosts"].items():
            disk = r.get("disk") or {}
            free = f"{disk['free_kb'] / 2**20:.0f}G free" if disk else "disk ?"
            lines.append(f"{h}: running {r.get('containers_running')}/{r.get('max_sandboxes') or '-'} "
                         f"ours {r.get('containers_ours')} {free} images {r.get('images_ready')} "
                         f"sidecars {r.get('sidecars_running')} dispense {r.get('dispense')}"
                         + (f" ERROR {r['error']}" if r.get("error") else ""))
            running, cap = r.get("containers_running"), r.get("max_sandboxes")
            if isinstance(running, int) and cap and running >= 0.9 * cap:
                lines.append(f"  WARNING {h} is at {running}/{cap} trial containers: samplers on it wait for room "
                             "(add a host to agentic.envs.hosts for new samplers, or lower HARBOR_MAX_SANDBOXES)")
            if r.get("images_missing"):
                lines.append(f"  missing images ({len(r['images_missing'])}): {' '.join(r['images_missing'][:5])}"
                             " -> yeto sandbox prepare pulls them")
            room = r.get("room") or {}
            lines += [f"  WARNING {w}" for w in room.get("warnings") or []]
            if room.get("effective") is not None:
                lines.append(f"  room {room['effective']} trial containers ({room.get('limits')})")
            lines += [f"  unhealthy {k}" for k in r.get("sidecars_unhealthy") or []]
    elif cmd == "validate":
        for r in report["results"]:
            lines.append(f"{r['host']}/{r['task_id']}: passed={r.get('passed')} oracle={r.get('oracle')} "
                         f"nop={r.get('nop')} {r.get('reason') or ''}".rstrip())
        lines.append(f"passed: {len(report['passed'])}")
    elif cmd == "assign":
        for h, r in report["hosts"].items():
            lines.append(f"{h}: {len(r['samplers'])} samplers, containers {r['containers_needed'] or '?'}"
                         f"/{r['max_sandboxes'] or '-'}: {' '.join(r['samplers'])}")
        lines += [f"  WARNING {w}" for w in report["warnings"]]
    else:
        for h, r in report.items():
            lines.append(f"{h}: removed {len(r['containers'])} containers, {len(r['images'])} images"
                         + (" (dry run)" if r["dry_run"] else ""))
    return "\n".join(lines)
