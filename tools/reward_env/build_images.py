#!/usr/bin/env python3
"""Plan (default) or build prebaked reward-environment images on Modal (CPU only).

    # plan only: no network, no Modal; writes a JSON report
    python tools/reward_env/build_images.py --benchmark tb2 --tasks-dir ~/work/tb2-data --out plan.json

    # build (needs a Modal token; costs CPU build time; needs approval first)
    python tools/reward_env/build_images.py ... --build --app yeto-reward-env-build [--tasks a,b]

``--build`` never targets the production app ``yeto-tbench2``.  Each image is
built by Modal from ``Image.from_registry(base).run_commands(<prebake script>)``;
Modal caches layers by content, so sandboxes later created from the same
definition (``yeto.cloud.modal_reward_env.PrebakedModalSandboxBackend``) reuse the build.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from yeto.rl.harness.reward_env import benchmark as bm  # noqa: E402


def plan_report(adapter, task_ids):
    rows = []
    for tid in task_ids:
        plan = adapter.prebake_plan(tid)
        rows.append({"task_id": tid, "base_image": plan.base_image, "digest": plan.digest(),
                     "complete": plan.complete, "commands": list(plan.commands), "skipped": list(plan.skipped)})
    return {"benchmark": adapter.name, "schema": bm.PREBAKE_SCHEMA, "tasks": rows,
            "complete": sum(r["complete"] for r in rows), "total": len(rows)}


def build(adapter, task_ids, app_name):
    import modal

    from yeto.cloud.modal_reward_env import modal_image
    from yeto.rl.harness.reward_env.tb2 import check_build_app

    app = modal.App.lookup(check_build_app(app_name), create_if_missing=True)
    results = []
    for tid in task_ids:
        plan = adapter.prebake_plan(tid)
        started = time.monotonic()
        try:
            with modal.enable_output():
                modal_image(plan).build(app)
            results.append({"task_id": tid, "ok": True, "seconds": round(time.monotonic() - started, 1)})
        except Exception as exc:  # noqa: BLE001 - report and continue
            results.append({"task_id": tid, "ok": False, "error": f"{type(exc).__name__}: {exc}"[:500],
                            "seconds": round(time.monotonic() - started, 1)})
        print(json.dumps(results[-1]), flush=True)
    return results


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--benchmark", default="tb2")
    p.add_argument("--tasks-dir")
    p.add_argument("--tasks", help="comma separated subset")
    p.add_argument("--out")
    p.add_argument("--build", action="store_true")
    p.add_argument("--app")
    a = p.parse_args(argv)
    adapter = bm.get_adapter(a.benchmark, **({"tasks_dir": a.tasks_dir} if a.tasks_dir else {}))
    ids = a.tasks.split(",") if a.tasks else adapter.task_ids()
    report = plan_report(adapter, ids)
    if a.build:
        if not a.app:
            p.error("--build needs an explicit --app (never yeto-tbench2)")
        report["build"] = build(adapter, ids, a.app)
    text = json.dumps(report, indent=2, ensure_ascii=False)
    if a.out:
        Path(a.out).write_text(text)
    print(f"{report['complete']}/{report['total']} plans complete" if a.out else text)
    return 0 if all(r.get("ok", True) for r in report.get("build", [])) else 1


if __name__ == "__main__":
    raise SystemExit(main())
