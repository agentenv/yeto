"""The run manifest and the dashboard fleet log."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

from .rl_backend import _miles_overlay
from .topology import rl_island_placement_records
from .island_tasks import island_task_plan


def _miles_overlay_manifest(args) -> dict:
    """``miles_overlay`` for the run manifest; absent (old manifests unchanged) when off."""

    overlay = _miles_overlay()
    record = overlay.overlay_record(overlay.resolve_overlay(args))
    return {"miles_overlay": record} if record else {}


def _island_tasks_manifest(args) -> dict:
    """``island_tasks`` for the run manifest (rl-per-island-tasks); absent without the flag."""

    plan = island_task_plan(args)
    if plan is None:
        return {}
    return {"island_tasks": {str(island): data.record() for island, data in sorted(plan.items())}}


def _code_identity_manifest(args) -> dict:
    """rl-identity-tiering L9.1 dual write: ``semantic_sha256`` of the prepared
    algorithm spec and ``code_pack_digest`` (transition: the source tree hash).
    ``semantic_sha256`` is absent when no spec was prepared."""

    from ..rl.engine.algorithm import AlgorithmSpec
    from ..provenance import CODE_PACK_MODE_SOURCE_TREE

    out = {"code_pack_digest": getattr(args, "source_sha256", None),
           "code_pack_mode": CODE_PACK_MODE_SOURCE_TREE}
    raw = getattr(args, "rl_algorithm_spec_json", None)
    if raw:
        out["semantic_sha256"] = AlgorithmSpec.from_dict(json.loads(raw)).semantic_sha256()
    return out


def _write_run_manifest(args) -> dict | None:
    """Record the engine pins actually used (launch.log + <run dir>/run_manifest.json)."""

    if getattr(args, "training_mode", "sft") != "rl":
        return None
    from .. import rl as _rl
    from .. import runs

    manifest = {
        "rl_engine": "ports",
        "rl_image": getattr(args, "rl_image", None),
        "miles_commit": _rl.MILES_NEXT_COMMIT,
        "sglang_commit": _rl.SGLANG_NEXT_COMMIT,
        "source_sha256": getattr(args, "source_sha256", None),
        **_code_identity_manifest(args),
        "cluster_prefix": args.cluster_prefix,
        **_miles_overlay_manifest(args),
        **_island_tasks_manifest(args),
        # launch-preflight-guards: preflight readings, negative_test, island_overrides
        **(getattr(args, "_launch_preflight_manifest", None) or {}),
        "island_placements": rl_island_placement_records(args),
        "written_unix": time.time(),
    }
    print(f"[launcher] RL image {manifest['rl_image']}, "
          f"miles {manifest['miles_commit']}, sglang {manifest['sglang_commit']}", flush=True)
    if manifest.get("miles_overlay"):
        print(f"[launcher] Miles code = {manifest['miles_overlay']['summary']} "
              "(image manifest does NOT match the running Miles code)", flush=True)
    try:
        path = runs.run_dir(args.cluster_prefix) / "run_manifest.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(manifest, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    except OSError as e:
        print(f"[launcher] could not write the run manifest: {e}", file=sys.stderr)
    return manifest


def _dashboard_fleet_log(args, tasks, results):
    """fleet-dashboard 2.3: the run's ``fleet.jsonl`` writer, or None if it
    cannot be built (never blocks a launch)."""
    try:
        from .. import runs
        from ..dashboard.fleet import from_env

        return from_env(runs.run_dir(args.cluster_prefix) / "fleet.jsonl",
                        {name: tasks[name] for name in results})
    except Exception as e:  # noqa: BLE001
        print(f"[launcher] fleet.jsonl disabled: {e}", file=sys.stderr)
        return None


def _no_sync_events_dir(args) -> Path:
    """``<run dir>/events`` (yeto run registry), for --rl-single-island-no-sync."""

    from .. import runs

    return runs.run_dir(args.cluster_prefix) / "events"
