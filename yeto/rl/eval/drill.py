"""Drill bindings for the eval island (rl-spot-cost-saving 3.2).

A loader and an attempt that do no inference and no judging: each unit waits
``drill_unit_s`` seconds and returns a deterministic outcome. They let the real
eval-island Modal function (store, unit log, reclaim handler, launcher restart)
run on real Modal containers without a model, so a reclaim drill costs only
the container time. Never use them for a real evaluation: the scores mean
nothing.

``config`` keys: ``drill_unit_s`` (default 15).
"""

from __future__ import annotations

import hashlib
import time
from pathlib import Path
from typing import Any, Mapping


class DrillLoader:
    def __init__(self, config: Mapping[str, Any]) -> None:
        self.loads: list[dict[str, Any]] = []
        self.emit = None

    def load(self, manifest: Mapping[str, Any], files_dir: Path) -> str:
        self.loads.append({"policy_version": manifest["policy_version"], "t": time.time()})
        return str(manifest["rl/policy_token"])


def loader_factory(config: Mapping[str, Any]) -> DrillLoader:
    return DrillLoader(config)


def attempt_factory(config: Mapping[str, Any]):
    unit_s = float(config.get("drill_unit_s", 15.0))

    def attempt(task: Any, trial: int, *, policy_version: int, policy_token: str) -> dict[str, Any]:
        time.sleep(unit_s)
        h = int(hashlib.sha256(f"{policy_version}/{task.task_id}/{trial}".encode()).hexdigest()[:8], 16)
        return {"reward": float(h % 2), "success": bool(h % 2), "end_reason": "completed",
                "turns": 1, "tokens": 0, "drill": True}

    return attempt
