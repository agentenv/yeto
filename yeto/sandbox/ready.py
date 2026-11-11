"""The readiness manifest (``sandbox/ready.json`` in the run store) and its view.

Written by ``yeto sandbox prepare`` / ``validate`` (read, merge, write), read by
every sampler's dispenser at each ledger scan. It is a runtime filter: it is
not part of the dispenser plan, so samplers agree on the plan whatever the
manifest says and a task that becomes ready later is dispensed then::

    {"schema": "yeto.sandbox-ready/v1", "run": "<label>", "updated": <unix s>,
     "hosts": {"<host>": {"docker_host": "...", "bridge_ip": "...", "data_root": "...", "probed_at": t}},
     "tasks": {"<task id>": {"portable": bool, "hosts": {"<host>": {
         "prepared": bool, "image": "...", "digest": "sha256:...", "sidecars": {"<name>": {...}},
         "prepared_at": t, "error": null | "...",
         "validation": null | {"passed": true | false | null, "oracle": r, "nop": r, "reason": "...", "at": t}}}}}}

``validation.passed``: true (oracle 1, nop 0), false (the task is broken: the
dispenser settles it ``dropped`` with the reason), null (the check itself
failed on infrastructure: retried by the next ``validate``).
"""

from __future__ import annotations

import json
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

SCHEMA = "yeto.sandbox-ready/v1"


def empty(run: str) -> dict[str, Any]:
    return {"schema": SCHEMA, "run": run, "updated": None, "hosts": {}, "tasks": {}}


def read_manifest(objects: Any, key: str) -> dict[str, Any] | None:
    raw = objects.read(key)
    if not raw:
        return None
    try:
        doc = json.loads(raw)
    except ValueError:
        return None
    return doc if isinstance(doc, dict) and doc.get("schema") == SCHEMA else None


def merge_write(objects: Any, key: str, run: str, *, hosts: Mapping[str, Any] | None = None,
                tasks: Mapping[str, Mapping[str, Any]] | None = None) -> dict[str, Any]:
    """Read, merge (host facts replace; per-task per-host entries update field-wise), write."""
    doc = read_manifest(objects, key) or empty(run)
    doc["run"] = run
    for name, facts in (hosts or {}).items():
        doc["hosts"][name] = dict(facts)
    for tid, entry in (tasks or {}).items():
        cur = doc["tasks"].setdefault(tid, {"hosts": {}})
        for k, v in entry.items():
            if k != "hosts":
                cur[k] = v
        for h, fields in (entry.get("hosts") or {}).items():
            cur["hosts"].setdefault(h, {}).update(fields)
    doc["updated"] = time.time()
    objects.write(key, json.dumps(doc, sort_keys=True, indent=1).encode())
    return doc


@dataclass
class ReadyView:
    """Which tasks a sampler on ``host`` may take (module doc)."""

    doc: Mapping[str, Any] | None
    host: str | None = None
    require_validation: bool = True
    affinity: str = "strict"
    _cache: dict[str, tuple[str, str | None, bool]] = field(default_factory=dict)

    def _host_ready(self, entry: Mapping[str, Any]) -> bool:
        if not entry.get("prepared") or entry.get("error"):
            return False
        if not self.require_validation:
            return True
        return bool((entry.get("validation") or {}).get("passed") is True)

    def classify(self, task_id: str | None) -> tuple[str, str | None, bool]:
        """``(state, reason, own_host)``; state ``ready`` / ``pending`` / ``failed``."""
        if task_id is None:
            return "pending", None, False
        hit = self._cache.get(task_id)
        if hit is not None:
            return hit
        task = ((self.doc or {}).get("tasks") or {}).get(task_id)
        result: tuple[str, str | None, bool]
        if not task:
            result = ("pending", None, False)
        else:
            hosts = task.get("hosts") or {}
            vals = [(h, e.get("validation") or {}) for h, e in hosts.items()]
            if any(v.get("passed") is True for _, v in vals):
                broken = None
            else:
                fails = [f"{h}:{v.get('reason') or 'failed'}" for h, v in vals if v.get("passed") is False]
                broken = "validate_failed: " + "; ".join(sorted(fails)) if fails else None
            if broken:
                result = ("failed", broken, False)
            else:
                ready_on = {h for h, e in hosts.items() if self._host_ready(e)}
                own = self.host is not None and self.host in ready_on
                if self.host is None:
                    ok = bool(ready_on)
                elif self.affinity == "strict" or not task.get("portable"):
                    ok = own
                else:
                    ok = bool(ready_on)
                result = ("ready" if ok else "pending", None, own)
        self._cache[task_id] = result
        return result
