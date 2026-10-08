"""Head-side ``fleet.jsonl`` writer (fleet-dashboard 2.3).

``FleetController`` reports island lifecycle edges here; every
``tick_s`` (default 5 min) a ``cost_tick`` carries each island's $/h,
accumulated wall clock and estimated $ (price table x GPUs x wall clock;
an unpriced island has ``null`` amounts, never 0). Writing is best-effort:
a failure is printed and never reaches the controller.
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Callable

from .cost import load_prices, price_per_gpu_h

_ISLAND_INDEX = re.compile(r"-l(\d+)(?:-|$)")


def island_id_of(name: str) -> int | None:
    m = _ISLAND_INDEX.search(name or "")
    return int(m.group(1)) if m else None


def meta_from_task(task: Any) -> dict:
    """cloud/region/gpu/gpus of a sky Task (defensive: any shape -> partial dict)."""
    meta: dict[str, Any] = {"cloud": None, "region": None, "gpu": None, "gpus": None}
    try:
        # Modal islands are not sky Tasks: the launcher keeps a ModalIslandConfig
        # (gpu "H100!" = exact, no upgrade; gpus_per_node x num_nodes).
        if getattr(task, "resources", None) is None and isinstance(getattr(task, "gpu", None), str) \
                and isinstance(getattr(task, "gpus_per_node", None), int):
            nodes = getattr(task, "num_nodes", 1)
            meta["cloud"] = "modal"
            meta["region"] = getattr(task, "region", None)
            meta["gpu"] = task.gpu.rstrip("!")
            meta["gpus"] = task.gpus_per_node * (nodes if isinstance(nodes, int) and nodes > 0 else 1)
            return meta
        resources = getattr(task, "resources", None)
        if resources is None:
            return meta
        res = next(iter(resources)) if isinstance(resources, (set, frozenset, list, tuple)) else resources
        cloud = getattr(res, "cloud", None)
        meta["cloud"] = str(cloud).lower() if cloud is not None else None
        meta["region"] = getattr(res, "region", None)
        acc = getattr(res, "accelerators", None)
        if isinstance(acc, dict) and acc:
            gpu, count = next(iter(acc.items()))
            meta["gpu"], meta["gpus"] = str(gpu), count
        nodes = getattr(task, "num_nodes", None)
        if isinstance(meta["gpus"], (int, float)) and isinstance(nodes, int) and nodes > 1:
            meta["gpus"] = meta["gpus"] * nodes
    except Exception:  # noqa: BLE001 - metadata is optional
        pass
    return meta


class FleetLog:
    def __init__(self, path: str | os.PathLike, islands: dict[str, dict], *, prices: dict | None = None,
                 budget_usd: float | None = None, tick_s: float = 300.0,
                 clock: Callable[[], float] = time.time):
        self.path = Path(path)
        self.meta = {name: dict(m) for name, m in islands.items()}
        self.prices = prices if prices is not None else load_prices(None)
        self.budget_usd = budget_usd
        self.tick_s = tick_s
        self.clock = clock
        self.open_since: dict[str, float] = {}
        self.closed_s: dict[str, float] = {name: 0.0 for name in islands}
        self.last_tick: float | None = None

    def _write(self, rec: dict) -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
        except OSError as exc:
            print(f"[launcher] fleet.jsonl write failed: {exc}", file=sys.stderr)

    def rate(self, name: str) -> float | None:
        m = self.meta.get(name, {})
        unit = price_per_gpu_h(self.prices, m.get("cloud"), m.get("gpu"), m.get("price_key"))
        gpus = m.get("gpus")
        return unit * gpus if unit is not None and isinstance(gpus, (int, float)) else None

    def event(self, kind: str, name: str, **fields: Any) -> dict:
        now = self.clock()
        self.meta.setdefault(name, {})
        self.closed_s.setdefault(name, 0.0)
        if kind == "island_ready":
            self.open_since.setdefault(name, now)
        elif kind in ("island_lost", "island_stop") and name in self.open_since:
            self.closed_s[name] += now - self.open_since.pop(name)
        m = self.meta[name]
        rec = {"event": kind, "time_unix": now, "island": name, "island_id": island_id_of(name),
               **{k: m.get(k) for k in ("cloud", "region", "gpu", "gpus", "price_key")}, **fields}
        self._write(rec)
        return rec

    def wall_s(self, name: str, now: float) -> float:
        return self.closed_s.get(name, 0.0) + (now - self.open_since[name] if name in self.open_since else 0.0)

    def cost_tick(self) -> dict:
        now = self.clock()
        self.last_tick = now
        rows = []
        for name in self.meta:
            rate = self.rate(name)
            wall_h = self.wall_s(name, now) / 3600.0
            rows.append({"island": name, "island_id": island_id_of(name), "rate_usd_h": rate,
                         "wall_h": wall_h, "cost_usd": rate * wall_h if rate is not None else None,
                         "running": name in self.open_since})
        priced = [r["cost_usd"] for r in rows if r["cost_usd"] is not None]
        rec = {"event": "cost_tick", "time_unix": now, "islands": rows,
               "total_usd": sum(priced) if priced else None, "budget_usd": self.budget_usd,
               "estimate": "价目表 x GPU 数 x 墙钟；非账单"}
        self._write(rec)
        return rec

    def maybe_tick(self) -> dict | None:
        now = self.clock()
        if self.last_tick is None or now - self.last_tick >= self.tick_s:
            return self.cost_tick()
        return None


def from_env(path: str | os.PathLike, tasks: dict[str, Any]) -> FleetLog:
    """Production constructor: price table/budget from YETO_DASHBOARD_PRICES /
    YETO_DASHBOARD_BUDGET_USD (both optional)."""
    budget = os.environ.get("YETO_DASHBOARD_BUDGET_USD")
    return FleetLog(path, {name: meta_from_task(t) for name, t in tasks.items()},
                    prices=load_prices(os.environ.get("YETO_DASHBOARD_PRICES") or None),
                    budget_usd=float(budget) if budget else None)
