"""Cost estimate (design D7): price table x (GPU + CPU + memory) x wall clock.

Host CPU/memory (tasks 8.2) are priced from ``host_prices`` ({cloud:
{cpu_core_h, mem_gib_h}}) times the island's requested ``cpus`` /
``memory_gib``; when either side is unknown the row is marked "仅 GPU".

Every number here is an estimate, not a bill. An island whose price key is
not in the table is "未定价" (``None``), never $0; local islands cost $0 and
are excluded from the efficiency ranking.
"""

from __future__ import annotations

import json
from pathlib import Path

EXAMPLE_PRICES = Path(__file__).with_name("prices.example.json")
LOCAL_CLOUDS = ("local", "本地")


def load_prices(path: str | None) -> dict:
    p = Path(path) if path else EXAMPLE_PRICES
    data = json.loads(p.read_text(encoding="utf-8"))
    data["_source"] = str(p) if path else "示例价目表（内置，非账单）"
    return data


def price_per_gpu_h(prices: dict, cloud: str | None, gpu: str | None,
                    price_key: str | None = None) -> float | None:
    if cloud and cloud.lower() in LOCAL_CLOUDS:
        return 0.0
    table = {k.lower(): v for k, v in (prices.get("prices") or {}).items()}
    keys = []
    if price_key:
        keys.append(price_key.lower())
    if cloud and gpu:
        keys += [f"{cloud}:{gpu}".lower(), f"*:{gpu}".lower()]
    for k in keys:
        if k in table and table[k] is not None:
            return float(table[k])
    return None


def host_unit_prices(prices: dict, cloud: str | None) -> dict | None:
    table = {k.lower(): v for k, v in (prices.get("host_prices") or {}).items()}
    row = table.get((cloud or "").lower()) or table.get("*")
    if not isinstance(row, dict):
        return None
    cpu, mem = row.get("cpu_core_h"), row.get("mem_gib_h")
    if cpu is None or mem is None:
        return None
    return {"cpu_core_h": float(cpu), "mem_gib_h": float(mem)}


def island_rate(prices: dict, isl: dict) -> dict:
    """$/h for one island = GPU part + host part (CPU cores + memory GiB).
    ``gpu_only`` is True when the host part could not be priced."""
    unit = price_per_gpu_h(prices, isl.get("cloud"), isl.get("gpu"), isl.get("price_key"))
    gpus = isl.get("gpus")
    gpu_rate = unit * gpus if unit is not None and isinstance(gpus, (int, float)) else None
    host_rate = None
    if (isl.get("cloud") or "").lower() in LOCAL_CLOUDS:
        host_rate = 0.0
    else:
        hp = host_unit_prices(prices, isl.get("cloud"))
        cpus, mem = isl.get("cpus"), isl.get("memory_gib")
        if hp and isinstance(cpus, (int, float)) and isinstance(mem, (int, float)):
            host_rate = hp["cpu_core_h"] * cpus + hp["mem_gib_h"] * mem
    rate = None if gpu_rate is None else gpu_rate + (host_rate or 0.0)
    return {"unit_usd_gpu_h": unit, "gpu_rate_usd_h": gpu_rate, "host_rate_usd_h": host_rate,
            "rate_usd_h": rate, "gpu_only": gpu_rate is not None and host_rate is None}


def island_cost(isl: dict, prices: dict, now: float) -> dict:
    rt = island_rate(prices, isl)
    unit = rt["unit_usd_gpu_h"]
    if isl.get("ready_ts") is None:
        hours = None
    else:
        open_s = max(0.0, now - isl["open_ts"]) if isl.get("open_ts") is not None else 0.0
        hours = (isl.get("closed_s", 0.0) + open_s) / 3600.0
    rate = rt["rate_usd_h"]
    running = isl.get("open_ts") is not None
    return {"id": isl["id"], "unit_usd_gpu_h": unit, "rate_usd_h": rate, "hours": hours,
            "gpu_rate_usd_h": rt["gpu_rate_usd_h"], "host_rate_usd_h": rt["host_rate_usd_h"],
            "gpu_only": rt["gpu_only"], "cpus": isl.get("cpus"), "memory_gib": isl.get("memory_gib"),
            "cost_usd": rate * hours if rate is not None and hours is not None else None,
            "running": running, "local": (isl.get("cloud") or "").lower() in LOCAL_CLOUDS,
            "priced": unit is not None}


def cost_view(reducer, now: float) -> dict:
    rows = []
    for iid in reducer.island_ids():
        isl = reducer.islands[iid]
        if isl.get("ready_ts") is None and isl.get("cloud") is None:
            continue  # no fleet data at all for this island
        row = island_cost(isl, reducer.prices, now)
        pts = isl["points"]
        rewards = [pts[x]["reward"] for x in sorted(pts) if pts[x].get("reward") is not None]
        toks = [pts[x]["tok_s"] for x in sorted(pts) if pts[x].get("tok_s") is not None]
        tok_s = toks[-1] if toks else None
        rate = row["rate_usd_h"]
        row["usd_per_1m_tok"] = (rate / (tok_s * 3600 / 1e6)) if rate and tok_s else (
            0.0 if rate == 0 and row["priced"] else None)
        row["reward_per_usd"] = ((rewards[-1] - rewards[0]) / row["cost_usd"]
                                 if len(rewards) >= 2 and row["cost_usd"] else None)
        row["ranked"] = bool(row["priced"] and not row["local"])
        rows.append(row)
    priced = [r for r in rows if r["cost_usd"] is not None]
    total = sum(r["cost_usd"] for r in priced) if priced else None
    burn = sum(r["rate_usd_h"] for r in rows if r["running"] and r["rate_usd_h"] is not None) if rows else None
    cap = reducer.budget_usd
    pct = (total / cap * 100) if total is not None and cap else None
    hours_to_cap = ((cap - total) / burn) if cap and total is not None and burn else None
    return {"islands": rows, "total_usd": total, "burn_usd_h": burn, "budget_usd": cap,
            "budget_pct": pct, "hours_to_cap": hours_to_cap,
            "unpriced": [r["id"] for r in rows if not r["priced"]],
            "gpu_only": [r["id"] for r in rows if r.get("gpu_only")],
            "note": "估算：价目表 ×（GPU + CPU + 内存）× 墙钟；非账单"
                    + ("；部分岛缺 CPU/内存规格，仅 GPU" if any(r.get("gpu_only") for r in rows) else ""),
            "prices_source": reducer.prices.get("_source")}
