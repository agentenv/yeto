"""Cost estimate (design D7): price table x GPU count x wall clock.

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


def island_cost(isl: dict, prices: dict, now: float) -> dict:
    unit = price_per_gpu_h(prices, isl.get("cloud"), isl.get("gpu"), isl.get("price_key"))
    gpus = isl.get("gpus")
    if isl.get("ready_ts") is None:
        hours = None
    else:
        open_s = max(0.0, now - isl["open_ts"]) if isl.get("open_ts") is not None else 0.0
        hours = (isl.get("closed_s", 0.0) + open_s) / 3600.0
    rate = unit * gpus if unit is not None and isinstance(gpus, (int, float)) else None
    running = isl.get("open_ts") is not None
    return {"id": isl["id"], "unit_usd_gpu_h": unit, "rate_usd_h": rate, "hours": hours,
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
            "note": "估算：价目表 × GPU 数 × 墙钟；非账单",
            "prices_source": reducer.prices.get("_source")}
