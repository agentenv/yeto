"""Cost estimate: price table x (GPU + CPU + memory) x wall clock.

Host CPU/memory are priced from ``host_prices`` ({cloud:
{cpu_core_h, mem_gib_h}}) times the island's requested ``cpus`` /
``memory_gib``; when either side is unknown the row is marked "GPU only".

Every number here is an estimate, not a bill. An island whose price key is
not in the table is "unpriced" (``None``), never $0; local islands cost $0 and
are excluded from the efficiency ranking.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

INFERRED_RUNNING_S = 900.0  # an inferred island with an event this recent is still billed
LOCAL_CLOUDS = ("local", "\u672c\u5730")  # "local" in Chinese, as older tapes wrote it


def load_prices(path: str | None) -> dict:
    """A price file (``--prices``), else the unified price table (:mod:`yeto.cost.prices`)."""
    if not path:
        from yeto.cost import prices  # noqa: PLC0415

        data = prices.dashboard_table()
        marked = ", ".join(data["estimate_keys"])
        data["_source"] = (f"unified price table ({prices.TABLE_NAME}, built in: list prices, not billing data; "
                           f"estimate (illustrative rows): {marked})")
        return data
    p = Path(path)
    data = json.loads(p.read_text(encoding="utf-8"))
    data["_source"] = str(p)
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


def is_estimate(prices: dict, cloud: str | None, gpu: str | None, price_key: str | None = None) -> bool:
    """The price :func:`price_per_gpu_h` finds is an illustrative row of the unified price
    table (``estimate_keys``): shown marked as an estimate."""
    marked = {k.lower() for k in prices.get("estimate_keys") or ()}
    if not marked or (cloud and cloud.lower() in LOCAL_CLOUDS):
        return False
    table = {k.lower(): v for k, v in (prices.get("prices") or {}).items()}
    keys = ([price_key.lower()] if price_key else []) + ([f"{cloud}:{gpu}".lower(), f"*:{gpu}".lower()]
                                                         if cloud and gpu else [])
    hit = next((k for k in keys if table.get(k) is not None), None)
    return hit in marked


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
    running = isl.get("open_ts") is not None
    if isl.get("ready_ts") is None and isl.get("hw_inferred"):
        # no fleet.jsonl (store-only run): billed from each launch's first to last event
        spans = [(v["first_ts"], v["last_ts"]) for v in (isl.get("launches") or {}).values()
                 if v.get("first_ts") is not None and v.get("last_ts") is not None]
        if not spans and isl.get("first_ts") is not None and isl.get("last_event_ts") is not None:
            spans = [(isl["first_ts"], isl["last_event_ts"])]
        hours = sum(b - a for a, b in spans) / 3600.0 if spans else None
        running = isl.get("last_event_ts") is not None and now - isl["last_event_ts"] < INFERRED_RUNNING_S
    elif isl.get("ready_ts") is None:
        hours = None
    else:
        open_s = max(0.0, now - isl["open_ts"]) if isl.get("open_ts") is not None else 0.0
        hours = (isl.get("closed_s", 0.0) + open_s) / 3600.0
    rate = rt["rate_usd_h"]
    return {"id": isl["id"], "unit_usd_gpu_h": unit, "rate_usd_h": rate, "hours": hours,
            "gpu_rate_usd_h": rt["gpu_rate_usd_h"], "host_rate_usd_h": rt["host_rate_usd_h"],
            "gpu_only": rt["gpu_only"], "cpus": isl.get("cpus"), "memory_gib": isl.get("memory_gib"),
            "cost_usd": rate * hours if rate is not None and hours is not None else None,
            "running": running, "local": (isl.get("cloud") or "").lower() in LOCAL_CLOUDS,
            "priced": unit is not None,
            "estimate": is_estimate(prices, isl.get("cloud"), isl.get("gpu"), isl.get("price_key"))}


def nc_node_rates(pool_events) -> dict[str, float]:
    """USD per machine hour of each pool shard, as the pool's provider stated it
    (``hourly_cost``, on ``nc_elastic_member_ready``): the whole node, whatever its
    engine uses of it. The latest rate of a shard wins."""
    rates: dict[str, float] = {}
    for e in pool_events or ():
        if e.get("event") == "nc_elastic_member_ready" and e.get("shard") and e.get("usd_per_h") is not None:
            rates[str(e["shard"])] = float(e["usd_per_h"])
    return rates


def nc_lease_spans(pool_events) -> dict[str, tuple[float, float | None]]:
    """Billed span of each NC shard from the pool's events: its ticket's order to the
    ticket's release (None: still billed). A node bills from the order, long before
    and after its sampler writes events, so the sampler tape alone undercounts."""
    shard_of: dict[str, str] = {}
    spans: dict[str, list] = {}
    for e in pool_events or ():
        name, ts, ticket = e.get("event"), e.get("ts"), e.get("ticket")
        if ts is None or not ticket:
            continue
        if name == "nc_elastic_ordered" and e.get("shard"):
            shard_of[str(ticket)] = str(e["shard"])
            spans.setdefault(str(e["shard"]), [float(ts), None])
        elif name == "nc_elastic_released" and str(ticket) in shard_of:
            span = spans[shard_of[str(ticket)]]
            span[1] = float(ts) if span[1] is None else max(span[1], float(ts))
    return {k: (v[0], v[1]) for k, v in spans.items()}


def _utc_seconds(value) -> float | None:
    import datetime as _dt

    if isinstance(value, (int, float)):
        return float(value)
    if not isinstance(value, str):
        return None
    try:
        return _dt.datetime.strptime(value.strip(), "%Y-%m-%dT%H:%M:%SZ").replace(
            tzinfo=_dt.timezone.utc).timestamp()
    except ValueError:
        return None


def run_stop_marks(reducer) -> list[tuple[float, str]]:
    """When the run itself says its islands stopped, for an island whose ``fleet.jsonl`` has
    no ``island_stop`` (a breaker stop or ``yeto down`` that killed the launcher before it
    wrote one, an older tape, a ``kill -9``): the ``run up`` state's ``trainer_down`` /
    ``trainer_finished`` / ``trainer_exit_seen`` stamps and the end of its last up period
    (``yeto run down``), and the run registry's ``finished_at`` once the run left RUNNING.
    A breaker trip alone (``stop_reason``) is not one: the trainer still drains until its
    ``trainer_down``."""
    marks: list[tuple[float, str]] = []
    up = getattr(reducer, "up_state", None)
    if up:
        try:
            state = json.loads(Path(up).expanduser().read_text())
        except (OSError, ValueError):
            state = None
        if isinstance(state, dict):
            steps = state.get("steps") if isinstance(state.get("steps"), dict) else {}
            down = _utc_seconds((steps.get("trainer_down") or {}).get("at"))
            if down is not None:
                reason = (state.get("breaker") or {}).get("stop_reason") if isinstance(
                    state.get("breaker"), dict) else None
                marks.append((down, "run up trainer_down" + (f" ({reason})" if reason else "")))
            for key in ("trainer_finished", "trainer_exit_seen"):
                t = _utc_seconds(state.get(key))
                if t is not None:
                    marks.append((t, f"run up {key}"))
            ups = [p for p in state.get("ups") or () if isinstance(p, dict)]
            if ups and ups[-1].get("end") is not None:
                t = _utc_seconds(ups[-1]["end"])
                if t is not None:
                    marks.append((t, "run up down"))
    meta_of = getattr(reducer, "run_meta_fresh", None)
    meta = meta_of() if callable(meta_of) else None
    if isinstance(meta, dict) and meta.get("state") not in (None, "RUNNING"):
        t = _utc_seconds(meta.get("finished_at"))
        if t is not None:
            marks.append((t, f"run {str(meta['state']).lower()}"))
    return marks


def island_stopped_at(reducer, isl: dict, marks: list[tuple[float, str]]) -> tuple[float, str] | None:
    """Where the billed wall clock of an island still open in ``fleet.jsonl`` ends: the
    earliest stop mark after its latest ``island_ready`` (an earlier mark belongs to a
    launch before a relaunch), or our own stop of it (``stopped_by_us``), and never before
    its last event (it ran until then). None: still running."""
    ready = isl.get("last_ready_ts") or isl.get("open_ts")
    if isl.get("open_ts") is None or ready is None:
        return None
    found = [(t, src) for t, src in marks if t >= ready]
    stop = reducer.operator_stop() if hasattr(reducer, "operator_stop") else None
    if stop and stop.get("time_unix") is not None and stop["time_unix"] >= ready and reducer.stopped_by_us(isl):
        found.append((float(stop["time_unix"]), f"operator stop ({stop.get('cause') or 'stop'})"))
    if not found:
        return None
    t, src = min(found)
    last = isl.get("last_event_ts")
    return (max(t, last) if last is not None else t), src


def cost_view(reducer, now: float) -> dict:
    rows = []
    async_ = getattr(reducer, "async_", None)
    leases = nc_lease_spans(getattr(async_, "pool", None))
    node_rates = nc_node_rates(getattr(async_, "pool", None))
    marks = run_stop_marks(reducer)
    for iid in reducer.island_ids():
        isl = reducer.islands[iid]
        if isl.get("ready_ts") is None and isl.get("cloud") is None:
            continue  # no fleet data at all for this island
        row = island_cost(isl, reducer.prices, now)
        ended = island_stopped_at(reducer, isl, marks) if row["running"] and isl.get("ready_ts") is not None else None
        if ended is not None and ended[0] < now:
            # fleet.jsonl never closed it, but the run says it stopped: bill up to that stop
            row["hours"] = (isl.get("closed_s", 0.0) + max(0.0, ended[0] - isl["open_ts"])) / 3600.0
            row["running"] = False
            row["cost_usd"] = row["rate_usd_h"] * row["hours"] if row["rate_usd_h"] is not None else None
            row["billed_until"], row["billed_until_source"] = ended
        if iid in node_rates:  # the provider's own machine-hour rate beats the price table
            row.update(rate_usd_h=node_rates[iid], priced=True, gpu_only=False, rate_source="provider hourly_cost")
            if row["hours"] is not None:
                row["cost_usd"] = row["rate_usd_h"] * row["hours"]
        if (isl.get("cloud") or "").lower() == "nc" and iid in leases:
            start, end = leases[iid]
            row["hours"] = max(0.0, (end if end is not None else now) - start) / 3600.0
            row["running"] = end is None
            row["cost_usd"] = row["rate_usd_h"] * row["hours"] if row["rate_usd_h"] is not None else None
            row["billed_from"] = "nc pool events (order to release)"
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
    fleet_note = fleet_rows(reducer, rows, now)
    priced = [r for r in rows if r["cost_usd"] is not None]
    total = sum(r["cost_usd"] for r in priced) if priced else None
    burn = sum(r["rate_usd_h"] for r in rows if r["running"] and r["rate_usd_h"] is not None) if rows else None
    cap = reducer.budget_usd
    pct = (total / cap * 100) if total is not None and cap else None
    hours_to_cap = ((cap - total) / burn) if cap and total is not None and burn else None
    out = {"islands": rows, "total_usd": total, "burn_usd_h": burn, "budget_usd": cap,
           "budget_pct": pct, "hours_to_cap": hours_to_cap,
           "unpriced": [r["id"] for r in rows if not r["priced"]],
           "gpu_only": [r["id"] for r in rows if r.get("gpu_only")],
           "note": "estimate: price table x (GPU + CPU + memory) x wall clock; not billing data"
                   + ("; some islands lack CPU/memory specs, GPU only" if any(r.get("gpu_only") for r in rows) else ""),
           "prices_source": reducer.prices.get("_source")}
    if fleet_note is not None:
        out["fleet"] = fleet_note
        out["note"] += "; fleet: " + fleet_note["basis"]
    return out


def _spans_hours(spans) -> float | None:
    """Hours covered by the union of ``(start, end)`` spans; None without one."""
    spans = sorted((float(a), float(b)) for a, b in spans if a is not None and b is not None and b >= a)
    if not spans:
        return None
    total, cur = 0.0, list(spans[0])
    for a, b in spans[1:]:
        if a > cur[1]:
            total += cur[1] - cur[0]
            cur = [a, b]
        else:
            cur[1] = max(cur[1], b)
    return (total + cur[1] - cur[0]) / 3600.0


def fleet_rows(reducer, rows: list[dict], now: float) -> dict | None:
    """A fleet of async islands bills its pools, each island's trainer and the syncer
    (``breakers.fleet.fleet_spend``): pools + Σ island trainers + syncer. ``rows`` gets
    each trainer's price from the plan (``trainer_usd_per_h``: the run config's, which
    the launch bills, over the price table), one row per pool sampler (the plan pool's
    node price, else the price table's for its provider and card; hours: the union of its
    incarnations' first to last event, a lower bound) and one for the syncer (the plan's
    ``syncer.usd_per_h``; unset: by where the plan says it runs, as ``run up`` bills it
    (``breakers.fleet.syncer_rate``: the controller head VM or its own syncer VM at the price
    table's syncer row of that VM's provider); a plan without it or no plan: unpriced;
    hours: the trainers' first to last event). None (``rows`` unchanged) off a fleet."""
    from . import fleet_islands  # noqa: PLC0415

    if getattr(reducer, "async_", None) is None:
        return None
    fleet = fleet_islands.resolve(reducer)
    if fleet is None:
        return None
    by_id = {r["id"]: r for r in rows}
    asy = reducer.async_
    trainer_rows, pool_rows = [], []
    for isl in fleet["islands"]:
        tid, price = isl.get("trainer"), isl.get("trainer_usd_per_h")
        row = by_id.get(tid) if tid is not None else None
        if row is not None:
            row.update(kind="trainer", island=isl["learner"])
            if price is not None:
                row.update(rate_usd_h=price, gpu_rate_usd_h=price, host_rate_usd_h=None, priced=True, gpu_only=False,
                           rate_source="fleet plan trainer_usd_per_h (the whole trainer: GPU, CPU and memory)")
                if row["hours"] is not None:
                    row["cost_usd"] = price * row["hours"]
            trainer_rows.append(row)
        for pool in isl["pools"]:
            for sid in pool["samplers"]:
                if sid in by_id:  # already billed (an NC shard with its own lease)
                    pool_rows.append(by_id[sid])
                    by_id[sid].update(kind="pool sampler", island=isl["learner"], pool=pool["name"])
                    continue
                s = asy.sampler.get(sid) or {}
                prov = s.get("provenance") or {}
                card = prov.get("card_model")
                m = re.search(r"tp(\d+)", str(prov.get("topology") or ""))
                gpus = int(m.group(1)) if m else 1
                rate, source = pool.get("usd_per_h"), "fleet plan pool usd_per_h (one node hour)"
                if rate is None:
                    cloud, why = pool.get("provider"), ""
                    if not cloud:  # no plan: the provider is not on the tapes; the trainers' cloud
                        cloud = next((reducer.islands[r["id"]].get("cloud") for r in trainer_rows
                                      if (reducer.islands.get(r["id"]) or {}).get("cloud")), None)
                        why = " (provider not on the tapes: the trainers' cloud)"
                    unit = price_per_gpu_h(reducer.prices, cloud, card)
                    rate = None if unit is None else unit * gpus
                    source = f"price table {cloud}:{card} x {gpus}{why}"
                    if is_estimate(reducer.prices, cloud, card):
                        source += " (estimate: illustrative price, not a price list)"
                incs = s.get("incarnations") or []
                hours = _spans_hours((x.get("first_ts"), x.get("last_ts")) for x in incs)
                isl_rec = reducer.islands.get(sid) or {}
                if hours is None:
                    hours = _spans_hours([(isl_rec.get("first_ts"), isl_rec.get("last_event_ts"))])
                last = isl_rec.get("last_event_ts")
                running = last is not None and now - last < INFERRED_RUNNING_S and not s.get("ended")
                row = {"id": sid, "kind": "pool sampler", "island": isl["learner"], "pool": pool["name"],
                       "unit_usd_gpu_h": None if rate is None else rate / gpus, "rate_usd_h": rate, "hours": hours,
                       "gpu_rate_usd_h": rate, "host_rate_usd_h": None, "gpu_only": False, "cpus": None,
                       "memory_gib": None, "cost_usd": rate * hours if rate is not None and hours is not None else None,
                       "running": running, "local": False, "priced": rate is not None, "rate_source": source,
                       "billed_from": "sampler tapes (first to last event of each incarnation; boot not on the tapes)",
                       "usd_per_1m_tok": None, "reward_per_usd": None, "ranked": False}
                rows.append(row)
                pool_rows.append(row)
    tr_recs = [reducer.islands[r["id"]] for r in trainer_rows if r["id"] in reducer.islands]
    firsts = [x.get("first_ts") for x in tr_recs if x.get("first_ts") is not None]
    lasts = [x.get("last_event_ts") for x in tr_recs if x.get("last_event_ts") is not None]
    hours = (max(lasts) - min(firsts)) / 3600.0 if firsts and lasts and max(lasts) >= min(firsts) else None
    syncer = fleet.get("syncer")
    if syncer is None:  # a plan written before the syncer price, or no plan: where the syncer ran is unknown
        rate = None
        source = ("the fleet plan has no syncer price (written before it)" if fleet.get("source") == "plan"
                  else "no fleet plan: the syncer's price is unknown")
    else:
        from yeto.breakers import fleet as breakers_fleet  # noqa: PLC0415

        pl = syncer.get("placement")
        placement = (breakers_fleet.SyncerPlacement(pl["where"], pl.get("provider"))
                     if isinstance(pl, dict) and pl.get("where") in ("head", "vm", "none") else None)
        rate, source = breakers_fleet.syncer_rate(syncer.get("usd_per_h"), placement)
        if syncer.get("usd_per_h") is not None:
            source = "fleet plan syncer.usd_per_h"
        elif placement is None:
            source = "syncer.usd_per_h unset and the fleet plan does not say where the syncer runs (written before it)"
        else:
            source = "syncer.usd_per_h unset: " + source
    srow = {"id": "syncer", "kind": "syncer", "island": None, "unit_usd_gpu_h": None, "rate_usd_h": rate,
            "hours": hours, "gpu_rate_usd_h": None, "host_rate_usd_h": None, "gpu_only": False, "cpus": None,
            "memory_gib": None, "cost_usd": rate * hours if rate is not None and hours is not None else None,
            "running": any(r.get("running") for r in trainer_rows), "local": False, "priced": rate is not None,
            "rate_source": source, "billed_from": "the trainers' first to last event",
            "usd_per_1m_tok": None, "reward_per_usd": None, "ranked": False}
    rows.append(srow)

    def part(rs: list[dict]) -> float | None:
        vals = [r["cost_usd"] for r in rs if r.get("cost_usd") is not None]
        return sum(vals) if vals else None

    return {"pools_usd": part(pool_rows), "trainers_usd": part(trainer_rows), "syncer_usd": srow["cost_usd"],
            "basis": "pools (node hours on the sampler tapes) + each island's trainer + the syncer"}


def run_up_view(path: str | None, now: float) -> dict | None:
    """The ``yeto run up`` state of the run (``--up-state``): its cumulative runtime over
    every up (stopped gaps excluded), the current up's runtime and the cumulative spend
    estimate the up monitor last recorded. None without a readable state file."""
    if not path:
        return None
    import json
    from pathlib import Path

    from yeto.rl.up_segments import uptime_summary

    try:
        state = json.loads(Path(path).expanduser().read_text())
    except (OSError, ValueError):
        return None
    if not isinstance(state, dict):
        return None
    out = uptime_summary(state, now)
    degraded = state.get("degraded_pools")
    if isinstance(out, dict) and isinstance(degraded, dict) and degraded:
        out = dict(out, degraded_pools={str(k): str(v) for k, v in degraded.items()})
    return out
