"""``yeto cost --run <id>``: cost per resource and per hour, and GPU effective utilization.

Effective utilization = productive time / billed time, where

* billed time is the sum over the run's billed compute resources (Modal apps,
  NC leases, sky clusters) of their billable span; and
* productive time is, per event source (one tape = one island / sampler),
  the union of the gaps between consecutive progress events (trained round,
  sampled round, cut, release ...) that are at most ``gap_s`` apart; a longer
  gap is waiting / loading / idling, not work. Summed over sources.

It is an estimate from event tapes, not a hardware counter; the hardware
utilization samples (``gpu_util_pct``), when present, are reported next to it.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from .billing import CostSummary, ModalBilling, Rates, resource_costs
from .guard import TapeTail, record_time
from .idle import _gpu_util, classify
from .ledger import ResourceLedger

DEFAULT_GAP_S = 10 * 60
COMPUTE_KINDS = ("modal_app", "nc_lease", "sky_cluster")


def productive_seconds(records: Iterable[tuple[str, dict[str, Any]]], gap_s: float = DEFAULT_GAP_S
                       ) -> tuple[dict[str, float], list[float]]:
    by_src: dict[str, list[float]] = {}
    utils: list[float] = []
    for src, rec in records:
        kind = classify(rec)
        if kind == "progress":
            t = record_time(rec, 0.0)
            if t > 0:
                by_src.setdefault(src, []).append(t)
        elif kind == "gpu":
            u = _gpu_util(rec)
            if u is not None:
                utils.append(u)
    out = {}
    for src, ts in by_src.items():
        ts.sort()
        out[src] = sum(b - a for a, b in zip(ts, ts[1:]) if b - a <= gap_s)
    return out, utils


def build_report(run: str, *, summary: CostSummary, tape_globs: Iterable[str],
                 gap_s: float = DEFAULT_GAP_S) -> dict[str, Any]:
    recs = TapeTail(tape_globs).read()
    prod, utils = productive_seconds(recs, gap_s)
    billed = sum(i.billable_s for i in summary.items if i.kind in COMPUTE_KINDS)
    productive = sum(prod.values())
    hours: dict[str, float] = {}
    for i in summary.items:
        for h, usd in i.hours.items():
            hours[h] = hours.get(h, 0.0) + usd
    return {
        "run": run, "total_usd": round(summary.total_usd, 2),
        "unpriced": [i.key for i in summary.unpriced], "billing_error": summary.billing_error,
        "resources": [{"resource": i.key, "name": i.name, "open": i.open,
                       "billable_h": round(i.billable_s / 3600, 3),
                       "billed_usd": round(i.billed_usd, 2), "estimated_usd": round(i.estimated_usd, 2),
                       "usd": None if i.usd is None else round(i.usd, 2),
                       "rate_usd_per_hour": i.rate_usd_per_hour, "note": i.note}
                      for i in summary.items],
        "by_hour": {h: round(v, 2) for h, v in sorted(hours.items())},
        "billed_compute_h": round(billed / 3600, 3),
        "productive_h": round(productive / 3600, 3),
        "productive_by_source_h": {Path(k).name: round(v / 3600, 3) for k, v in sorted(prod.items())},
        "effective_utilization": None if billed <= 0 else round(productive / billed, 4),
        "gpu_util_pct_mean": None if not utils else round(sum(utils) / len(utils), 1),
        "gpu_util_samples": len(utils),
    }


def default_tape_globs(run: str) -> list[str]:
    from .. import runs

    d = runs.run_dir(run)
    return [str(d / "events" / "*.jsonl"), str(d / "events" / "**" / "*.jsonl"),
            str(d / "tapes" / "*.jsonl")]


def format_report(rep: dict[str, Any]) -> str:
    lines = [f"run {rep['run']}: total ${rep['total_usd']:.2f}"
             + (f"  (UNPRICED, not counted: {', '.join(rep['unpriced'])})" if rep["unpriced"] else "")]
    if rep.get("billing_error"):
        lines.append(f"  Modal billing API error: {rep['billing_error']} (estimates only)")
    lines.append(f"  {'resource':<48} {'open':<5} {'hours':>8} {'billed $':>10} {'est. $':>9} {'total $':>9}")
    for r in rep["resources"]:
        usd = "unpriced" if r["usd"] is None else f"{r['usd']:.2f}"
        lines.append(f"  {r['resource'][:48]:<48} {'yes' if r['open'] else 'no':<5} {r['billable_h']:>8.2f} "
                     f"{r['billed_usd']:>10.2f} {r['estimated_usd']:>9.2f} {usd:>9}"
                     + (f"  [{r['note']}]" if r["note"] else ""))
    lines.append("  by hour (UTC): " + ", ".join(f"{h} ${v:.2f}" for h, v in rep["by_hour"].items()))
    eff = rep["effective_utilization"]
    lines.append(f"  GPU effective utilization: "
                 f"{'n/a' if eff is None else f'{100 * eff:.1f} %'} "
                 f"(productive {rep['productive_h']:.2f} h / billed {rep['billed_compute_h']:.2f} h)"
                 + ("" if rep["gpu_util_pct_mean"] is None else
                    f"; hardware util mean {rep['gpu_util_pct_mean']} % over {rep['gpu_util_samples']} samples"))
    return "\n".join(lines)


def cmd_cost(args) -> int:
    import time

    led = ResourceLedger.for_run(args.run)
    if not led.path.exists():
        print(f"[cost] run {args.run}: no resource ledger at {led.path}")
        return 1
    billing = None if args.no_billing else ModalBilling()
    summary = resource_costs(led.resources(), time.time(), rates=Rates.from_env(args.prices),
                             billing=billing)
    globs = default_tape_globs(args.run) + list(args.tapes or [])
    rep = build_report(args.run, summary=summary, tape_globs=globs, gap_s=args.gap_minutes * 60)
    print(json.dumps(rep, indent=2, default=str) if args.json else format_report(rep))
    return 0


def cmd_cleanup(args) -> int:
    from .cleanup import cleanup_run

    return 0 if cleanup_run(args.run, dry_run=args.dry_run, include_pools=getattr(args, "include_pools", False)) \
        else 1


def add_parsers(sub) -> None:
    cost = sub.add_parser("cost", help="cost of a run per resource and per hour, and GPU effective "
                          "utilization (Modal billing API + estimates)")
    cost.add_argument("--run", required=True, help="run name (its --cluster-prefix)")
    cost.add_argument("--json", action="store_true")
    cost.add_argument("--no-billing", action="store_true",
                      help="do not call the Modal billing API (estimates only)")
    cost.add_argument("--prices", default=None, help="price table JSON (yeto.dashboard.cost format)")
    cost.add_argument("--tapes", nargs="*", help="extra event tape globs (e.g. pulled sampler tapes)")
    cost.add_argument("--gap-minutes", type=float, default=DEFAULT_GAP_S / 60,
                      help="progress events further apart than this are not productive time")
    clean = sub.add_parser("cleanup", help="release exactly the resources in a run's resource ledger "
                           "(by id; never by name pattern, never --all)")
    clean.add_argument("--run", required=True, help="run name (its --cluster-prefix)")
    clean.add_argument("--dry-run", action="store_true", help="list what would be released")
    clean.add_argument("--include-pools", action="store_true", dest="include_pools",
                       help="also release the sampler pools' machines of the run (NC leases, Daytona "
                       "sandboxes, Modal sampler calls, Sky sampler clusters; normally pool-stop's job)")
    clean.add_argument("--include-nc", action="store_true", dest="include_pools",
                       help="deprecated: the earlier name of --include-pools (every pool provider)")


def add_launch_arguments(p) -> None:
    g = p.add_argument_group("cost guard (budget breaker, idle shutdown)")
    g.add_argument("--budget-usd", type=float, default=None,
                   help="stop the run gracefully (cut, then teardown) when its cost reaches this "
                   "(default: no budget stop)")
    g.add_argument("--budget-warn-usd", type=float, default=None,
                   help="emit cost_budget_warn at this cost (default: 80%% of --budget-usd)")
    g.add_argument("--budget-stop-grace-minutes", type=float, default=20.0,
                   help="after a graceful stop, release every ledger resource by force when the "
                   "run is still up this much later (default 20)")
    g.add_argument("--idle-warn-minutes", type=float, default=15.0,
                   help="idle_warn after this long without progress / with 0%% GPU (0 disables idle "
                   "shutdown; default 15)")
    g.add_argument("--idle-stop-minutes", type=float, default=15.0,
                   help="stop the run this long after idle_warn if still idle (default 15)")
    g.add_argument("--load-grace-minutes", type=float, default=45.0,
                   help="no idle verdict for this long after the start / a loading event (weight "
                   "loading is silent; default 45)")
    g.add_argument("--max-load-minutes", type=float, default=120.0,
                   help="loading events alone cannot exempt a run longer than this since its last "
                   "progress (default 120)")
    g.add_argument("--cost-poll-seconds", type=float, default=60.0,
                   help="cost guard tick (default 60)")
    g.add_argument("--cost-prices", default=None,
                   help="price table JSON for estimates of not-yet-billed time")
    g.add_argument("--cost-tapes", nargs="*", default=None,
                   help="extra event tape globs the idle detector reads (e.g. pulled NC sampler tapes)")
    g.add_argument("--no-cost-guard", action="store_true",
                   help="run without the cost guard (the resource ledger is still written)")

