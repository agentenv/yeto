"""``yeto rollout plan``: an offline capacity estimate of a sampler fleet (no API call).

Given the island count, the model and release sizes, the bandwidth of one island,
the trainer's step and the islands' group rate, it prints the cold start (base
pull), the release fan-out per step and how far the samplers lag behind the
trainer, the group balance (backlog or a waiting trainer), the sandbox-host
containers and hosts, and the hourly cost. Every input names where its default
comes from: ``measured`` (a K3.1 run or a bench, see :data:`SOURCES`) or
``assumption`` (set it from your own numbers).
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, fields
from typing import Any

GB = 1e9

# where each default comes from (printed next to the value)
SOURCES: dict[str, tuple[str, str]] = {
    "islands": ("input", "sampler sandboxes (Daytona MI355X x8)"),
    "base_gb": ("measured", "Kimi-K3.1-Pro MXFP4 base checkpoint, 1.56 TB"),
    "release_gb": ("measured", "128-expert K3.1 release, 207 GB (a 64-expert release was 103.3 GB)"),
    "pull_mb_s": ("measured", "base pull of one MI355X node from R2 (NC, GPU-B 2026-11-09): stable at "
                              "1.2-1.8 GB/s under a cap; range 1000-2500; Daytona's limit is unknown, measure it"),
    "release_mb_s": ("measured (key input)", "release download of one MI355X node: the K3.1 sampler fetched "
                                             "103.3 GB in 106.6 s (0.97 GB/s); range 1000-2500 under a cap"),
    "reload_s": ("assumption", "in-place expert reload of the engines after the download"),
    "store_gb_s": ("assumption", "aggregate store egress for the whole fleet; 0 = no shared cap "
                                 "(R2 publishes none; check with your account)"),
    "engine_start_s": ("assumption", "vLLM start of the base after the pull (weights from local disk)"),
    "step_s": ("assumption", "trainer step on Modal 6 x 8 B300, 128 experts (2 x 8, 64 experts measured "
                             "144-162 s)"),
    "groups_per_step": ("assumption", "groups the trainer consumes per step"),
    "groups_per_island_h": ("assumption", "groups one island completes per hour (agentic cybergym: trials of "
                                          "tens of minutes; measure it on your first island)"),
    "max_policy_age": ("assumption", "the oldest policy version the trainer still takes"),
    "trials_per_island": ("assumption", "trial containers one sampler runs at once (HARBOR_MAX_SANDBOXES)"),
    "host_max_sandboxes": ("measured", "trial containers per sandbox host (agentic.envs max_sandboxes): a "
                                       "64-core / 754 GB host ran 320 at 11 % load once its bridge was a /20; "
                                       "160 per host (8 samplers) keeps memory headroom"),
    "island_usd_h": ("assumption", "one MI355X x8 sandbox per hour (NC quote $24/h per 8-card node; "
                                   "use your Daytona spot price)"),
    "trainer_usd_h": ("measured", "Modal 6 x 8 B300: 3 x $136/h per 2 x 8 island (dashboard price table)"),
    "host_usd_h": ("assumption", "one sandbox host per hour"),
}


@dataclass
class PlanInputs:
    islands: int = 16
    base_gb: float = 1560.0
    release_gb: float = 207.0
    pull_mb_s: float = 1500.0
    release_mb_s: float = 1500.0
    reload_s: float = 300.0
    store_gb_s: float = 0.0
    engine_start_s: float = 900.0
    step_s: float = 300.0
    groups_per_step: int = 32
    groups_per_island_h: float = 6.0
    max_policy_age: int = 4
    trials_per_island: int = 20
    host_max_sandboxes: int = 160
    island_usd_h: float = 24.0
    trainer_usd_h: float = 408.0
    host_usd_h: float = 0.0


def _rate(per_island_mb_s: float, islands: int, store_gb_s: float) -> float:
    """MB/s one island gets when every island reads at once."""
    rate = per_island_mb_s
    if store_gb_s > 0:
        rate = min(rate, store_gb_s * 1000 / max(1, islands))
    return rate


def estimate(p: PlanInputs) -> dict[str, Any]:
    n = max(1, int(p.islands))
    pull = _rate(p.pull_mb_s, n, p.store_gb_s)
    cold_s = p.base_gb * 1000 / pull + p.engine_start_s
    apply = _rate(p.release_mb_s, n, p.store_gb_s)
    release_s = p.release_gb * 1000 / apply + p.reload_s
    lag_steps = release_s / p.step_s  # releases published while one is applied
    # the follower loads the newest release only: a sampler serves a policy about lag_steps behind
    age = math.ceil(lag_steps) + 1  # + the step the trainer is in
    made_h = n * p.groups_per_island_h
    used_h = p.groups_per_step * 3600 / p.step_s
    containers = n * p.trials_per_island
    hosts = math.ceil(containers / max(1, p.host_max_sandboxes))
    usd_h = n * p.island_usd_h + p.trainer_usd_h + hosts * p.host_usd_h
    warnings = []
    if age > p.max_policy_age:
        warnings.append(f"samplers serve a policy about {age} versions old (a release takes {release_s / 60:.0f} min "
                        f"to apply, a step {p.step_s / 60:.0f} min) > max_policy_age {p.max_policy_age}: groups "
                        "expire. Release every k steps (release cadence), smaller releases (fewer experts), or more "
                        "download bandwidth per island")
    if max(p.pull_mb_s, p.release_mb_s) > 3000:
        warnings.append("more than ~3 GB/s into one node: an MI355X node reading R2 at ~7 GB/s dropped off the "
                        "network after 1-2 minutes (yeto caps the base pull at 2500 MB/s); keep it lower")
    if p.release_mb_s < 1000:
        warnings.append(f"one island downloads a release at {p.release_mb_s:.0f} MB/s (< 1 GB/s): the per-step "
                        "fan-out holds back the training pace; consider delta releases or fanning the release out "
                        "inside the fleet")
    if made_h < used_h:
        warnings.append(f"the islands make {made_h:.0f} groups/h, the trainer takes {used_h:.0f}: the trainer waits "
                        f"{100 * (1 - made_h / used_h):.0f}% of the time; about {math.ceil(used_h / p.groups_per_island_h)} "
                        "islands keep it busy")
    elif made_h > 1.5 * used_h:
        warnings.append(f"the islands make {made_h:.0f} groups/h, the trainer takes {used_h:.0f}: the backlog grows "
                        f"{made_h - used_h:.0f} groups/h until max_backlog_groups holds the samplers back "
                        "(paid idle islands); fewer islands would do")
    if p.store_gb_s <= 0:
        warnings.append(f"no store cap given: {n} islands pull {n * pull / 1000:.1f} GB/s together at the cold "
                        f"start and {n * p.release_gb / 1000:.2f} TB per release; set --store-gb-s if your "
                        "account has a limit")
    return {
        "inputs": asdict(p),
        "cold_start": {"pull_mb_s_per_island": round(pull, 1), "minutes": round(cold_s / 60, 1),
                       "fleet_tb": round(n * p.base_gb / 1000, 2),
                       "usd": round(n * p.island_usd_h * cold_s / 3600, 2)},
        "release": {"fleet_tb_per_step": round(n * p.release_gb / 1000, 2),
                    "fleet_tb_per_hour": round(n * p.release_gb / 1000 * 3600 / p.step_s, 2),
                    "apply_minutes": round(release_s / 60, 1), "steps_behind": round(lag_steps, 2),
                    "policy_age": age},
        "groups": {"made_per_h": round(made_h, 1), "used_per_h": round(used_h, 1),
                   "trainer_busy": round(min(1.0, made_h / used_h), 2),
                   "backlog_growth_per_h": round(max(0.0, made_h - used_h), 1)},
        "sandbox": {"containers": containers, "hosts": hosts, "per_host": p.host_max_sandboxes},
        "cost": {"usd_per_h": round(usd_h, 2), "islands_usd_h": round(n * p.island_usd_h, 2),
                 "trainer_usd_h": p.trainer_usd_h, "hosts_usd_h": round(hosts * p.host_usd_h, 2)},
        "warnings": warnings,
    }


def format_plan(report: dict[str, Any]) -> str:
    i, c, r, g, s, k = (report[x] for x in ("inputs", "cold_start", "release", "groups", "sandbox", "cost"))
    lines = ["inputs (source):"]
    for f in fields(PlanInputs):
        src, why = SOURCES[f.name]
        lines.append(f"  {f.name:<22} {i[f.name]!s:<8} {src:<10} {why}")
    lines += [
        "estimate:",
        f"  cold start        {c['minutes']} min per island ({c['pull_mb_s_per_island']} MB/s), "
        f"{c['fleet_tb']} TB pulled by the fleet, ${c['usd']} of island time",
        f"  release per step  {r['fleet_tb_per_step']} TB fan-out ({r['fleet_tb_per_hour']} TB/h), "
        f"{r['apply_minutes']} min to apply on an island = {r['steps_behind']} steps; policy age ~{r['policy_age']}",
        f"  groups            {g['made_per_h']} made/h vs {g['used_per_h']} used/h: trainer busy "
        f"{int(100 * g['trainer_busy'])}%, backlog +{g['backlog_growth_per_h']}/h",
        f"  sandbox hosts     {s['containers']} trial containers -> {s['hosts']} hosts of {s['per_host']}",
        f"  cost              ${k['usd_per_h']}/h (islands ${k['islands_usd_h']}, trainer ${k['trainer_usd_h']}, "
        f"hosts ${k['hosts_usd_h']})",
    ]
    lines += [f"WARNING {w}" for w in report["warnings"]]
    return "\n".join(lines)


def add_args(q: Any) -> None:
    defaults = PlanInputs()
    for f in fields(PlanInputs):
        flag = "--" + f.name.replace("_", "-")
        q.add_argument(flag, type=type(getattr(defaults, f.name)), default=getattr(defaults, f.name),
                       help=f"{SOURCES[f.name][0]}: {SOURCES[f.name][1]} (default %(default)s)")
    q.add_argument("--json", action="store_true", help="print JSON")


def main(args: Any) -> int:
    p = PlanInputs(**{f.name: getattr(args, f.name) for f in fields(PlanInputs)})
    report = estimate(p)
    print(json.dumps(report, indent=1) if args.json else format_plan(report))
    return 0
