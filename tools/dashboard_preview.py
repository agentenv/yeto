#!/usr/bin/env python3
"""yeto-fleet-dashboard 9.1: static preview page built from a REAL run directory.

Only real tape / run data goes in (missing -> null -> "无"). Sources:
  * learner tape  rl-island-*.jsonl     -> phases (rl_timeline_span), reward quantiles, publications, events
  * modal-hostmem-rank*.jsonl           -> per-node GPU memory (rollout node included)
  * runs/<run>/{fleet.jsonl,meta.json}  -> cost (GPU+CPU+memory, tasks 8.2) and run name (8.3) via the Reducer
  * <run dir>/metrics.json (optional)   -> per-round truncation / logprob diff / response length parsed
                                           from launch.log by the s16 metrics script
usage: python tools/dashboard_preview.py <s1-runs run dir> -o out.html
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from yeto.dashboard.reducer import Reducer  # noqa: E402
from yeto.dashboard.sources import TapeSource, discover  # noqa: E402

TEMPLATE = Path(__file__).with_name("dashboard_preview.html")
PHASES = {"generate": "R", "train": "T", "outer_sync": "S", "publish": "P"}


def _jsonl(p: Path) -> list[dict]:
    out = []
    for line in p.read_text(errors="replace").splitlines():
        try:
            out.append(json.loads(line))
        except ValueError:
            pass
    return out


def build(run_dir: Path) -> dict:
    paths = [str(p) for p in (run_dir / "runs").glob("*") if p.is_dir()]
    paths += [str(p) for p in (run_dir / "tape-direct").glob("*") if p.is_dir()]
    r = Reducer(budget_usd=None)
    for p in discover(paths, max_depth=3):
        s = TapeSource(p)
        s.pump(r)
    ov = r.overview()
    learner = next(iter(sorted((run_dir / "tape-direct").glob("**/rl-island-*.jsonl"))), None)
    ev = _jsonl(learner) if learner else []
    start = next((e for e in ev if e.get("event") == "rl_driver_start"), None)
    t_unix0 = next((e["time_unix"] for e in ev if e.get("event") == "rl_engine_selected"), None)
    spans = [e for e in ev if e.get("event") == "rl_timeline_span"]
    # span start/end are monotonic seconds of the driver; anchor to wall clock via the record's time_unix (= end)
    rounds: dict[int, dict] = {}
    pub_spans = []
    for sp in spans:
        off = sp["time_unix"] - sp["end"]
        a, b = sp["start"] + off, sp["end"] + off
        if sp["task"] == "publish":
            pub_spans.append((a, b))
            continue
        rid = sp.get("rollout_id")
        if rid is None or sp["task"] not in PHASES:
            continue
        rounds.setdefault(rid, {"round": rid, "phases": {}})["phases"][PHASES[sp["task"]]] = [a, b]
    # the publish after round k's sync publishes policy v(k+1): attach it to round k
    for k, row in rounds.items():
        s_end = (row["phases"].get("S") or [None, None])[1]
        nxt = [p for p in pub_spans if s_end is not None and p[0] >= s_end - 1]
        if nxt:
            row["phases"]["P"] = list(min(nxt))
    trained = {e["rollout_id"]: e for e in ev if e.get("event") == "rl_round_trained"}
    local = {e.get("local_round_id", 0) - 1: e for e in ev if e.get("event") == "rl_local_round"}
    pubs = {e.get("policy_version"): e for e in ev if e.get("event") == "rl_publication"}
    metrics = {}
    mpath = run_dir / "metrics.json"
    if mpath.is_file():
        metrics = {x["round"]: x for x in json.loads(mpath.read_text()).get("rounds", [])}
    out_rounds = []
    for k in sorted(rounds):
        row, tr, lo, m = rounds[k], trained.get(k, {}), local.get(k, {}), metrics.get(k, {})
        dur = {p: round(v[1] - v[0], 1) for p, v in row["phases"].items()}
        out_rounds.append({
            "round": k, "policy_version": k, "published_version": k + 1 if (k + 1) in pubs else None,
            "reward": lo.get("reward_mean"), "reward_std": lo.get("reward_std"),
            "p10": tr.get("reward_p10"), "p50": tr.get("reward_p50"), "p90": tr.get("reward_p90"),
            "trunc": m.get("truncated"), "logprob_diff": m.get("logprob_diff"),
            "resp_mean": tr.get("resp_len_mean"), "resp_p95": tr.get("resp_len_p95"),
            "gen_tok_s": m.get("gen_tok_s"), "train_tok_s": m.get("train_tok_s"),
            "grad_norm": lo.get("grad_norm"), "phases": row["phases"], "dur": dur,
        })
    # per-node GPU memory/util over time (hostmem), downsampled to <= 400 points per node
    nodes = {}
    for hp in sorted((run_dir / "tape-direct").glob("**/modal-hostmem-rank*.jsonl")):
        rank = hp.parent.name.replace("rank", "")
        rows = [x for x in _jsonl(hp) if x.get("gpu_mem_used_mib")]
        step = max(1, len(rows) // 400)
        nodes[rank] = [[x["time_unix"], round(sum(x["gpu_mem_used_mib"]) / len(x["gpu_mem_used_mib"]) / 1024, 1),
                        (round(sum(u for u in x["gpu_util_pct"] if u is not None) / len(x["gpu_util_pct"]), 1)
                         if x.get("gpu_util_pct") else None)] for x in rows[::step]]
    keep = {"rl_engine_selected", "rl_driver_start", "rl_driver_phase", "rl_publication", "rl_round_trained",
            "rl_reconfiguration", "rl_cell_snapshot", "rl_elastic_recommendation"}
    events = [{"ts": e.get("time_unix"), "type": e["event"],
               "text": " ".join(f"{k}={e[k]}" for k in ("phase", "rollout_id", "policy_version", "recommendation",
                                                          "result", "execution_mode") if e.get(k) is not None)}
              for e in ev if e.get("event") in keep]
    meta = {}
    for m in (run_dir / "runs").glob("*/meta.json"):
        meta = json.loads(m.read_text())
    a = meta.get("args", {})
    return {
        "generated_from": str(run_dir), "overview": {k: ov[k] for k in ("run", "run_inferred", "run_kind",
                                                                         "global_status", "cost", "islands",
                                                                         "alerts", "first_ts", "data_ts")},
        "t0": t_unix0, "driver_start": start and start["time_unix"], "rounds": out_rounds, "nodes": nodes,
        "node_roles": {"0": "训练（TP2 PP4 EP2）+ 驱动", "1": "推理（SGLang TP8）"},
        "events": events,
        "config": {"model": a.get("model"), "gpu": a.get("gpu"), "placement": a.get("rl_placement"),
                   "image": a.get("rl_image"), "budget": a.get("budget")},
        "commands": [
            ["本地看板", f"yeto dashboard serve --tapes {run_dir}/runs/{ov['run']} {run_dir}/tape-direct --port 8787"],
            ["运行状态", f"yeto status {ov['run']}"],
            ["Modal 日志", f"modal app logs yeto-{ov['run']}"],
            ["停止应用", f"modal app stop --yes yeto-{ov['run']}"],
        ],
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("-o", "--out", required=True)
    ns = ap.parse_args()
    data = build(Path(ns.run_dir))
    html = TEMPLATE.read_text(encoding="utf-8").replace(
        "/*__DATA__*/null", json.dumps(data, ensure_ascii=False, default=str).replace("</", "<\\/"))
    Path(ns.out).parent.mkdir(parents=True, exist_ok=True)
    Path(ns.out).write_text(html, encoding="utf-8")
    print(f"preview: {len(data['rounds'])} rounds, nodes {sorted(data['nodes'])} -> {ns.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
