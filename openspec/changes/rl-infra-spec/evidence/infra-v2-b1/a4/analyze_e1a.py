"""E1-A (3.4 X2) + E1-E criteria (evidence/infra-e1/plan.md, 8-card original wording).

Usage: analyze_e1a.py <baseline dir> <switch dir>
Each dir: rl-island-0.jsonl (tape), gpu.txt (index,uuid,name,driver), compute-apps.txt
(timestamped nvidia-smi compute-apps samples), elastic-state/ (reconfig journal/epochs, ledger,
inbox), start_utc.txt/end_utc.txt.
"""
import json
import sys
from pathlib import Path


def tape(d):
    return [json.loads(l) for l in (Path(d) / "rl-island-0.jsonl").read_text().splitlines() if l.strip()]


def journal(d):
    p = Path(d) / "elastic-state/reconfig/journal.jsonl"
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]


def rounds(ev):
    return {e["rollout_id"]: e for e in ev if e["event"] == "rl_round_trained"}


def gpus(d):
    out = {}
    for l in (Path(d) / "gpu.txt").read_text().splitlines():
        parts = [x.strip() for x in l.split(",")]
        if len(parts) >= 2:
            out[int(parts[0])] = parts[1]
    return out


def samples(d):
    cur, out = None, []
    p = Path(d) / "compute-apps.txt"
    if not p.exists():
        return out
    for l in p.read_text().splitlines():
        if l[:4].isdigit() and "T" in l[:12]:
            cur = {"t": l.strip(), "apps": []}
            out.append(cur)
        elif cur is not None and "," in l:
            uuid, pid = [x.strip() for x in l.split(",")[:2]]
            cur["apps"].append((uuid, pid))
    return out


def main(base, sw):
    res = {}
    j = journal(sw)
    phases = [r for r in j if r["kind"] == "phase"]
    reqs = [r for r in j if r["kind"] == "request"]
    term = {r["request_id"]: r["phase"] for r in phases if r["phase"] in
            ("SUCCEEDED", "REBUILT_OLD", "CANCELLED", "RECOVERY_REQUIRED")}
    epochs = json.loads((Path(sw) / "elastic-state/reconfig/epochs.json").read_text())
    committed_epochs = [r.get("config_epoch") for r in phases if r["phase"] == "COMMITTED"]
    res["a"] = {"terminal": term, "final_epoch": epochs.get("config_epoch"),
                "requests": [(r["request_id"], r["body"]) for r in reqs],
                "pass": len(term) == 2 and all(v == "SUCCEEDED" for v in term.values())
                and epochs.get("config_epoch") == 2}
    eb, es = tape(base), tape(sw)
    rb, rs = rounds(eb), rounds(es)
    diff = [r for r in range(12) if r not in rb or r not in rs
            or rb[r]["trained_sample_ids_sha256"] != rs[r]["trained_sample_ids_sha256"]]
    steps_ok = all(len(x.get("applied_lrs") or []) == 1 for x in list(rb.values()) + list(rs.values()))
    res["b"] = {"rounds_base": len(rb), "rounds_switch": len(rs), "differ": diff,
                "one_step_each": steps_ok,
                "pass": len(rb) == 12 and len(rs) == 12 and not diff and steps_ok}
    last_pub, per_round = None, {}
    for e in es:
        if e["event"] == "rl_publication":
            last_pub = e
        if e["event"] == "rl_driver_phase" and e.get("phase") == "generate":
            per_round[e["rollout_id"]] = len((last_pub or {}).get("sync/publication_members") or [])
    want = {r: (4 if 2 <= r <= 6 else 2) for r in range(12)}
    res["c"] = {"members_per_round": per_round, "expected": want, "pass": per_round == want}
    g = gpus(sw)
    pool = set(g.values())
    trainer_uuids = {g[i] for i in (0, 1, 2, 3) if i in g}
    smp = samples(sw)
    trainer_sets = [frozenset(a for a in s["apps"] if a[0] in trainer_uuids) for s in smp]
    trainer_sets = [t for t in trainer_sets if t]
    stable = len(set(trainer_sets)) == 1 if trainer_sets else None
    res["d"] = {"trainer_uuids": sorted(trainer_uuids), "samples_with_trainer": len(trainer_sets),
                "distinct_trainer_pid_sets": len(set(trainer_sets)), "pass": bool(stable)}
    outside = sorted({a[0] for s in smp for a in s["apps"]} - pool)
    res["e"] = {"pool": sorted(pool), "apps_outside_pool": outside, "samples": len(smp),
                "pass": bool(smp) and not outside}
    down = next((r["request_id"] for r in reqs if r["body"].get("target") == "T4R2S2"), None)
    seq = [(r["kind"], r.get("phase") or r.get("op")) for r in j if r.get("request_id") == down
           or (r["kind"] == "fork_op" and r.get("tx_id", "").endswith(str(down)))]
    try:
        qi = seq.index(("phase", "QUIESCING"))
        si = seq.index(("fork_op", "stop"))
        forder = qi < si
    except ValueError:
        forder = False
    res["f"] = {"down_sequence": seq, "pass": forder}
    t0 = Path(sw, "start_utc.txt").read_text().strip()
    t1 = Path(sw, "end_utc.txt").read_text().strip()
    res["g"] = {"wall": [t0, t1], "note": "GPU-hours = 8 x wall (standby G6/G7 counted)", "pass": True}
    # E1-E
    wait = [(r["request_id"], r.get("safe_point_rollout_id")) for r in phases if r["phase"] == "WAIT_SAFE"]
    status_files = sorted(str(p.name) for p in (Path(sw) / "elastic-state/inbox").glob("*.status.json"))
    res["E1-E"] = {"wait_safe_boundaries": wait, "inbox_status_files": status_files}
    print(json.dumps(res, indent=1, default=str))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
