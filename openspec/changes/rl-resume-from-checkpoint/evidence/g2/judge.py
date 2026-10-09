"""S17 G2 judge: reads every tape of the G2 runs and checks the prelaunch criteria.
usage: judge.py [out.json]   (reads s1-runs/s17-g2-{a,a2,b1,b2,c})"""
import glob, json, os, sys
B = "/home/michael/work/s1-runs"


def tapes(run):
    files = sorted(glob.glob(f"{B}/{run}/tape-direct/**/rl-island-0*.jsonl", recursive=True))
    if not files:
        files = sorted(glob.glob(f"{B}/{run}/tape-*.jsonl"))
    out = []
    for f in files:
        for line in open(f):
            try:
                r = json.loads(line)
            except ValueError:
                continue
            r["_file"] = os.path.relpath(f, B)
            out.append(r)
    return out


def rounds(recs):
    """rid -> list of {lrs, ids, reward, loss, grad_norm, file} in tape order. An
    rl_local_round follows its rl_round_trained in the same file (its local_round_id
    counts rounds of that launch, not rollout ids)."""
    out = {}
    pending = {}
    for r in recs:
        if r.get("event") == "rl_round_trained":
            row = {"lrs": r.get("applied_lrs"), "ids": r.get("trained_sample_ids_sha256"),
                   "reward": None, "loss": None, "grad_norm": None, "file": r["_file"],
                   "time_unix": r.get("time_unix")}
            out.setdefault(r["rollout_id"], []).append(row)
            pending[r["_file"]] = row
        elif r.get("event") == "rl_local_round" and pending.get(r["_file"]) is not None:
            row = pending.pop(r["_file"])
            row.update(reward=r.get("reward_mean"), loss=r.get("loss"), grad_norm=r.get("grad_norm"),
                       local_round_id=r.get("local_round_id"))
    return out


def ev(recs, name):
    return [r for r in recs if r.get("event") == name]


res = {}
runs = {n: tapes(f"s17-g2-{n}") for n in ("a", "a2", "b1r", "b2", "c", "c2", "c3") if glob.glob(f"{B}/s17-g2-{n}")}
R = {n: rounds(t) for n, t in runs.items()}
for n, t in runs.items():
    res[n] = {"events": len(t), "rounds": {k: v for k, v in sorted(R[n].items())},
              "cuts": [{k: c.get(k) for k in ("rollout_id", "cut_id", "cut_bytes", "save_s", "store_copy_s",
                                               "store_commit_s", "total_s", "write_bytes_per_s", "policy_hash",
                                               "store_synced", "latest_seq", "pruned", "final",
                                               "trainer_onloaded_for_cut")} for c in ev(t, "rl_cut_saved")],
              "cut_failures": [c for c in ev(t, "rl_round_cut") if not c.get("ok")],
              "resumes": [{k: x.get(k) for k in ("rollout_id", "cut_id", "incarnation", "policy_hash", "checks",
                                                  "config_diff", "restore_s", "cut_bytes", "state_restore",
                                                  "lr_at_next_round")} for x in ev(t, "rl_resume")],
              "resume_checks": ev(t, "rl_resume_check"),
              "publications": sorted({x.get("policy_version") for x in ev(t, "rl_publication")} - {None})}


def first(n, rid):
    v = R[n].get(rid) or []
    return v[-1] if v else None  # the surviving (last) training of that round


crit = {}
# 1 restore hash = saved hash
saved = {c["cut_id"]: c["policy_hash"] for n in runs for c in res[n]["cuts"]}
# a cut written just before a container was stopped may miss the mirrored tape: take the
# saving container's own rl_round_cut lines from the launcher log, and the store pointers
for n in runs:
    for line in open(f"{B}/s17-g2-{n}/launch.ts.log", errors="replace"):
        if "YETO_RL_EVENT" in line and '"rl_round_cut"' in line:
            try:
                r = json.loads(line[line.index("{"):])
            except ValueError:
                continue
            if r.get("ok") and r.get("cut_id"):
                saved.setdefault(r["cut_id"], r.get("policy_hash"))
store_pointers = {}
for f in glob.glob(f"{B}/s17-g2-*/store/**/round-cut.json", recursive=True):
    p = json.load(open(f))
    store_pointers.setdefault(p["cut_id"], p.get("policy_hash"))
res["store_pointers"] = store_pointers
for n in [m for m in ("b2", "c", "c2", "c3") if m in res]:
    rs = res[n]["resumes"]
    crit[f"1_restore_hash_{n}"] = [{"cut": x["cut_id"], "restored": x["policy_hash"], "saved": saved.get(x["cut_id"]),
                                   "same": x["policy_hash"] == saved.get(x["cut_id"]),
                                   "store_pointer": store_pointers.get(x["cut_id"]), "checks": x["checks"]} for x in rs]
    crit[f"1_lr_check_{n}"] = [{k: x.get(k) for k in ("rollout_id", "expected", "applied", "ok")} for x in res[n]["resume_checks"]]
# 3 lr + ids bitwise vs A
PAIRS = [(n, r) for n, r in (("b2", range(3, 6)), ("c2", range(4, 6)), ("c3", range(2, 6))) if n in res]
for n, rids in PAIRS:
    rows = []
    for rid in rids:
        a, x = first("a", rid), first(n, rid)
        rows.append({"rid": rid, "lr_same": bool(a and x and a["lrs"] == x["lrs"]),
                     "ids_same": bool(a and x and a["ids"] == x["ids"]),
                     "a": a and {k: a[k] for k in ("lrs", "ids")}, n: x and {k: x[k] for k in ("lrs", "ids")}})
    crit[f"3_lr_ids_vs_a_{n}"] = rows
# 4 metrics vs noise
noise = []
for rid in range(6):
    a, a2 = first("a", rid), first("a2", rid)
    row = {"rid": rid}
    for m in ("reward", "loss", "grad_norm"):
        row[m] = None if not (a and a2) or a[m] is None or a2[m] is None else abs(a[m] - a2[m])
    row["bitwise_a_a2"] = bool(a and a2 and all(a[m] == a2[m] for m in ("reward", "loss", "grad_norm")))
    noise.append(row)
crit["4_noise_a_vs_a2"] = noise
for n, rids in PAIRS:
    rows = []
    for rid in rids:
        a, x = first("a", rid), first(n, rid)
        nz = noise[rid]
        row = {"rid": rid}
        for m in ("reward", "loss", "grad_norm"):
            d = None if not (a and x) or a[m] is None or x[m] is None else abs(a[m] - x[m])
            row[m] = {"diff_vs_a": d, "noise": nz[m], "within": None if d is None or nz[m] is None else d <= nz[m]}
        rows.append(row)
    crit[f"4_metrics_{n}"] = rows
res["criteria"] = crit
out = sys.argv[1] if len(sys.argv) > 1 else f"{B}/s17-g2-judge.json"
json.dump(res, open(out, "w"), indent=1, sort_keys=True, default=str)
print(json.dumps(crit, indent=1, default=str)[:6000])
