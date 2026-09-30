"""L-2.3 criteria 1-6 over the three A2 arms (S, O, OD). Usage: analyze_a2.py <dir S> <dir O> <dir OD>.

Each dir holds rl-island-0.jsonl (event tape), launch.log(.gz) and rc.txt.
"""
import ast
import gzip
import json
import re
import sys
from pathlib import Path

N_EVAL = 32


def load(d):
    d = Path(d)
    ev = [json.loads(l) for l in (d / "rl-island-0.jsonl").read_text().splitlines() if l.strip()]
    lp = d / "launch.log.gz"
    log = gzip.open(lp, "rt").read() if lp.exists() else (d / "launch.log").read_text()
    rc = (d / "rc.txt").read_text().strip()
    scores = {}
    for m in re.finditer(r"eval (\d+): (\{[^}]*\})", log):
        try:
            scores[int(m.group(1))] = ast.literal_eval(m.group(2))
        except Exception:
            pass
    return {"ev": ev, "log": log, "rc": rc, "scores": scores}


def rounds(a):
    return {e["rollout_id"]: e for e in a["ev"] if e["event"] == "rl_round_trained"}


def evals(a):
    return [e for e in a["ev"] if e["event"] == "rl_eval"]


def c3(a):
    out, pub, start = [], None, None
    for e in a["ev"]:
        k = e["event"]
        if k == "rl_publication":
            pub = e
            if start is not None:
                out.append(("publication inside overlap window", e.get("policy_version")))
        elif k == "rl_driver_phase" and e.get("phase") == "generate" and start is not None:
            out.append(("generate inside overlap window", e.get("rollout_id")))
        elif k == "rl_eval_overlap_start":
            start = {"e": e, "pub": pub}
        elif k == "rl_eval" and e.get("overlapped"):
            if start is None:
                out.append(("overlapped rl_eval without start", e.get("policy_version")))
            elif start["pub"] is None or e.get("rl/policy_token") != start["pub"].get("rl/policy_token"):
                out.append(("token != latest publication at eval start", e.get("policy_version")))
            start = None
    n = sum(1 for e in evals(a) if e.get("overlapped"))
    return {"overlapped_evals": n, "violations": out, "pass": n > 0 and not out}


def c4(a):
    ev, out = a["ev"], []
    pubs = [i for i, e in enumerate(ev) if e["event"] == "rl_publication"]
    for i in pubs:
        prev = [e for e in ev[:i] if e["event"] in ("rl_fault_injected", "rl_publication")]
        if not prev or prev[-1]["event"] != "rl_fault_injected":
            out.append(("no rl_fault_injected before publication", ev[i].get("policy_version")))
    last = None
    for e in ev:
        if e["event"] == "rl_publication":
            last = e.get("policy_version")
        if e["event"] == "rl_driver_phase" and e.get("phase") == "generate" and e.get("policy_version") != last:
            out.append(("generate policy_version != latest publication", e.get("policy_version"), last))
    seq = [e["event"] for e in ev if e["event"] == "rl_eval_overlap_start" or (e["event"] == "rl_eval" and e.get("overlapped"))]
    alt = all(s == ("rl_eval_overlap_start" if i % 2 == 0 else "rl_eval") for i, s in enumerate(seq)) and len(seq) % 2 == 0
    if not alt:
        out.append(("overlap_start / rl_eval not strictly alternating", seq))
    return {"publications": len(pubs), "violations": out, "pass": len(pubs) == 4 and not out}


def c6(a):
    spans = [e for e in a["ev"] if e["event"] == "rl_timeline_span"]
    tr = {e["rollout_id"]: e for e in spans if e.get("task") == "train"}
    pts = []
    for e in spans:
        if e.get("task") != "eval":
            continue
        t = tr.get(e.get("rollout_id"))
        inter = 0.0 if t is None else max(0.0, min(e["end"], t["end"]) - max(e["start"], t["start"]))
        pts.append({"rollout_id": e.get("rollout_id"), "eval_s": e["end"] - e["start"], "intersect_train_s": inter})
    return {"points": pts, "pass": sum(1 for p in pts if p["intersect_train_s"] > 0) >= 2}


def main(ds, do, dd):
    S, O, OD = load(ds), load(do), load(dd)
    res = {}
    c1 = {}
    for n, a in (("S", S), ("O", O), ("OD", OD)):
        c1[n] = {"rc": a["rc"], "strict_failure": any(e["event"] == "rl_strict_failure" for e in a["ev"]),
                 "OverlapGuardError": "OverlapGuardError" in a["log"]}
    res["1"] = {"arms": c1, "pass": all(v["rc"] == "rc=0" and not v["strict_failure"] and not v["OverlapGuardError"] for v in c1.values())}
    rs, ro = rounds(S), rounds(O)
    diff = [r for r in sorted(set(rs) | set(ro)) if r not in rs or r not in ro or any(
        rs[r].get(k) != ro[r].get(k) for k in ("trained_sample_ids_sha256", "trained_groups", "trained_samples"))]
    lrs = all(len(x.get("applied_lrs") or []) == 1 for a in (rs, ro) for x in a.values())
    res["2"] = {"rounds_S": len(rs), "rounds_O": len(ro), "differ": diff, "applied_lrs_len1": lrs,
                "pass": len(rs) == 3 and not diff and lrs}
    res["3"] = {"O": c3(O), "OD": c3(OD)}
    res["3"]["pass"] = res["3"]["O"]["pass"] and res["3"]["OD"]["pass"]
    res["4"] = c4(OD)
    es = {e["policy_version"]: e.get("rl/policy_token") for e in evals(S)}
    eo = {e["policy_version"]: e.get("rl/policy_token") for e in evals(O)}
    hard = bool(es) and es == eo
    sc = []
    for v in sorted(set(S["scores"]) & set(O["scores"])):
        ks = [k for k in S["scores"][v] if k.startswith("eval/") and k.count("/") == 1 and "-" not in k]
        for k in ks:
            if k in O["scores"][v]:
                sc.append({"v": v, "key": k, "S": S["scores"][v][k], "O": O["scores"][v][k],
                           "ok": abs(S["scores"][v][k] - O["scores"][v][k]) <= 2 / N_EVAL + 1e-12})
    res["5"] = {"S_points": es, "O_points": eo, "hard": hard, "scores": sc,
                "pass": hard and bool(sc) and all(x["ok"] for x in sc) and set(S["scores"]) == set(O["scores"])}
    res["6"] = c6(O)
    print(json.dumps(res, indent=1, default=str))


if __name__ == "__main__":
    main(*sys.argv[1:4])
