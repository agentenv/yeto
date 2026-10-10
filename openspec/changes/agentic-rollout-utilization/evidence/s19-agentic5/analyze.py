"""S19 #7 (ARU 5.5 criterion 5) + #8 (spot 4.4) judgment. Pre-registered criteria:
infra-drafts/S19-AGENTIC5-PRELAUNCH-REVIEW.md §3. Reads raw data already on disk; writes judgment.json here."""
import glob
import json
import statistics as st
import sys
from pathlib import Path

B = Path("/home/michael/work/s1-runs")
RUNS = {"A": "s19-agentic5-a-20261010b", "B": "s19-agentic5-b-20261010b"}
OUT = Path(__file__).with_name("judgment.json")


def events(run):
    tapes = sorted(glob.glob(str(B / run / "tape-direct/**/rl-island-0*.jsonl"), recursive=True))
    srcs = tapes + [str(B / run / "launch.log")]
    seen, ev = set(), []
    for src in srcs:
        if not Path(src).exists():
            continue
        for line in open(src, errors="replace"):
            for marker in ("YETO_RL_EVENT ", "[yeto] rl_inflight_save ", "[yeto] spot_reclaim "):
                i = line.find(marker)
                if i < 0:
                    continue
                try:
                    e = json.loads(line[i + len(marker):].strip())
                except ValueError:
                    continue
                if marker.startswith("[yeto] "):
                    e = {"event": marker.split()[1], **e}
                key = json.dumps(e, sort_keys=True)
                if key not in seen:
                    seen.add(key)
                    ev.append(e)
        if tapes and src == tapes[-1]:
            pass
    tape_ev = []
    for src in tapes:
        for line in open(src, errors="replace"):
            try:
                e = json.loads(line[line.find("{"):])
            except ValueError:
                continue
            if isinstance(e, dict) and "event" in e:
                tape_ev.append(e)
    return tapes, ev + [e for e in tape_ev if json.dumps(e, sort_keys=True) not in seen]


def q(xs, p):
    xs = sorted(x for x in xs if x is not None)
    return xs[min(len(xs) - 1, int(p * (len(xs) - 1) + 0.5))] if xs else None


def summ(name, run):
    tapes, ev = events(run)
    by = {}
    for e in ev:
        by.setdefault(e["event"], []).append(e)
    car = {}
    for c in by.get("rl_rollout_carry_over", []):
        car[c.get("rollout_id")] = c
    keys = ("resumed_groups", "suspended_groups", "suspended_trajectories", "cross_version_tokens",
            "trained_response_tokens", "cross_version_scored_tokens", "cross_version_unscored_samples",
            "cross_version_unscored_reasons", "cross_version_truncated_fraction", "cross_version_ratio_p50",
            "cross_version_ratio_p90", "cross_version_ratio_p99", "cross_version_ratio_min",
            "cross_version_ratio_max", "cross_version_unknown_version_tokens")
    per = {r: {k: c.get(k) for k in keys if k in c} for r, c in sorted(car.items(), key=lambda kv: kv[0] or -1)}
    trained = {e.get("rollout_id"): e for e in by.get("rl_round_trained", [])}
    loc = by.get("rl_local_round", [])
    saves, paths = [], set()
    for x in by.get("rl_inflight_save", []):
        k = x.get("path") or json.dumps(x, sort_keys=True)
        if k not in paths:
            paths.add(k)
            saves.append(x)
    reh = [s for s in saves if s.get("kind") == "rehearsal"]
    rec = [s for s in saves if s.get("kind") == "reclaim"]
    tot = [s.get("total_s") for s in reh if s.get("total_s") is not None]
    return {
        "run": run, "tapes": tapes,
        "rc": (B / run / "rc.txt").read_text().strip() if (B / run / "rc.txt").exists() else None,
        "rounds_trained": sorted(k for k in trained if k is not None),
        "per_round_carry": per,
        "tis_clipfrac_by_round": {e.get("rollout_id", e.get("local_round_id")): e.get("tis_clipfrac") for e in loc},
        "grad_norm": [e.get("grad_norm") for e in loc],
        "invariant_failed": len(by.get("rl_invariant_failed", [])),
        "rehearsals": [{k: s.get(k) for k in ("round", "entries", "agentic_entries", "bytes", "export_s", "write_s",
                                                "commit_s", "commit_error", "total_s", "error")} for s in reh],
        "rehearsal_total_s_p50_p90_max": [q(tot, .5), q(tot, .9), q(tot, 1.0)] if tot else None,
        "reclaim_saves": rec,
        "spot_reclaim": by.get("spot_reclaim", []),
    }


def judge(res):
    a, b = res["A"], res["B"]
    out = {}
    rounds_ok = lambda s: sum(1 for c in s["per_round_carry"].values() if (c.get("resumed_groups") or 0) > 0)
    out["validity"] = {"A_rc": a["rc"], "A_rounds": len(a["rounds_trained"]), "A_resumed_rounds": rounds_ok(a),
                       "B_resumed_rounds": rounds_ok(b),
                       "pass": a["rc"] == "rc=0" and len(a["rounds_trained"]) >= 8 and rounds_ok(a) >= 3 and rounds_ok(b) >= 3}
    crossing = [(n, r, c) for n, s in res.items() for r, c in s["per_round_carry"].items() if (c.get("cross_version_tokens") or 0) > 0]
    unscored = {f"{n}{r}": c.get("cross_version_unscored_samples") for n, r, c in crossing}
    out["5a_unscored_zero"] = {"rounds": unscored, "pass": bool(crossing) and all(v == 0 for v in unscored.values())}
    with_q = [r for r, c in a["per_round_carry"].items() if c.get("cross_version_ratio_p50") is not None]
    out["5b_quantiles_A_rounds"] = {"rounds": with_q, "pass": len(with_q) >= 3}
    fr = [c.get("cross_version_truncated_fraction") for _, _, c in crossing if c.get("cross_version_truncated_fraction") is not None]
    F = max(fr) if fr else None
    out["5c_no_fallback"] = {"max_fraction": F, "pass": F is not None and F < 0.5}
    if F is None:
        rec = "no data"
    elif F < 0.1:
        rec = "keep warn 0.2 / fallback 0.5 (margin >= 2x)"
    elif F < 0.2:
        rec = "keep, margin below 2x"
    else:
        rec = "thresholds unchanged; a real run would warn; main agent decides"
    out["calibration"] = {"F": F, "recommendation": rec}
    reh = a["rehearsals"] + b["rehearsals"]
    ok = [r for r in reh if r.get("error") is None and r.get("total_s") is not None]
    out["8a_rehearsals"] = {"n": len(reh), "ok": len(ok), "max_total_s": max((r["total_s"] for r in ok), default=None),
                            "pass": bool(reh) and len(ok) == len(reh) and all(r["total_s"] <= 25 for r in ok)
                            and len(a["rehearsals"]) >= len(a["rounds_trained"]) > 0}
    sr = b["spot_reclaim"]
    rc = b["reclaim_saves"]
    out["8b_reclaim"] = {"spot_reclaim": sr, "reclaim_save": rc,
                         "pass": bool(sr) and sr[0].get("outcome") == "saved" and (sr[0].get("handler_s") or 99) <= 25
                         and bool(rc) and (rc[0].get("total_s") or 99) <= 25}
    return out


if __name__ == "__main__":
    res = {n: summ(n, r) for n, r in RUNS.items()}
    j = judge(res)
    OUT.write_text(json.dumps({"summary": res, "judgment": j}, indent=1, sort_keys=True, default=str))
    print(json.dumps(j, indent=1, default=str))
