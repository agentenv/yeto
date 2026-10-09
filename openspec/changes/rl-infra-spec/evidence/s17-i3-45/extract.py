"""S17 N7 I3: per-run G-4.5 evidence from files already on disk (launch.log events, pulled ledger/journal, snapshots).
usage: python3 extract.py <run dir> [...]  -> prints a summary and writes <run dir>/g45-extract.json"""
import glob, io, json, os, sys, tarfile
from collections import Counter


def events(R):
    out = []
    for line in open(f"{R}/launch.log", errors="replace"):
        at = line.find("YETO_RL_EVENT ")
        if at >= 0:
            try:
                out.append(json.loads(line[at + 14:]))
            except ValueError:
                pass
    return out


def jl(p):
    try:
        return [json.loads(l) for l in open(p) if l.strip()]
    except (OSError, ValueError):
        return []


def snapshot_journals(R):
    """Newest readable snapshot: (reconfig journal, ledger journal)."""
    for p in sorted(glob.glob(f"{R}/es-snapshots/*.tgz"), reverse=True):
        try:
            with tarfile.open(p) as tf:
                def rd(n):
                    try:
                        return [json.loads(l) for l in tf.extractfile(n).read().decode().splitlines() if l.strip()]
                    except (KeyError, AttributeError):
                        return []
                return rd("elastic-state/reconfig/journal.jsonl"), rd("elastic-state/ledger/journal.jsonl"), p
        except (tarfile.TarError, EOFError, OSError):
            continue
    return [], [], None


def main():
    for R in sys.argv[1:]:
        ev = events(R)
        pulled_j = jl(f"{R}/pulled/elastic-state__reconfig__journal.jsonl")
        pulled_l = jl(f"{R}/pulled/elastic-state__ledger__journal.jsonl")
        snap_j, snap_l, snap = snapshot_journals(R)
        J = snap_j if len(snap_j) >= len(pulled_j) else pulled_j
        Lg = snap_l if len(snap_l) >= len(pulled_l) else pulled_l
        phases = [(e.get("request_id"), e.get("phase"), round(e.get("time_unix", 0), 1)) for e in ev if e.get("event") == "rl_reconfig_phase"]
        jphases = [(r.get("request_id"), r.get("phase"), round(r.get("wall_time", 0), 1), str(r.get("error") or "")[:200])
                   for r in J if r.get("kind") == "phase"]
        rounds = [(e.get("rollout_id"), e.get("train_step"), (e.get("data_cursor") or {}).get("sample_offset"))
                  for e in ev if e.get("event") == "rl_round_trained"]
        offsets = [r[2] for r in rounds if r[2] is not None]
        kinds = Counter(r.get("kind") or r.get("event") or r.get("state") for r in Lg)
        opt = [r for r in Lg if "optimizer_applied" in json.dumps(r)]
        opt_keys = Counter(json.dumps({k: r.get(k) for k in ("rollout_id", "round", "local_step", "step") if k in r}, sort_keys=True) for r in opt)
        consumed = [r for r in Lg if "consumed" in json.dumps(r).lower()]
        cons_keys = Counter(json.dumps({k: r.get(k) for k in ("group_id", "group", "rollout_id", "sample_ids_sha256") if k in r}, sort_keys=True) for r in consumed)
        res = {
            "run": os.path.basename(R),
            "rc": open(f"{R}/rc.txt").read().strip() if os.path.exists(f"{R}/rc.txt") else None,
            "deliver": [json.loads(l).get("action") for l in open(f"{R}/deliver.jsonl")] if os.path.exists(f"{R}/deliver.jsonl") else None,
            "injection_lines": [l.strip()[:200] for l in open(f"{R}/launch.log", errors="replace") if "TEST INJECTION" in l or "INJECT" in l and "export" not in l][:6],
            "rebuilt_events": [e for e in ev if e.get("event") == "rl_trainer_rebuilt"],
            "tape_reconfig_phases": phases,
            "journal_phases": jphases,
            "journal_source": "snapshot " + str(snap) if J is snap_j and snap_j else "pulled",
            "recovery_records": [r for r in J if "RECOVERY" in json.dumps(r)][:4],
            "rounds_trained": rounds,
            "rounds_after_terminal": None,
            "cursor_monotonic": all(b >= a for a, b in zip(offsets, offsets[1:])),
            "ledger_kinds": dict(kinds),
            "optimizer_applied_dups": {k: v for k, v in opt_keys.items() if v > 1},
            "consumed_dups": {k: v for k, v in cons_keys.items() if v > 1},
            "finalized": any(e.get("event") == "rl_learner_finalized" for e in ev),
        }
        term = [p for p in jphases if p[1] in ("SUCCEEDED", "REBUILT_OLD", "CANCELLED", "RECOVERY_REQUIRED")]
        if term:
            t_end = term[-1][2]
            res["terminal"] = term[-1]
            start = [p for p in jphases if p[1] == "REBUILDING_TRAINER"]
            res["rebuilding_to_terminal_s"] = round(t_end - start[0][2], 1) if start else None
            res["rounds_after_terminal"] = [e.get("rollout_id") for e in ev if e.get("event") == "rl_round_trained" and e.get("time_unix", 0) > t_end]
        json.dump(res, open(f"{R}/g45-extract.json", "w"), indent=1, default=str)
        print(json.dumps({k: v for k, v in res.items() if k not in ("rebuilt_events", "recovery_records")}, default=str)[:1800])
        print()


if __name__ == "__main__":
    main()
