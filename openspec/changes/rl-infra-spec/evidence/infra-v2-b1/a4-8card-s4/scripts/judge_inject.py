"""Structured judgement of fault-injection / membership cases from the journal and the event tape (not just the terminal state).

usage: judge_inject.py <case> <journal.jsonl|elastic-state dir> <tape.jsonl> [--router-samples F] [--expect-members 1,1,3,3,1,1]
                       [--cells c0,c1,c2] [--out judgment.json] [--marker-dir DIR]
cases: e1a_c | e1b | wd | a4b
verdicts (exit code): PASS 0 | FAIL 1 | INVALID_TEST 4.  INVALID_TEST = the injection target does not exist or the injection was not applied/reached:
neither a product pass nor a product failure.  --marker-dir writes the marker files injection_not_reached / recovery_failed.

Event names/fields (infra-e1 interface; tape: {"event":..}, journal: {"kind":..}):
  test_injection  kind target_members phase ts applied   (watchdog: kind=block_update + reached_ts)
  rl_membership   config_epoch members tx_id kind round  (kind = up|down; round = first round the membership serves)
  test_hold       stage=start|end (two records; the end record has start_ts, end_ts)
  journal watchdog: injection_reached, classification in {INJECTION_NOT_REACHED, FIRED_AFTER_BLOCK_RELEASED (also INVALID), FIRED_ON_BLOCKED_UPDATE (the only hit)}
  journal test_injection: injection_kind instead of kind (tape: kind)
  journal phase REBUILD_OLD / REBUILT_OLD: cause (payload_mismatch, ...)
"""
import json, os, sys
from pathlib import Path

TERMINAL = ("SUCCEEDED", "REBUILT_OLD", "CANCELLED", "RECOVERY_REQUIRED")
EXIT = {"PASS": 0, "FAIL": 1, "INVALID_TEST": 4}


def load(p):
    out = []
    p = Path(p)
    if p.is_dir():
        p = p / "reconfig" / "journal.jsonl" if (p / "reconfig").exists() else p / "elastic-state" / "reconfig" / "journal.jsonl"
    for l in p.read_text().splitlines():
        try: out.append(json.loads(l))
        except Exception: pass
    return out


def named(recs, name):
    """records of one event name from tape ('event') or journal ('kind') form."""
    return [r for r in recs if r.get("event") == name or (r.get("event") is None and r.get("kind") == name)]


def ikind(e):
    """injection kind: tape field 'kind'; in the journal the record kind is 'test_injection' and the injection kind is 'injection_kind'."""
    return e.get("injection_kind") if e.get("event") is None else e.get("kind")


def members_of(r):
    m = r.get("members")
    return sorted(m) if isinstance(m, (list, tuple)) else None


def injection_validity(events, kinds, known):
    """events: test_injection records. -> (valid, reasons, applied_events)."""
    inj = [e for e in events if kinds is None or ikind(e) in kinds]
    if not inj:
        return False, ["no test_injection event of kind %s: injection not reached" % (sorted(kinds) if kinds else "any")], []
    ok = [e for e in inj if e.get("applied") is True]
    reasons = []
    if not ok:
        reasons.append("test_injection.applied is not true on any event: injection not applied")
        return False, reasons, []
    good = []
    for e in ok:
        t = e.get("target_members")
        if t is not None and (not t or (known and not set(t) <= set(known))):
            reasons.append("injection target %s does not exist among %s" % (t, sorted(known)))
            continue
        good.append(e)
    return bool(good), reasons, good


def phases(journal):
    return [r for r in journal if r.get("kind") == "phase"]


def judge_e1b(journal, tape, known, router_samples=None):
    res = {"case": "e1b", "checks": {}}
    ev = named(tape, "test_injection") + named(journal, "test_injection")
    valid, why, good = injection_validity(ev, None, known)
    res["injection_valid"] = valid; res["invalid_reasons"] = why
    if not valid:
        res["verdict"] = "INVALID_TEST"; res["marker"] = "injection_not_reached"; return res
    ph = phases(journal)
    term = [r["phase"] for r in ph if r.get("phase") in TERMINAL]
    old = [r for r in ph if r.get("phase") in ("REBUILD_OLD", "REBUILT_OLD")]
    res["terminal"] = term; res["causes"] = [r.get("cause") for r in old]
    res["checks"]["terminal_REBUILT_OLD"] = "REBUILT_OLD" in term and "SUCCEEDED" not in term
    res["checks"]["cause_payload_mismatch"] = bool(old) and all(r.get("cause") == "payload_mismatch" for r in old)
    # E1-A/E1-B (a): sample the router inside the test_hold window only
    holds = named(tape, "test_hold") + named(journal, "test_hold")
    if router_samples is not None:
        a = judge_hold_window(holds, router_samples, res)
        if a is None:   # no completed test_hold window, or too few samples inside it: (a) could not be observed -> not a product result
            res["invalid_reasons"] = ["test_hold window missing/incomplete or <3 router samples inside it: %s" % res.get("hold_window")]
            res["verdict"] = "INVALID_TEST"; res["marker"] = "injection_not_reached"; return res
        res["checks"]["a_cordoned_in_hold_window"] = a
    if "RECOVERY_REQUIRED" in term or not term:
        res["verdict"] = "FAIL"; res["marker"] = "recovery_failed"; return res
    res["verdict"] = "PASS" if all(v is True for v in res["checks"].values()) else "FAIL"
    return res


def judge_hold_window(holds, samples, res):
    if not holds:
        res["hold_window"] = None; return None
    # two records per hold, stage=start|end; the end record carries start_ts and end_ts. No end record = hold never completed = not verified.
    ends = [h for h in holds if h.get("stage") == "end" and h.get("start_ts") is not None and h.get("end_ts") is not None]
    if not ends:
        res["hold_window"] = None; return None
    t0 = min(h["start_ts"] for h in ends); t1 = max(h["end_ts"] for h in ends)
    pre = {u for s in samples if s["t"] < t0 for u in ((s.get("data") or {}).get("inflight") or {})}
    win = [s for s in samples if t0 <= s["t"] <= t1 and s.get("data")]
    new_urls = {u for s in win for u in s["data"].get("inflight", {})} - pre
    busy = [(s["t"], u, n) for s in win for u, n in s["data"].get("inflight", {}).items() if u in new_urls and n]
    res["hold_window"] = {"t0": t0, "t1": t1, "samples": len(win), "new_urls": sorted(new_urls), "busy": busy[:5]}
    if len(win) < 3:
        return None     # too few samples in the window = not observed (invalid test), never a pass
    return not busy


def judge_wd(journal, tape, known):
    res = {"case": "wd", "checks": {}}
    wd = named(journal, "watchdog")
    ev = [e for e in named(tape, "test_injection") + named(journal, "test_injection") if ikind(e) == "block_update"]
    res["watchdog"] = [{k: w.get(k) for k in ("tx_id", "phase", "injection_reached", "classification")} for w in wd]
    fired = [w for w in wd if w.get("classification") == "FIRED_ON_BLOCKED_UPDATE"]
    reached = [e for e in ev if e.get("applied") is True and e.get("reached_ts") is not None]
    if not fired or not reached:
        res["invalid_reasons"] = ["watchdog classification is not FIRED_ON_BLOCKED_UPDATE or block_update never reached (reached_ts missing/applied!=true): %s" % [w.get("classification") for w in wd]]
        res["verdict"] = "INVALID_TEST"; res["marker"] = "injection_not_reached"; return res
    res["injection_valid"] = True
    acts = named(journal, "watchdog_action")
    term = [r["phase"] for r in phases(journal) if r.get("phase") in TERMINAL]
    res["terminal"] = term
    res["checks"]["watchdog_action_killed"] = any(a.get("killed") for a in acts)
    res["checks"]["terminal_REBUILT_OLD"] = "REBUILT_OLD" in term and "SUCCEEDED" not in term
    res["note"] = "valid test; remaining wd criteria (plan-3.8-4.4-v2 s4 (3)-(5)) via analyze_wd.py"
    if "RECOVERY_REQUIRED" in term or not term:
        res["verdict"] = "FAIL"; res["marker"] = "recovery_failed"; return res
    res["verdict"] = "PASS" if all(res["checks"].values()) else "FAIL"
    return res


def judge_a4b(journal, tape, known):
    res = {"case": "a4b", "checks": {}}
    ev = named(tape, "test_injection") + named(journal, "test_injection")
    valid, why, good = injection_validity(ev, {"tool_wait"}, known)
    res["injection_valid"] = valid; res["invalid_reasons"] = why
    if not valid:
        res["verdict"] = "INVALID_TEST"; res["marker"] = "injection_not_reached"; return res
    term = [r["phase"] for r in phases(journal) if r.get("phase") in TERMINAL]
    res["terminal"] = term
    res["checks"]["terminal_CANCELLED"] = "CANCELLED" in term and "SUCCEEDED" not in term
    if "RECOVERY_REQUIRED" in term or not term:
        res["verdict"] = "FAIL"; res["marker"] = "recovery_failed"; return res
    res["verdict"] = "PASS" if all(res["checks"].values()) else "FAIL"
    return res


def judge_e1a_c(journal, tape, expect):
    """E1-A (c), re-based on rl_membership per round + retained rl_publication check (after the shrink, the next full publication has 1 member)."""
    res = {"case": "e1a_c", "checks": {}}
    mem = [r for r in named(tape, "rl_membership") if r.get("round") is not None and r.get("members") is not None]
    if not mem:
        res["verdict"] = "INVALID_TEST"; res["invalid_reasons"] = ["no rl_membership events with round/members on the tape"]; return res
    mem.sort(key=lambda r: (r["round"], r.get("config_epoch", 0)))
    got = []
    for rnd in range(1, len(expect) + 1):
        cur = [r for r in mem if r["round"] <= rnd]
        got.append(len(cur[-1]["members"]) if cur else None)
    res["members_per_round"] = got; res["expected"] = expect
    res["checks"]["members_per_round_equal"] = got == expect
    downs = [r for r in mem if r.get("kind") == "down"]
    pubs = [r for r in tape if r.get("event") == "rl_publication"]
    ok = False
    if downs:
        i_down = tape.index(downs[-1]) if downs[-1] in tape else None
        after = [p for p in (tape[i_down + 1:] if i_down is not None else []) if p.get("event") == "rl_publication"]
        if after:
            pm = after[0].get("sync/publication_members", after[0].get("publication_members"))
            res["first_publication_after_down_members"] = pm
            ok = (len(pm) if isinstance(pm, (list, tuple)) else pm) == 1
    res["checks"]["publication_after_down_is_1_member"] = ok
    res["verdict"] = "PASS" if all(res["checks"].values()) else "FAIL"
    return res


def main(argv):
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("case"); ap.add_argument("journal"); ap.add_argument("tape")
    ap.add_argument("--router-samples"); ap.add_argument("--expect-members"); ap.add_argument("--cells", default="")
    ap.add_argument("--out"); ap.add_argument("--marker-dir")
    a = ap.parse_args(argv)
    journal, tape = load(a.journal), load(a.tape)
    known = set(filter(None, a.cells.split(",")))
    for r in named(journal, "add_intent"): known |= set(r.get("members") or [])
    for r in named(tape, "rl_membership"): known |= set(r.get("members") or [])
    samples = [json.loads(l) for l in open(a.router_samples)] if a.router_samples and os.path.exists(a.router_samples) else None
    if a.case == "e1b" and samples is None: samples = []   # E1-B (a) needs router samples; none = not observed = invalid test
    if a.case == "e1b": res = judge_e1b(journal, tape, known, samples)
    elif a.case == "wd": res = judge_wd(journal, tape, known)
    elif a.case == "a4b": res = judge_a4b(journal, tape, known)
    elif a.case == "e1a_c": res = judge_e1a_c(journal, tape, [int(x) for x in a.expect_members.split(",")])
    else: raise SystemExit("unknown case " + a.case)
    if a.out: Path(a.out).write_text(json.dumps(res, indent=1, default=str))
    if a.marker_dir and res.get("marker"):
        Path(a.marker_dir, res["marker"]).write_text(json.dumps({"case": a.case, "verdict": res["verdict"], "why": res.get("invalid_reasons") or res.get("terminal")}))
    print(json.dumps(res, default=str)); return EXIT[res["verdict"]]


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
