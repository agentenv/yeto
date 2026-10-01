"""Structured judgement of the A4 fault-injection / membership cases from the journal, the event tape and the
in-container samplers/probes (never from the terminal state alone).

usage: judge_inject.py <case> <journal.jsonl|elastic-state dir> <tape.jsonl> [--router-samples F] [--gpu-samples F]
                       [--probe-after F] [--probe-stale F] [--probe-oldepoch F] [--dkill-log F] [--expect-members 1,1,3,3,1,1]
                       [--cells c0,c1,c2] [--out judgment.json] [--marker-dir DIR]
cases: e1a_c | e1b | wd | a4b | a4bu | d123 | d4 | r5 | r6 | r7 | r5c   (r*: E1-D ⑤⑥⑦ with the 3.7 restart recovery, recovery-design.md §10.4)
verdicts (exit code): PASS 0 | FAIL 1 | INVALID_TEST 4.
  INVALID_TEST = the test could not observe what it claims to judge: the injection target does not exist, the injection
  was not applied/reached, a required sampler/probe output is missing, the sampled window is too short, or the engines
  under test never became identifiable. Neither a product pass nor a product failure; never PASS on empty evidence.
  --marker-dir writes the marker files injection_not_reached / recovery_failed / evidence_missing.

Event names/fields (infra-e1 interface; tape: {"event":..}, journal: {"kind":..}):
  test_injection  kind target_members phase ts applied   (watchdog: kind=block_update + reached_ts / released_ts / outcome)
  member_engines  target_members engine_urls           (the fork's UpdatableEngines: identity of the new engines, independent of the router)
  lora_warmup     target_members adapter engines[{engine,ok,..}] applied
  lora_readback   target_members reference_lora_keys engines[{engine,keys,lora_keys,lora_keys_differ}] blind
  rl_membership   config_epoch members tx_id kind round  (kind = up|down; round = 0-based rollout_id of the first rollout the membership serves)
  test_hold       stage=start|end (two records; the end record has start_ts, end_ts)
  journal watchdog: wall_time, target_cells, classification in {INJECTION_NOT_REACHED, FIRED_AFTER_BLOCK_RELEASED (both INVALID), FIRED_ON_BLOCKED_UPDATE}
  journal watchdog_action: killed[{cell,worker,generation}], errors[{cell,kind,error}]
  journal phase REBUILD_OLD / REBUILT_OLD: cause (payload_mismatch, lora_unverifiable, ...), inconsistent_engines
  journal drain_timeout: active_requests tool_wait blockers;  undrain_failed: error
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
    if not p.exists():
        return out
    for l in p.read_text().splitlines():
        try: out.append(json.loads(l))
        except Exception: pass
    return out


def load_probe(p):
    """fork_probe.py output file: the last line that is a JSON object with probe_attested=true, else None."""
    if not p or not os.path.exists(p):
        return None
    for l in reversed(Path(p).read_text().splitlines()):
        if l.startswith("{"):
            try:
                d = json.loads(l)
                if d.get("probe_attested") is True:
                    return d
            except Exception:
                pass
    return None


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


def _invalid(res, reason, marker="injection_not_reached"):
    res.setdefault("invalid_reasons", []).append(reason)
    res["verdict"] = "INVALID_TEST"; res["marker"] = marker
    return res


def _finish(res, term):
    if "RECOVERY_REQUIRED" in term or not term:
        res["verdict"] = "FAIL"; res["marker"] = "recovery_failed"; return res
    res["verdict"] = "PASS" if all(v is True for v in res["checks"].values()) else "FAIL"
    return res


def new_engine_urls(journal, tape, targets):
    """URLs of the engines the member publication targeted (member_engines, written by the publisher from the fork's
    UpdatableEngines). Independent of the router listing. None = no such record for these targets."""
    for r in reversed(named(journal, "member_engines") + named(tape, "member_engines")):
        if set(r.get("target_members") or []) == set(targets) and r.get("engine_urls"):
            return sorted(r["engine_urls"])
    return None


# ----------------------------------------------------------------------------------------------- E1-B
def judge_e1b(journal, tape, known, router_samples=None, probes=None):
    """3.5 / E1-B (a)-(d). probes = {"after": ..., "stale": ..., "oldepoch": ...} (fork_probe.py outputs, parsed)."""
    res = {"case": "e1b", "checks": {}, "invalid_reasons": []}
    ev = named(tape, "test_injection") + named(journal, "test_injection")
    valid, why, good = injection_validity([e for e in ev if ikind(e) == "lora_perturb"], None, known)
    res["injection_valid"] = valid; res["invalid_reasons"] += why
    if not valid:
        return _invalid(res, "lora_perturb injection not applied")
    targets = sorted(set(m for e in good for m in (e.get("target_members") or [])))
    res["targets"] = targets
    # target-state evidence: the engines' pools hold the adapter and the perturbed one reads back differently
    rb = [r for r in named(journal, "lora_readback") + named(tape, "lora_readback")
          if set(r.get("target_members") or []) == set(targets)]
    if not rb:
        return _invalid(res, "no lora_readback record for the targets: injection effect on the engines not observed", "evidence_missing")
    engines = rb[-1].get("engines") or []
    res["readback"] = {"engines": engines, "blind": rb[-1].get("blind"), "reference_lora_keys": rb[-1].get("reference_lora_keys")}
    if not engines or any((e.get("lora_keys") or 0) == 0 for e in engines) or rb[-1].get("blind"):
        return _invalid(res, "read-back has engines without adapter keys (blind): the perturbation could not be observed", "evidence_missing")
    if not any((e.get("lora_keys_differ") or 0) > 0 for e in engines):
        return _invalid(res, "read-back shows no adapter difference on any target engine: the injection did not change the engines' adapter")
    ph = phases(journal)
    term = [r["phase"] for r in ph if r.get("phase") in TERMINAL]
    old = [r for r in ph if r.get("phase") in ("REBUILD_OLD", "REBUILT_OLD")]
    res["terminal"] = term; res["causes"] = [r.get("cause") for r in old]
    res["checks"]["terminal_REBUILT_OLD"] = "REBUILT_OLD" in term and "SUCCEEDED" not in term
    res["checks"]["cause_payload_mismatch"] = bool(old) and all(r.get("cause") == "payload_mismatch" for r in old)
    res["checks"]["refusal_names_engines"] = bool(old) and all(bool(r.get("inconsistent_engines")) for r in old)
    # (a): the new engines (identified independently) take no request before admission: router sampled inside the hold window
    urls = new_engine_urls(journal, tape, targets)
    res["new_engine_urls"] = urls
    if urls is None:
        return _invalid(res, "no member_engines record: the new engines cannot be identified independently of the router", "evidence_missing")
    holds = named(tape, "test_hold") + named(journal, "test_hold")
    if router_samples is None:
        return _invalid(res, "router samples missing: (a) not observed", "evidence_missing")
    a = judge_hold_window(holds, router_samples, res, urls)
    if a is None:
        return _invalid(res, "test_hold window missing/incomplete, <3 router samples inside it, or the new engines never listed by the router: %s" % res.get("hold_window"), "evidence_missing")
    res["checks"]["a_cordoned_in_hold_window"] = a
    # (b): after REBUILT_OLD the old members serve the same token and the injected cells are stopped (fork probe)
    probes = probes or {}
    after = probes.get("after")
    if after is None:
        return _invalid(res, "fork status probe after the terminal state missing: (b) not observed", "evidence_missing")
    statuses = after.get("cell_statuses") or {}
    tcells = [t.split(":", 1)[1] for t in targets]
    res["probe_after"] = {"membership": after.get("membership"), "versions": after.get("versions"),
                          "target_statuses": {c: statuses.get(c) for c in tcells}}
    # a stopped cell is deregistered from the fork controller (absent from get_cell_statuses) or reported non-Serving
    res["checks"]["b_injected_cells_not_serving"] = bool(statuses) and all(
        c not in statuses or "Serving" not in str(statuses[c]) for c in tcells)
    vers = after.get("versions") or {}
    serving = {k: v for k, v in vers.items() if k not in tcells}
    res["checks"]["b_old_members_same_version"] = len(serving) >= 1 and len(set(serving.values())) == 1
    # (c): a late ACK on a stopped cell's old URL changes no serving member's version
    stale = probes.get("stale")
    if stale is None:
        return _invalid(res, "stale-ACK probe missing: (c) not observed", "evidence_missing")
    res["probe_stale"] = {k: stale.get(k) for k in ("stale_ack", "versions_equal", "stale_url")}
    res["checks"]["c_stale_ack_changes_nothing"] = stale.get("versions_equal") is True and (
        stale.get("stale_url") is None or stale.get("stale_url") in urls)
    # (d): the old epoch is refused
    oe = probes.get("oldepoch")
    if oe is None:
        return _invalid(res, "old-epoch probe missing: (d) not observed", "evidence_missing")
    res["probe_oldepoch"] = {k: oe.get(k) for k in ("refused", "error")}
    res["checks"]["d_old_epoch_refused"] = oe.get("refused") is True
    return _finish(res, term)


def judge_hold_window(holds, samples, res, expected_new_urls=None):
    """True/False = the new engines were idle/busy inside the completed hold window; None = not observed
    (no completed hold, <3 samples in it, or the expected new engines are not listed by the router in it)."""
    if not holds:
        res["hold_window"] = None; return None
    ends = [h for h in holds if h.get("stage") == "end" and h.get("start_ts") is not None and h.get("end_ts") is not None]
    if not ends:
        res["hold_window"] = None; return None
    t0 = min(h["start_ts"] for h in ends); t1 = max(h["end_ts"] for h in ends)
    pre = {u for s in samples if s.get("t", 0) < t0 for u in ((s.get("data") or {}).get("inflight") or {})}
    win = [s for s in samples if t0 <= s.get("t", -1) <= t1 and s.get("data")]
    listed = {u for s in win for u in s["data"].get("inflight", {})}
    new_urls = sorted(listed - pre) if expected_new_urls is None else sorted(set(expected_new_urls) & listed)
    busy = [(s["t"], u, n) for s in win for u, n in s["data"].get("inflight", {}).items() if u in new_urls and n]
    res["hold_window"] = {"t0": t0, "t1": t1, "samples": len(win), "new_urls": new_urls, "busy": busy[:5],
                          "expected_new_urls": expected_new_urls, "listed_minus_pre": sorted(listed - pre)}
    if len(win) < 3:
        return None     # too few samples in the window = not observed (invalid test), never a pass
    if not new_urls:
        return None     # nothing to check: "no busy new engine" over an empty set is not an observation
    if expected_new_urls is not None and set(new_urls) != set(expected_new_urls):
        return None     # some new engine was never listed by the router during the window
    return not busy


# ----------------------------------------------------------------------------------------------- watchdog
def _apps_at(gpu_samples, t, before=True):
    """set of (uuid, pid) of the last sample at/before t (before=True) or first at/after t."""
    if before:
        c = [s for s in gpu_samples if s.get("t", 0) <= t]
        s = c[-1] if c else None
    else:
        c = [s for s in gpu_samples if s.get("t", 0) >= t]
        s = c[0] if c else None
    return None if s is None else {(a[0], a[1]) for a in (s.get("apps") or []) if len(a) >= 2}


def judge_wd(journal, tape, known, gpu_samples=None, probe_after=None, gpu_release_s=60.0, unblock_s=60.0):
    """plan-3.8-4.4-v2 §4 (1)-(5)."""
    res = {"case": "wd", "checks": {}, "invalid_reasons": []}
    wd = named(journal, "watchdog")
    ev = [e for e in named(tape, "test_injection") + named(journal, "test_injection") if ikind(e) == "block_update"]
    res["watchdog"] = [{k: w.get(k) for k in ("tx_id", "phase", "injection_reached", "classification", "target_cells")} for w in wd]
    fired = [w for w in wd if w.get("classification") == "FIRED_ON_BLOCKED_UPDATE"]
    reached = [e for e in ev if e.get("applied") is True and e.get("reached_ts") is not None]
    if not fired or not reached:
        return _invalid(res, "watchdog classification is not FIRED_ON_BLOCKED_UPDATE or block_update never reached (reached_ts missing/applied!=true): %s" % [w.get("classification") for w in wd])
    res["injection_valid"] = True
    w = fired[0]; targets = sorted(w.get("target_cells") or [])
    if not targets:
        return _invalid(res, "watchdog record names no target cells")
    acts = [a for a in named(journal, "watchdog_action") if a.get("tx_id") == w.get("tx_id")]
    killed = [k for a in acts for k in (a.get("killed") or []) if isinstance(k, dict)]
    errors = [e for a in acts for e in (a.get("errors") or [])] + [a["error"] for a in acts if a.get("error")]
    res["killed"] = killed; res["errors"] = errors
    # (1) every target cell's workers killed (generation recorded), no unresolved target
    res["checks"]["killed_covers_all_targets"] = bool(killed) and set(k.get("cell") for k in killed) == set(targets) and all(
        k.get("worker") and k.get("generation") is not None for k in killed)
    res["checks"]["no_kill_errors"] = not errors
    # (2) the blocked call returned (released by the kill, not by its own timer) within unblock_s, terminal REBUILT_OLD
    rel = [e for e in ev if e.get("applied") is True and e.get("released_ts") is not None]
    res["release"] = [{k: e.get(k) for k in ("outcome", "released_ts")} for e in rel]
    t_fire = w.get("wall_time")
    res["checks"]["unblocked_by_the_kill"] = bool(rel) and str(rel[-1].get("outcome", "")).startswith("target")
    res["checks"]["unblocked_within_%ds" % int(unblock_s)] = bool(rel) and t_fire is not None and (rel[-1]["released_ts"] - t_fire) <= unblock_s
    ph = phases(journal)
    term = [r["phase"] for r in ph if r.get("phase") in TERMINAL]
    res["terminal"] = term
    res["checks"]["terminal_REBUILT_OLD"] = "REBUILT_OLD" in term and "SUCCEEDED" not in term
    t_init = next((r.get("wall_time") for r in ph if r.get("phase") == "INITIALIZING" and r.get("tx_id") == w.get("tx_id")), None)
    t_old = next((r.get("wall_time") for r in ph if r.get("phase") == "REBUILT_OLD" and r.get("tx_id") == w.get("tx_id")), None)
    # (3)(4) from the in-container GPU sampler: target GPUs = GPUs that were empty before INITIALIZING and got processes during the transaction
    if gpu_samples is None or t_init is None:
        return _invalid(res, "gpu samples or INITIALIZING timestamp missing: (3)(4) not observed", "evidence_missing")
    pre = _apps_at(gpu_samples, t_init, before=True)
    during = set()
    for s in gpu_samples:
        if t_init <= s.get("t", 0) <= (t_fire or t_init):
            during |= {(a[0], a[1]) for a in (s.get("apps") or []) if len(a) >= 2}
    if pre is None or not during:
        return _invalid(res, "gpu sampler has no samples before INITIALIZING or during the transaction", "evidence_missing")
    old_gpus = {u for u, _ in pre}
    target_gpus = {u for u, _ in during} - old_gpus
    res["gpus"] = {"old": sorted(old_gpus), "target": sorted(target_gpus), "old_pids": sorted(p for _, p in pre)}
    if not target_gpus:
        return _invalid(res, "no GPU received a new process during the transaction: the target generation never started", "evidence_missing")
    if t_old is None:
        res["checks"]["old_member_pids_unchanged"] = False; res["checks"]["target_gpus_released"] = False
        return _finish(res, term)
    # Observation window: [REBUILT_OLD, REBUILT_OLD + gpu_release_s], cut short only by an ORDERLY end of the run
    # (every old-member process disappears in the same sample and stays gone = the learner finished its rounds).
    # A window shorter than 10 s is no observation; any old pid vanishing while others stay is a changed old set.
    old_set = {(u, p) for u, p in pre if u in old_gpus}
    t_eval, t_end = t_old + gpu_release_s, None
    tail = [s for s in gpu_samples if s.get("t", 0) >= t_old]
    for s in tail:
        now = {(a[0], a[1]) for a in (s.get("apps") or []) if len(a) >= 2 and a[0] in old_gpus}
        if not now and old_set:
            t_end = s["t"]; break
    if t_end is not None and t_end < t_eval:
        before_end = [s for s in tail if s["t"] < t_end]
        t_eval = before_end[-1]["t"] if before_end else t_old
        res["gpus"]["run_ended_at"] = round(t_end - t_old, 1)
    last = max(s.get("t", 0) for s in gpu_samples)
    if min(last, t_eval) < t_old + 10:
        return _invalid(res, "gpu samples cover only %.0fs after REBUILT_OLD (< 10 s): release not observed" % (min(last, t_eval) - t_old), "evidence_missing")
    after = _apps_at(gpu_samples, t_eval, before=True)
    res["checks"]["old_member_pids_unchanged"] = old_set == {(u, p) for u, p in after if u in old_gpus}
    # released at the evaluation point AND never busy again until the samples end (a target process that
    # outlives the learner is still not released)
    later_busy = {a[0] for s in gpu_samples if s.get("t", 0) >= t_eval for a in (s.get("apps") or []) if len(a) >= 2 and a[0] in target_gpus}
    res["checks"]["target_gpus_released"] = not later_busy
    res["gpus"]["after_pids"] = sorted(after); res["gpus"]["evaluated_at"] = round(t_eval - t_old, 1)
    # (5) the fork did not restart the killed cells (status probe after the terminal state)
    if probe_after is None:
        return _invalid(res, "fork status probe after the terminal state missing: (5) not observed", "evidence_missing")
    statuses = probe_after.get("cell_statuses") or {}
    tcells = [t.split(":", 1)[1] for t in targets]
    res["probe_after"] = {c: statuses.get(c) for c in tcells}
    res["checks"]["killed_cells_not_serving"] = bool(statuses) and all(c not in statuses or "Serving" not in str(statuses[c]) for c in tcells)
    return _finish(res, term)


# ----------------------------------------------------------------------------------------------- A4b
def judge_a4b(journal, tape, known, variant="cancel", probe_after=None, router_samples=None):
    """E1-C (c) via the tool-wait injection. variant 'cancel': drain timeout -> undrain -> CANCELLED, routing restored.
    variant 'recovery' (a4bu): undrain fails -> RECOVERY_REQUIRED, never CANCELLED."""
    res = {"case": "a4b" if variant == "cancel" else "a4bu", "checks": {}, "invalid_reasons": []}
    ev = named(tape, "test_injection") + named(journal, "test_injection")
    valid, why, good = injection_validity(ev, {"tool_wait"}, known)
    res["injection_valid"] = valid; res["invalid_reasons"] += why
    if not valid:
        return _invalid(res, "tool_wait injection not applied")
    dt = named(journal, "drain_timeout")
    res["drain_timeout"] = [{k: d.get(k) for k in ("active_requests", "tool_wait", "blockers")} for d in dt]
    if not dt or not any((d.get("tool_wait") or 0) > 0 for d in dt):
        return _invalid(res, "no drain_timeout record with tool_wait>0: the injected tool wait did not block the drain")
    tx = dt[-1].get("tx_id")
    ph = [r for r in phases(journal) if tx is None or r.get("tx_id") == tx]
    term = [r["phase"] for r in ph if r.get("phase") in TERMINAL]
    res["terminal"] = term
    ops = [r for r in named(journal, "fork_op") if r.get("tx_id") == tx]
    res["checks"]["no_stop_issued"] = not any(o.get("op") == "stop" for o in ops)
    if variant == "cancel":
        res["checks"]["terminal_CANCELLED"] = "CANCELLED" in term and "SUCCEEDED" not in term and "RECOVERY_REQUIRED" not in term
        res["checks"]["no_undrain_failure"] = not [r for r in named(journal, "undrain_failed") if r.get("tx_id") == tx]
        # routing restored after CANCELLED: from the router sampler (cordoned list empty at the end, the cordoned
        # engines serve requests again) and/or the fork status probe (every cell Serving)
        t_cancel = next((r.get("wall_time") for r in ph if r.get("phase") == "CANCELLED"), None)
        restored = None
        if router_samples and t_cancel is not None:
            win = [x for x in router_samples if x.get("data") and x["t"] >= t_cancel - 6]
            cord = [tuple(sorted(x["data"].get("cordoned") or [])) for x in win if x["t"] < t_cancel]
            was_cordoned = sorted({u for c in cord for u in c})
            post = [x for x in win if x["t"] >= t_cancel]
            served_again = {u for x in post for u, n in x["data"].get("inflight", {}).items() if u in was_cordoned and n}
            res["router"] = {"cordoned_during_drain": was_cordoned, "post_samples": len(post),
                             "final_cordoned": sorted(post[-1]["data"].get("cordoned") or []) if post else None,
                             "served_again": sorted(served_again)}
            if was_cordoned and len(post) >= 3:
                restored = not res["router"]["final_cordoned"] and set(served_again) == set(was_cordoned)
        if probe_after is not None:
            statuses = probe_after.get("cell_statuses") or {}
            res["probe_after"] = {"membership": probe_after.get("membership"), "statuses": statuses}
            mem = (probe_after.get("membership") or {}).get("members") or []
            p_ok = bool(statuses) and all("Serving" in str(v) for v in statuses.values()) and (
                not mem or len(mem) == len([v for v in statuses.values() if "Serving" in str(v)]))
            restored = p_ok if restored is None else (restored and p_ok)
        if restored is None:
            return _invalid(res, "neither router samples (cordon list before/after CANCELLED) nor the fork status probe observed the routing restoration", "evidence_missing")
        res["checks"]["routing_restored"] = restored
        return _finish(res, term)
    # recovery variant
    uf = [r for r in named(journal, "undrain_failed") if r.get("tx_id") == tx]
    inj = [e for e in ev if ikind(e) == "undrain_fail" and e.get("applied") is True]
    if not inj:
        return _invalid(res, "undrain_fail injection not applied")
    res["undrain_failed"] = [u.get("error") for u in uf]
    res["checks"]["undrain_failed_recorded"] = bool(uf)
    res["checks"]["terminal_RECOVERY_REQUIRED"] = "RECOVERY_REQUIRED" in term and "CANCELLED" not in term and "SUCCEEDED" not in term
    rec = [e for e in named(tape, "rl_reconfiguration") if e.get("result") == "RECOVERY_REQUIRED"]
    res["checks"]["driver_reported_recovery"] = bool(rec)
    res["verdict"] = "PASS" if all(v is True for v in res["checks"].values()) else "FAIL"
    if res["verdict"] == "FAIL": res["marker"] = "recovery_failed"
    return res


# ----------------------------------------------------------------------------------------------- E1-D
def _tx_terminal(journal, request_id):
    return [r["phase"] for r in phases(journal) if r.get("request_id") == request_id and r.get("phase") in TERMINAL]


def judge_d123(journal, tape, known, dkill_log=None):
    """E1-D ①②③ in one run: up1 SUCCEEDED; dn1 SUCCEEDED after one incomplete stop + retry; up2 REBUILT_OLD (new SGLang
    killed during start_cells); up3 REBUILT_OLD (new engine killed during publish_members)."""
    res = {"case": "d123", "checks": {}, "invalid_reasons": []}
    inc = [r for r in journal if r.get("kind") in ("incomplete", "stop_incomplete") or (r.get("kind") == "fork_op" and r.get("status") == "incomplete")]
    res["incomplete"] = [{k: r.get(k) for k in ("kind", "op", "status", "tx_id", "request_id")} for r in inc]
    # dkill.py log lines: {"event":"kill","rule":<name>,"gpu":i,"res":{pid:"killed"|error}}
    kills = [k for k in (dkill_log or []) if k.get("event") == "kill" and any(v == "killed" for v in (k.get("res") or {}).values())]
    res["dkill"] = kills
    res["checks"]["up1_SUCCEEDED"] = _tx_terminal(journal, "up1") == ["SUCCEEDED"]
    res["checks"]["dn1_SUCCEEDED_after_incomplete_stop"] = _tx_terminal(journal, "dn1") == ["SUCCEEDED"] and bool(inc)
    if not kills:
        return _invalid(res, "dkill log shows no kill: ①② not injected", "evidence_missing")
    names = {k.get("rule") for k in kills}
    res["checks"]["d1_injected"] = "d1" in names
    res["checks"]["d2_injected"] = "d2" in names
    res["checks"]["up2_REBUILT_OLD"] = _tx_terminal(journal, "up2") == ["REBUILT_OLD"]
    res["checks"]["up3_REBUILT_OLD"] = _tx_terminal(journal, "up3") == ["REBUILT_OLD"]
    term = [r["phase"] for r in phases(journal) if r.get("phase") in TERMINAL]
    # a transaction that was submitted but never reached a terminal phase before the run ended is not observed
    not_observed = [rid for rid in ("up2", "up3") if not _tx_terminal(journal, rid)
                    and any(r.get("request_id") == rid for r in phases(journal))]
    if not_observed:
        res["not_observed"] = not_observed
        return _invalid(res, "transactions %s never reached a terminal phase (run ended first): not observed" % not_observed, "evidence_missing")
    return _finish(res, term)


def judge_d2(journal, tape, known, dkill_log=None):
    """E1-D 2 alone: the new engine is killed while up1 is VERIFYING (publish_members) -> REBUILT_OLD, old members kept."""
    res = {"case": "d2", "checks": {}, "invalid_reasons": []}
    kills = [k for k in (dkill_log or []) if k.get("event") == "kill" and k.get("rule") == "d2" and any(v == "killed" for v in (k.get("res") or {}).values())]
    res["dkill"] = kills
    if not kills:
        return _invalid(res, "dkill log shows no d2 kill: 2 not injected", "evidence_missing")
    t_ver = next((r.get("wall_time") for r in phases(journal) if r.get("request_id") == "up1" and r.get("phase") == "VERIFYING"), None)
    res["checks"]["killed_during_verifying"] = t_ver is not None and any(k.get("wall", 0) >= t_ver - 1 for k in kills)
    term = _tx_terminal(journal, "up1")
    if not term:
        return _invalid(res, "up1 never reached a terminal phase (run ended first): not observed", "evidence_missing")
    res["checks"]["up1_REBUILT_OLD"] = term == ["REBUILT_OLD"]
    return _finish(res, term)


def judge_d4(journal, tape, known):
    """E1-D ④: stop keeps failing past T_recovery -> RECOVERY_REQUIRED, driver ends, no data consumed after."""
    res = {"case": "d4", "checks": {}, "invalid_reasons": []}
    inc = [r for r in journal if r.get("kind") in ("incomplete", "stop_incomplete") or (r.get("kind") == "fork_op" and r.get("status") == "incomplete")]
    if not inc:
        return _invalid(res, "no incomplete stop recorded: the stop-failure injection was not reached")
    res["checks"]["up1_SUCCEEDED"] = _tx_terminal(journal, "up1") == ["SUCCEEDED"]
    res["checks"]["dn1_RECOVERY_REQUIRED"] = _tx_terminal(journal, "dn1") == ["RECOVERY_REQUIRED"]
    rec = [e for e in named(tape, "rl_reconfiguration") if e.get("result") == "RECOVERY_REQUIRED"]
    res["checks"]["driver_reported_recovery"] = bool(rec)
    t_rec = next((r.get("wall_time") for r in phases(journal) if r.get("phase") == "RECOVERY_REQUIRED"), None)
    prepared_after = [e for e in tape if e.get("event") in ("rl_batch_prepared", "ledger_prepared") and t_rec is not None and (e.get("ts") or e.get("wall_time") or 0) > t_rec]
    res["checks"]["no_prepared_after_recovery"] = not prepared_after
    res["verdict"] = "PASS" if all(v is True for v in res["checks"].values()) else "FAIL"
    if res["verdict"] == "FAIL": res["marker"] = "recovery_failed"
    return res



# ----------------------------------------------------------------------------------------------- E1-D ⑤⑥⑦ (restart recovery, recovery-design.md §10.4)
def _driver_start_times(tape):
    return [e.get("time_unix") or e.get("ts") or 0 for e in named(tape, "rl_driver_start")]


def _restarted(tape, launch_log):
    """The learner process was restarted in place: a second rl_driver_start on the tape, or the restart loop's line."""
    starts = len(named(tape, "rl_driver_start"))
    loop = any("in-place restart" in l for l in (launch_log or []))
    return starts >= 2 or loop, {"driver_starts": starts, "restart_loop_line": loop}


def _after_restart(tape):
    """tape records after the second rl_driver_start (the restarted learner)."""
    idx = [i for i, e in enumerate(tape) if e.get("event") == "rl_driver_start"]
    return tape[idx[1]:] if len(idx) >= 2 else []


def _identity(tape):
    """Trainer identity across the restart: the first publication of the restarted learner carries the same policy
    token the run published for that policy version before (same version -> same hash). (None, why) when unobservable."""
    after = _after_restart(tape)
    pubs_after = named(after, "rl_publication")
    if not pubs_after:
        return None, "no publication after the restart"
    first = pubs_after[0]; v = first.get("policy_version"); tok = first.get("rl/policy_token")
    before = tape[: len(tape) - len(after)]
    earlier = [e.get("rl/policy_token") for e in named(before, "rl_publication") if e.get("policy_version") == v]
    if not earlier:
        return None, "no earlier publication of policy version %s to compare" % v
    return tok == earlier[-1], {"version": v, "token_after": tok, "token_before": earlier[-1]}


def _ledger_duplicates(ledger):
    """rollout ids recorded as outer_recorded more than once = trained/consumed twice."""
    seen = {}
    for r in ledger or []:
        if r.get("kind") == "outer_recorded":
            seen[r.get("rollout_id")] = seen.get(r.get("rollout_id"), 0) + 1
    return sorted(k for k, n in seen.items() if n > 1)


def _rounds_after(tape, since=None):
    """train phases of the restarted learner (optionally only after time `since`)."""
    after = _after_restart(tape)
    return [e for e in named(after, "rl_driver_phase") if e.get("phase") == "train"
            and (since is None or (e.get("time_unix") or e.get("ts") or 0) >= since)]


def _recovery_records(journal):
    return [r for r in journal if r.get("kind") == "recovery"]


def judge_recovery(case, journal, tape, known, ledger=None, launch_log=None, probe_after=None, min_rounds_after=3, up="up1", down="dn1"):
    """r5: up1 killed at COMMITTED -> restart -> committed members rebuilt (recovery verified), up1 SUCCEEDED(recovered).
    r7: up1 SUCCEEDED, then the learner is killed in steady state (fork epoch back to 0 vs journal 1) -> restart ->
        reconcile restore_membership_state + recovery verified -> dn1 SUCCEEDED afterwards (transactions go on).
    r6: up1 killed at QUIESCING -> restart -> CANCELLED, no recovery, rounds go on.
    r5c (regression of ruling (c)): up1 SUCCEEDED, dn1 killed at COMMITTED -> restart finds the startup shape ->
        dn1 SUCCEEDED(recovered), no recovery record.
    All: the learner really restarted (else INVALID), trainer identity across the restart, no rollout consumed twice,
    >= min_rounds_after train rounds after the restart/recovery."""
    res = {"case": case, "checks": {}, "invalid_reasons": []}
    restarted, how = _restarted(tape, launch_log); res["restart"] = how
    if not restarted:
        return _invalid(res, "the learner was never restarted (no second rl_driver_start, no restart-loop line): kill not applied", "evidence_missing")
    recs = _recovery_records(journal); res["recovery"] = [{k: r.get(k) for k in ("tx_id", "status", "attempt", "error")} for r in recs]
    four = sorted(known) if len(known) == 4 else None
    def tx_ok(rid, expect, recovered=None):
        term = _tx_terminal(journal, rid)
        if not term:
            return None
        ok = term == [expect]
        if recovered is not None:
            last = [r for r in phases(journal) if r.get("request_id") == rid and r.get("phase") == expect][-1]
            ok = ok and (last.get("recovered_after_restart") is True) == recovered
        return ok
    if case in ("r5", "r7", "r5c"):
        v = tx_ok(up, "SUCCEEDED", recovered=(case == "r5"))
        if v is None:
            return _invalid(res, up + " never reached a terminal phase (run ended first): not observed", "evidence_missing")
        res["checks"][up + "_SUCCEEDED" + ("_recovered_after_restart" if case == "r5" else "")] = v
    if case == "r5":
        res["checks"]["killed_at_COMMITTED"] = any(r.get("request_id") == up and r.get("phase") == "COMMITTED" for r in phases(journal))
    if case == "r6":
        v = tx_ok(up, "CANCELLED")
        if v is None:
            return _invalid(res, up + " never reached a terminal phase (run ended first): not observed", "evidence_missing")
        res["checks"][up + "_CANCELLED"] = v
        res["checks"]["killed_at_QUIESCING"] = any(r.get("request_id") == up and r.get("phase") == "QUIESCING" for r in phases(journal))
        res["checks"]["no_recovery"] = not recs
        res["checks"]["no_fork_op_after_restart"] = not any(r.get("kind") == "fork_op" and str(r.get("tx_id", "")).startswith("rec-") for r in journal)
    if case in ("r7", "r5c"):
        v = tx_ok(down, "SUCCEEDED", recovered=(case == "r5c"))
        if v is None:
            return _invalid(res, down + " never reached a terminal phase (run ended first): not observed", "evidence_missing")
        res["checks"][down + "_SUCCEEDED" + ("_recovered_after_restart" if case == "r5c" else "_after_recovery")] = v
    if case == "r5c":
        res["checks"]["no_recovery"] = not recs
        pubs = named(_after_restart(tape), "rl_publication")
        res["checks"]["startup_shape_after_restart"] = bool(pubs) and len(pubs[0].get("sync/publication_members") or []) == 2
    t_verified = None
    if case in ("r5", "r7"):
        statuses = [r.get("status") for r in recs]
        res["checks"]["recovery_sequence"] = statuses == ["planned", "membership_restored", "verified"]
        ver = next((r for r in recs if r.get("status") == "verified"), None)
        if ver is None:
            if any(r.get("status") == "failed" for r in recs) or "RECOVERY_REQUIRED" in [r.get("phase") for r in phases(journal)]:
                res["checks"]["recovery_verified"] = False
            else:
                return _invalid(res, "no recovery verified/failed record: the recovery was not observed to finish", "evidence_missing")
        else:
            t_verified = ver.get("wall_time")
            c = ver.get("checks") or {}
            res["verified_checks"] = c
            res["checks"]["recovery_verified"] = True
            res["checks"]["policy_token_verified"] = c.get("policy_token") == "verified"
            res["checks"]["router_all_admitted"] = isinstance(c.get("router"), dict) and c["router"].get("not_admitted") == []
            tl = c.get("trainer_layout")
            res["checks"]["trainer_layout_world_4"] = isinstance(tl, dict) and tl.get("world") == 4
            res["checks"]["no_unconsumed_batches"] = c.get("unconsumed_batches") == []
            res["checks"]["verified_members_are_the_committed_4"] = four is not None and sorted(ver.get("members") or []) == four
        rec_ev = [e for e in named(tape, "rl_reconfiguration") if e.get("result") == "RECOVERED"]
        res["checks"]["tape_RECOVERED"] = bool(rec_ev) and (four is None or sorted(rec_ev[-1].get("members") or []) == four)
        fork_ops = [r for r in journal if r.get("kind") == "fork_op" and str(r.get("tx_id", "")).startswith("rec-")]
        res["checks"]["recovery_started_the_missing_cells"] = any(r.get("op") == "start" and r.get("status") == "done" for r in fork_ops)
        if probe_after is not None:
            st = (probe_after.get("cell_statuses") or {})
            res["probe_after"] = {"membership": probe_after.get("membership"), "cell_statuses": st}
            res["checks"]["probe_fork_epoch_matches_verified"] = ver is not None and (probe_after.get("membership") or {}).get("epoch") == ver.get("fork_epoch")
            res["checks"]["probe_all_4_cells_running"] = sum(1 for v in st.values() if "Running" in str(v)) == 4
        else:
            res["probe_after"] = "missing (file evidence only)"
    if case == "r7":
        rc = [r for r in journal if r.get("kind") == "reconcile" and r.get("action") == "restore_membership_state"]
        res["checks"]["reconcile_restore_membership_state_epoch_1"] = any(int(r.get("epoch", -1)) == 1 for r in rc)
    rounds = _rounds_after(tape, t_verified)
    res["rounds_after"] = len(rounds)
    res["checks"]["rounds_after_restart_ge_%d" % min_rounds_after] = len(rounds) >= min_rounds_after
    ident, detail = _identity(tape); res["identity"] = detail
    if ident is None:
        return _invalid(res, "trainer identity across the restart not observable: %s" % detail, "evidence_missing")
    res["checks"]["trainer_identity_same_policy_hash"] = ident
    if ledger is None:
        return _invalid(res, "ledger journal missing: duplicate consumption not observable", "evidence_missing")
    dup = _ledger_duplicates(ledger); res["ledger_duplicates"] = dup
    res["checks"]["no_rollout_consumed_twice"] = not dup
    term = [r["phase"] for r in phases(journal) if r.get("phase") in TERMINAL]
    if "RECOVERY_REQUIRED" in term:
        res["checks"]["no_RECOVERY_REQUIRED"] = False
    res["verdict"] = "PASS" if all(v is True for v in res["checks"].values()) else "FAIL"
    if res["verdict"] == "FAIL": res["marker"] = "recovery_failed"
    return res

# ----------------------------------------------------------------------------------------------- E1-A (c)
def judge_e1a_c(journal, tape, expect, initial_members=None, round_is_rollout_id=True):
    """E1-A (c), re-based on rl_membership per round + retained rl_publication check (after the shrink, the next full publication has expect[-1] members).

    Units (erratum, gpu-plan-v2 9.23): the emitted rl_membership.round is a 0-based rollout_id (the first rollout that uses the membership), the plan's round r is
    1-based with rollout_id = r-1 (9.18).  So the members serving round r are those of the last event with round <= r-1.  Before any event the members are the
    initial config's (initial_members; default = members of the first rl_publication on the tape, else expect[0]).
    round_is_rollout_id=False reproduces the first (buggy) reading `round <= r`, kept only so the regression test can show that it fails on real data."""
    res = {"case": "e1a_c", "checks": {}}
    mem = [r for r in named(tape, "rl_membership") if r.get("round") is not None and r.get("members")]
    if not mem:
        res["verdict"] = "INVALID_TEST"; res["marker"] = "evidence_missing"; res["invalid_reasons"] = ["no rl_membership events with round/members on the tape"]; return res
    mem.sort(key=lambda r: (r["round"], r.get("config_epoch", 0)))
    if initial_members is None:
        p0 = next((p for p in tape if p.get("event") == "rl_publication"), None)
        pm = (p0 or {}).get("sync/publication_members", (p0 or {}).get("publication_members"))
        initial_members = (len(pm) if isinstance(pm, (list, tuple)) else pm) if p0 is not None and pm is not None else expect[0]
    got = []
    for rnd in range(1, len(expect) + 1):
        lim = rnd - 1 if round_is_rollout_id else rnd
        cur = [r for r in mem if r["round"] <= lim]
        got.append(len(cur[-1]["members"]) if cur else (initial_members if round_is_rollout_id else None))
    res["members_per_round"] = got; res["expected"] = expect; res["initial_members"] = initial_members; res["round_is_rollout_id"] = round_is_rollout_id
    res["checks"]["members_per_round_equal"] = got == expect
    downs = [r for r in mem if r.get("kind") == "down"]
    ok = False
    final_n = expect[-1]
    if downs:
        i_down = tape.index(downs[-1]) if downs[-1] in tape else None
        after = [p for p in (tape[i_down + 1:] if i_down is not None else []) if p.get("event") == "rl_publication"]
        if after:
            pm = after[0].get("sync/publication_members", after[0].get("publication_members"))
            res["first_publication_after_down_members"] = pm
            ok = (len(pm) if isinstance(pm, (list, tuple)) else pm) == final_n
    res["checks"]["publication_after_down_has_final_member_count"] = ok; res["final_member_count"] = final_n
    # LoRA admission on the up: every member publication read back adapter keys on every engine (never blind)
    rbs = named(journal, "lora_readback")
    if rbs:
        res["lora_readback"] = [{"targets": r.get("target_members"), "lora_keys": [e.get("lora_keys") for e in (r.get("engines") or [])], "blind": r.get("blind")} for r in rbs]
        res["checks"]["lora_readback_never_blind"] = all(not r.get("blind") and all((e.get("lora_keys") or 0) > 0 for e in (r.get("engines") or [])) for r in rbs)
    res["verdict"] = "PASS" if all(res["checks"].values()) else "FAIL"
    return res


def main(argv):
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("case"); ap.add_argument("journal"); ap.add_argument("tape")
    ap.add_argument("--router-samples"); ap.add_argument("--gpu-samples"); ap.add_argument("--probe-after"); ap.add_argument("--probe-stale")
    ap.add_argument("--probe-oldepoch"); ap.add_argument("--dkill-log"); ap.add_argument("--expect-members"); ap.add_argument("--cells", default="")
    ap.add_argument("--out"); ap.add_argument("--marker-dir")
    ap.add_argument("--launch-log"); ap.add_argument("--ledger")   # r5/r6/r7/r5c: restart-loop line, ledger journal
    a = ap.parse_args(argv)
    journal, tape = load(a.journal), load(a.tape)
    known = set(filter(None, a.cells.split(",")))
    for r in named(journal, "add_intent"): known |= set(r.get("members") or [])
    for r in named(tape, "rl_membership"): known |= set(r.get("members") or [])
    def jl(p):
        return [json.loads(l) for l in open(p) if l.strip()] if p and os.path.exists(p) else None
    samples = jl(a.router_samples)
    probes = {"after": load_probe(a.probe_after), "stale": load_probe(a.probe_stale), "oldepoch": load_probe(a.probe_oldepoch)}
    if a.case == "e1b": res = judge_e1b(journal, tape, known, samples, probes)
    elif a.case == "wd": res = judge_wd(journal, tape, known, jl(a.gpu_samples), probes["after"])
    elif a.case == "a4b": res = judge_a4b(journal, tape, known, "cancel", probes["after"], samples)
    elif a.case == "a4bu": res = judge_a4b(journal, tape, known, "recovery")
    elif a.case == "d123": res = judge_d123(journal, tape, known, jl(a.dkill_log))
    elif a.case == "d2": res = judge_d2(journal, tape, known, jl(a.dkill_log))
    elif a.case == "d4": res = judge_d4(journal, tape, known)
    elif a.case == "e1a_c": res = judge_e1a_c(journal, tape, [int(x) for x in a.expect_members.split(",")])
    elif a.case in ("r5", "r6", "r7", "r5c"):
        ll = open(a.launch_log).read().splitlines() if a.launch_log and os.path.exists(a.launch_log) else None
        res = judge_recovery(a.case, journal, tape, known, load(a.ledger) if a.ledger else None, ll, probes["after"])
    else: raise SystemExit("unknown case " + a.case)
    if a.out: Path(a.out).write_text(json.dumps(res, indent=1, default=str))
    if a.marker_dir and res.get("marker"):
        Path(a.marker_dir, res["marker"]).write_text(json.dumps({"case": a.case, "verdict": res["verdict"], "why": res.get("invalid_reasons") or res.get("terminal")}))
    print(json.dumps(res, default=str)); return EXIT[res["verdict"]]


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
