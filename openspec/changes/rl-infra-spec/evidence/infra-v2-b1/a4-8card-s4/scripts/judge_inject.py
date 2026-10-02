"""Structured judgement of the A4 fault-injection / membership cases from the journal, the event tape and the
in-container samplers/probes (never from the terminal state alone).

usage: judge_inject.py <case> <journal.jsonl|elastic-state dir> <tape.jsonl> [--router-samples F] [--gpu-samples F]
                       [--probe-after F] [--probe-stale F] [--probe-oldepoch F] [--dkill-log F] [--expect-members 1,1,3,3,1,1]
                       [--cells c0,c1,c2] [--out judgment.json] [--marker-dir DIR]
cases: e1a_c | e1b | wd | a4b | a4bu | a4bc | d123 | d4 | r5 | r6 | r7 | r5c   (r*: E1-D ⑤⑥⑦ with the 3.7 restart recovery, recovery-design.md §10.4)
  a4bc = a4b + --side-effects F (3.3 X5 (b): the tool side-effect journal, no replay after the cancel)
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
  journal phase REBUILD_OLD / REBUILT_OLD: cause (payload_mismatch, lora_unverifiable, stop_retry_deadline, ...), inconsistent_engines,
                deadline_wall, recovery_deadline_wall (= deadline_wall + T_recovery; 2026-10-02 ruling)
  journal phase RECOVERY_REQUIRED: scope=island (request_id=None) AND scope=request (request_id, cause, island_record_seq)
  journal drain_timeout: active_requests tool_wait blockers;  undrain_failed: error
  side_effects.jsonl (a4bc): kind=tool_side_effect|tool_complete seq trajectory_id tool_call_id wall_time attempt
"""
import json, os, re, sys
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


def judge_a4bc(journal, tape, known, side_effects, probe_after=None, router_samples=None):
    """3.3 X5 (b): a4b + the tool side-effect journal (tool_wait.ToolSideEffectLog, written by the injected tool
    *before* its wait starts). PASS needs the a4b verdict AND: the injection ran with the journal switched on; inside the
    cancelled transaction (request .. CANCELLED) every (trajectory_id, tool_call_id) has exactly one tool_side_effect
    record (>= 1 pair); after CANCELLED no tool_side_effect repeats a pair of that transaction (new pairs = new rollouts
    are allowed and listed); no pair is duplicated anywhere in the journal. tool_complete is informational (the island may
    end before the 30 s wait does). no_stop + CANCELLED alone never prove "no replay" (SESSION6 §10)."""
    res = judge_a4b(journal, tape, known, "cancel", probe_after, router_samples)
    res["case"] = "a4bc"
    if res["verdict"] == "INVALID_TEST" or res.get("marker") == "recovery_failed":   # no CANCELLED: nothing to window the journal on
        return res
    inj = [e for e in named(tape, "test_injection") + named(journal, "test_injection")
           if ikind(e) == "tool_wait" and e.get("applied") is True]
    if not any(e.get("side_effect_log") is True for e in inj):
        return _invalid(res, "the tool-wait injection ran without the side-effect journal (--rl-test-tool-side-effect-log not in effect)")
    recs = [r for r in (side_effects or []) if isinstance(r, dict)]
    if not recs:
        return _invalid(res, "side_effects.jsonl missing or empty: a replay of the tool call is not observable", "evidence_missing")
    tx = named(journal, "drain_timeout")[-1].get("tx_id")
    ph = [r for r in phases(journal) if tx is None or r.get("tx_id") == tx]
    t_cancel = next((r.get("wall_time") for r in ph if r.get("phase") == "CANCELLED"), None)
    t_req = next((r.get("wall_time") for r in named(journal, "request") if r.get("tx_id") == tx), None)
    if t_cancel is None or t_req is None:
        return _invalid(res, "request/CANCELLED wall_time of the cancelled transaction missing from the journal", "evidence_missing")
    se = [r for r in recs if r.get("kind") == "tool_side_effect"]
    def pair(r): return (str(r.get("trajectory_id")), str(r.get("tool_call_id")))
    def count(rs):
        out = {}
        for r in rs: out[pair(r)] = out.get(pair(r), 0) + 1
        return out
    in_tx = count(r for r in se if t_req <= float(r.get("wall_time", -1)) < t_cancel)
    after = [r for r in se if float(r.get("wall_time", -1)) >= t_cancel]
    replayed = sorted(p for p in count(after) if p in in_tx)
    new_after = sorted(p for p in count(after) if p not in in_tx)
    dup_all = sorted(p for p, n in count(se).items() if n > 1)
    inj_ids = sorted({e.get("tool_call_id") for e in inj if e.get("tool_call_id")})
    completes = count(r for r in recs if r.get("kind") == "tool_complete")
    res["side_effects"] = {"records": len(recs), "tool_side_effect": len(se), "in_tx_pairs": {f"{a}|{b}": n for (a, b), n in in_tx.items()},
                          "replayed_after_cancel": [f"{a}|{b}" for a, b in replayed], "new_pairs_after_cancel": [f"{a}|{b}" for a, b in new_after],
                          "duplicate_pairs": [f"{a}|{b}" for a, b in dup_all], "injection_tool_call_ids": inj_ids,
                          "tool_complete_pairs": {f"{a}|{b}": n for (a, b), n in completes.items()},
                          "t_request": t_req, "t_cancel": t_cancel, "seq_monotonic": [r.get("seq") for r in recs] == sorted(r.get("seq") for r in recs)}
    if not in_tx:
        return _invalid(res, "no tool_side_effect record inside the cancelled transaction: the injected tool never executed (or clocks disagree)")
    res["checks"]["one_side_effect_per_call_in_tx"] = all(n == 1 for n in in_tx.values())
    res["checks"]["no_replay_after_cancel"] = not replayed
    res["checks"]["no_duplicate_pairs"] = not dup_all
    res["checks"]["injection_matches_journal"] = bool(inj_ids) and set(inj_ids) <= {b for _a, b in in_tx}
    res["checks"]["seq_monotonic"] = res["side_effects"]["seq_monotonic"]
    res["verdict"] = "PASS" if all(v is True for v in res["checks"].values()) else "FAIL"
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


def _d4_times(e):
    return e.get("time_unix") or e.get("ts") or e.get("wall_time") or 0


def judge_d4(journal, tape, known, ledger=None, recovery_timeout_s=None, margin_s=60.0):
    """E1-D ④: stop keeps failing past the deadline -> RECOVERY_REQUIRED, driver ends, no data consumed after.

    Revision (2026-10-02 user ruling, chain 6r2 d4 FAIL): the original checks were
        up1_SUCCEEDED, dn1_RECOVERY_REQUIRED (request-level terminal), driver_reported_recovery, no_prepared_after_recovery
    and 6r2 failed only on dn1_RECOVERY_REQUIRED: the journal had the island-level RECOVERY_REQUIRED (request_id=None)
    but no request-level terminal for dn1, and the path went through REBUILD_OLD. Ruling (i): the controller now writes
    the request-level terminal too (scope=request, cause, island_record_seq); one bounded REBUILD_OLD after the deadline
    is allowed; the overall limit is deadline_s + recovery_timeout (the deadline is never reset). Checks now:
      up1_SUCCEEDED; dn1_RECOVERY_REQUIRED (request-level, exactly one); island_record_consistent (island-level record on
      the same tx with the same error/epochs); at_most_one_REBUILD_OLD; deadline_not_reset (REBUILD_OLD.deadline_wall ==
      request.deadline_wall when journaled); within_time_limit (request -> request-level RECOVERY_REQUIRED <= deadline_s +
      recovery_timeout + margin; recovery_timeout from --recovery-timeout-s, else REBUILD_OLD.recovery_deadline_wall, else
      the code default 900); driver_reported_recovery; no_prepared_after_recovery (tape rl_batch_prepared/ledger_prepared
      and ledger prepared/optimizer_applied after the RECOVERY_REQUIRED time)."""
    res = {"case": "d4", "checks": {}, "invalid_reasons": []}
    inc = [r for r in journal if r.get("kind") in ("incomplete", "stop_incomplete") or (r.get("kind") == "fork_op" and r.get("status") == "incomplete")]
    if not inc:
        return _invalid(res, "no incomplete stop recorded: the stop-failure injection was not reached")
    res["checks"]["up1_SUCCEEDED"] = _tx_terminal(journal, "up1") == ["SUCCEEDED"]
    req = next((r for r in journal if r.get("kind") == "request" and r.get("request_id") == "dn1"), None)
    tx_id = req.get("tx_id") if req else "dn1"
    res["checks"]["dn1_RECOVERY_REQUIRED"] = _tx_terminal(journal, "dn1") == ["RECOVERY_REQUIRED"]
    rq = [r for r in phases(journal) if r.get("request_id") == "dn1" and r.get("phase") == "RECOVERY_REQUIRED"]
    isl = [r for r in phases(journal) if r.get("tx_id") == tx_id and r.get("request_id") is None and r.get("phase") == "RECOVERY_REQUIRED"]
    res["checks"]["island_record_consistent"] = bool(isl) and bool(rq) and all(
        i.get("error") == rq[0].get("error") and i.get("config_epoch") == rq[0].get("config_epoch")
        and i.get("fork_epoch") == rq[0].get("fork_epoch") for i in isl[-1:])
    rebuilds = [r for r in phases(journal) if r.get("tx_id") == tx_id and r.get("phase") == "REBUILD_OLD"]
    res["checks"]["at_most_one_REBUILD_OLD"] = len(rebuilds) <= 1
    res["checks"]["deadline_not_reset"] = all(req is None or r.get("deadline_wall") in (None, req.get("deadline_wall")) for r in rebuilds)
    deadline_s = float((req or {}).get("body", {}).get("deadline_s") or 0.0)
    if recovery_timeout_s is None:
        rb = rebuilds[0] if rebuilds else {}
        if rb.get("recovery_deadline_wall") is not None and rb.get("deadline_wall") is not None:
            recovery_timeout_s = float(rb["recovery_deadline_wall"]) - float(rb["deadline_wall"])
        else:
            recovery_timeout_s = 900.0
    t_rec_rq = rq[0].get("wall_time") if rq else None
    t_rec_isl = isl[0].get("wall_time") if isl else None
    t_rec = min(t for t in (t_rec_rq, t_rec_isl) if t is not None) if (rq or isl) else None
    t_end = t_rec_rq if t_rec_rq is not None else t_rec_isl
    res["time"] = {"deadline_s": deadline_s, "recovery_timeout_s": recovery_timeout_s, "margin_s": margin_s,
                   "request_wall": (req or {}).get("wall_time"), "recovery_required_wall": t_end,
                   "elapsed_s": None if not (req and t_end) else t_end - req["wall_time"]}
    res["checks"]["within_time_limit"] = bool(req and t_end) and (t_end - req["wall_time"]) <= deadline_s + recovery_timeout_s + margin_s
    rec = [e for e in named(tape, "rl_reconfiguration") if e.get("result") == "RECOVERY_REQUIRED"]
    res["checks"]["driver_reported_recovery"] = bool(rec)
    prepared_after = [e for e in tape if e.get("event") in ("rl_batch_prepared", "ledger_prepared") and t_rec is not None and _d4_times(e) > t_rec]
    prepared_after += [r for r in (ledger or []) if r.get("kind") in ("prepared", "optimizer_applied") and t_rec is not None and _d4_times(r) > t_rec]
    res["checks"]["no_prepared_after_recovery"] = t_rec is not None and not prepared_after
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
    ap.add_argument("--launch-log"); ap.add_argument("--ledger")   # r5/r6/r7/r5c/d4: restart-loop line, ledger journal
    ap.add_argument("--recovery-timeout-s", type=float); ap.add_argument("--margin-s", type=float, default=60.0)   # d4 (2026-10-02 ruling)
    ap.add_argument("--side-effects")   # a4bc: elastic-state/side_effects.jsonl (tool_wait.ToolSideEffectLog)
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
    elif a.case == "a4bc": res = judge_a4bc(journal, tape, known, jl(a.side_effects), probes["after"], samples)
    elif a.case == "d123": res = judge_d123(journal, tape, known, jl(a.dkill_log))
    elif a.case == "d2": res = judge_d2(journal, tape, known, jl(a.dkill_log))
    elif a.case == "d4": res = judge_d4(journal, tape, known, load(a.ledger) if a.ledger else None, a.recovery_timeout_s, a.margin_s)
    elif a.case == "e1a_c": res = judge_e1a_c(journal, tape, [int(x) for x in a.expect_members.split(",")])
    elif a.case in ("r5", "r6", "r7", "r5c"):
        ll = open(a.launch_log).read().splitlines() if a.launch_log and os.path.exists(a.launch_log) else None
        res = judge_recovery(a.case, journal, tape, known, load(a.ledger) if a.ledger else None, ll, probes["after"])
    else: raise SystemExit("unknown case " + a.case)
    if a.out: Path(a.out).write_text(json.dumps(res, indent=1, default=str))
    if a.marker_dir and res.get("marker"):
        Path(a.marker_dir, res["marker"]).write_text(json.dumps({"case": a.case, "verdict": res["verdict"], "why": res.get("invalid_reasons") or res.get("terminal")}))
    print(json.dumps(res, default=str)); return EXIT[res["verdict"]]



# ---------------------------------------------------------------------------------------------------------
# A27 (chain 6r1/6r2 d2, revised after 8/1r5 d2): the new engine SIGKILLed during VERIFYING must be *discovered* by
# either legitimate path -- C: fork workers_lost / publisher target_workers_lost / watchdog skipped[workers_lost]
# in the journal; B: the trainer's fail-fast connect (RolloutEngineJoinError "failed to join weight update group")
# surfacing as the REBUILD_OLD cause (journal phase record `cause=update_failed, error=...`, mirrored by the tape's
# rl_reconfiguration). The transaction must go REBUILD_OLD -> REBUILT_OLD, the old members must be serving the same
# version with no cordoned residue on the router, and training must continue afterwards (>= 2 further rounds
# trained and >= 2 further publications with increasing policy_version). A bare PublicationError (REBUILT_OLD without
# the continuation) is not a pass.
# Evidence trade-off (1r5 d2 was judged INVALID although the island demonstrably died): a proven failure beats
# missing evidence. When the post-terminal fork probe is missing, old-member recovery is read from the journal
# (REBUILT_OLD lists the targets as inconsistent_engines) + the tape (>= A27_MIN_ROUNDS_AFTER rl_publication events
# after REBUILT_OLD whose sync/publication_members are exactly the old members, i.e. every old member took every
# later version) + the ledger (L3, added by the ledger wrapper); the launch log (when given) records old members
# killed after the terminal state / a dead trainer. INVALID(evidence_missing) is kept only when neither the probe
# nor that fallback can say anything *and* no failure is established. Appended only; judge_d2 keeps its original
# checks and the new ones are added to the same result (verdict = all checks true).
A27_MIN_ROUNDS_AFTER = 2
A27_FAIL_FAST_RE = re.compile(r"failed to join weight update group|RolloutEngineJoinError|ExternalFailureError"
                              r"|update_weights failed on the rollout engine side")
_A27_EXTRA = {"probe_after": None, "router_samples": None, "launch_log": None}


def _a27_terminal_time(journal, request_id="up1"):
    for r in phases(journal):
        if r.get("request_id") == request_id and r.get("phase") == "REBUILT_OLD":
            return r.get("wall_time")
    return None


def _a27_members(v):
    """sync/publication_members is a list in the tape; tolerate its str() form in older tapes."""
    if isinstance(v, str):
        return set(re.findall(r"engine:[\w\-]+", v))
    return set(v or [])


def _a27_old_member_fate(launch_log, old_cells, t_old):
    """Launch-log facts after the terminal state: old member cells killed (rollout_server 'Killing server') and a dead
    trainer ('Cannot recover when all cells are dead'); None when no launch log was given."""
    if launch_log is None:
        return None
    killed, trainer_dead = [], False
    seen_terminal = t_old is None
    for l in launch_log:
        if not seen_terminal:
            seen_terminal = "REBUILT_OLD" in l
            if not seen_terminal:
                continue
        if "Cannot recover when all cells are dead" in l:
            trainer_dead = True
        m = re.search(r"Killing server cell_id='([^']+)'", l)
        if m and m.group(1) in old_cells and m.group(1) not in killed:
            killed.append(m.group(1))
    return {"old_members_killed_after_terminal": killed, "trainer_dead": trainer_dead}


def judge_d2_a27(res, journal, tape, known, probe_after=None, router_samples=None, launch_log=None):
    """Add the A27 recovery checks to a judge_d2 result (see the block comment above)."""
    if res.get("verdict") == "INVALID_TEST":
        return res
    ph = [r for r in phases(journal) if r.get("request_id") == "up1"]
    seq = [r.get("phase") for r in ph]
    res["a27_phases"] = seq
    targets = set()
    for r in named(journal, "add_intent"):
        targets |= set(r.get("members") or [])
    tcells = [t.split(":", 1)[1] if ":" in t else t for t in sorted(targets)]
    # (1) discovery, either path. C: the publisher's watch (target_workers_lost) or the watchdog's workers_lost skip.
    # B: the trainer's fail-fast connect is the REBUILD_OLD cause (journal phase record; the tape mirrors it).
    lost = named(journal, "target_workers_lost")
    wd_lost = [e for r in named(journal, "watchdog_action") for e in (r.get("skipped") or [])
               if e.get("kind") == "workers_lost"]
    rebuild = [r for r in ph if r.get("phase") == "REBUILD_OLD"]
    tape_reconf = [e for e in (tape or []) if e.get("event") == "rl_reconfiguration"]
    fail_fast = [{"source": src, "cause": r.get("cause"), "inconsistent_engines": r.get("inconsistent_engines"),
                  "error": str(r.get("error"))[:200]}
                 for src, rs in (("journal", rebuild), ("tape", tape_reconf)) for r in rs
                 if r.get("cause") == "update_failed" and A27_FAIL_FAST_RE.search(str(r.get("error") or ""))]
    res["a27_discovery"] = {"target_workers_lost": [r.get("lost_members") for r in lost],
                            "watchdog_workers_lost": [e.get("cell") for e in wd_lost],
                            "fail_fast": fail_fast,
                            "path": [p for p, on in (("workers_lost", bool(lost) or bool(wd_lost)),
                                                    ("fail_fast", bool(fail_fast))) if on]}
    res["checks"]["a27_failure_detected"] = bool(lost) or bool(wd_lost) or bool(fail_fast)
    # (2) REBUILD_OLD -> REBUILT_OLD, nothing else terminal
    res["checks"]["a27_rebuild_old_then_rebuilt_old"] = ("REBUILD_OLD" in seq and seq[-1] == "REBUILT_OLD"
                                                        and "RECOVERY_REQUIRED" not in seq)
    t_old = _a27_terminal_time(journal)
    # (5) training continues: >= A27_MIN_ROUNDS_AFTER rounds trained and publications with increasing versions
    after = [e for e in tape if t_old is not None and e.get("time_unix", 0) >= t_old]
    before = [e for e in tape if t_old is None or e.get("time_unix", 0) < t_old]
    rounds = sorted({e.get("rollout_id") for e in after if e.get("event") == "rl_round_trained"
                     if e.get("rollout_id") is not None})
    pubs_after = [e for e in after if e.get("event") == "rl_publication"]
    pubs = [e.get("policy_version") for e in pubs_after]
    syncs = [e for e in after if e.get("event") == "rl_driver_phase" and e.get("phase") == "sync"]
    res["a27_after_terminal"] = {"rounds_trained": rounds, "publication_versions": pubs, "syncs": len(syncs)}
    res["checks"]["a27_training_continued"] = (
        len(rounds) >= A27_MIN_ROUNDS_AFTER and len(pubs) >= A27_MIN_ROUNDS_AFTER
        and all(b > a for a, b in zip(pubs, pubs[1:])) and len(syncs) >= A27_MIN_ROUNDS_AFTER)
    # old members = the members of the last publication before up1's terminal state, minus the targets
    old_pubs = [e for e in before if e.get("event") == "rl_publication"]
    old_members = (_a27_members(old_pubs[-1].get("sync/publication_members")) if old_pubs else set()) - targets
    old_cells = {m.split(":", 1)[1] if m.startswith("engine:") else m for m in old_members}
    fate = _a27_old_member_fate(launch_log, old_cells, t_old)
    if fate is not None:
        res["a27_old_member_fate"] = fate
        res["checks"]["a27_old_members_survived"] = not fate["old_members_killed_after_terminal"] and not fate["trainer_dead"]
    failure_established = (res["checks"]["a27_training_continued"] is False
                           or res["checks"].get("a27_old_members_survived") is False)
    # (3) old members recovered: fork status probe after the terminal state, else the journal+tape fallback
    if probe_after is not None:
        statuses = probe_after.get("cell_statuses") or {}
        vers = probe_after.get("versions") or {}
        serving = {k: v for k, v in vers.items() if k not in tcells}
        res["a27_probe_after"] = {"source": "probe_after", "membership": probe_after.get("membership"),
                                  "versions": vers, "target_statuses": {c: statuses.get(c) for c in tcells}}
        res["checks"]["a27_old_members_same_version"] = len(serving) >= 1 and len(set(serving.values())) == 1
        res["checks"]["a27_targets_not_serving"] = bool(statuses) and all(
            c not in statuses or "Serving" not in str(statuses[c]) for c in tcells)
    else:
        rebuilt = [r for r in ph if r.get("phase") == "REBUILT_OLD"]
        inconsistent = set(rebuilt[-1].get("inconsistent_engines") or []) if rebuilt else set()
        members_after = [sorted(_a27_members(e.get("sync/publication_members"))) for e in pubs_after]
        res["a27_probe_after"] = {"source": "journal+tape (probe_after missing)", "old_members": sorted(old_members),
                                  "rebuilt_old_inconsistent_engines": sorted(inconsistent),
                                  "publication_members_after": members_after}
        if not old_members and not failure_established:
            return _invalid(res, "fork status probe after the terminal state missing and the tape names no old "
                                 "members: old-member recovery not observed", "evidence_missing")
        # every later version reached every old member and nobody else: same version on all old members
        res["checks"]["a27_old_members_same_version"] = (
            len(members_after) >= A27_MIN_ROUNDS_AFTER and all(set(m) == old_members for m in members_after))
        # the targets were marked inconsistent at REBUILT_OLD and took part in no later publication
        res["checks"]["a27_targets_not_serving"] = (bool(targets) and targets <= inconsistent
                                                    and all(not (set(m) & targets) for m in members_after))
    # (4) router: no cordoned residue after the terminal state (last >= 3 samples)
    data = [x for x in (router_samples or []) if isinstance(x.get("data"), dict)
            and (t_old is None or x.get("t", 0) >= t_old)]
    if len(data) < 3:
        if not failure_established:
            return _invalid(res, "fewer than 3 router samples after REBUILT_OLD: cordon residue not observed",
                            "evidence_missing")
        res["a27_router_tail"] = {"samples_after_terminal": len(data), "note": "too few samples; the island failed before"}
    else:
        res["a27_router_tail"] = [x["data"].get("cordoned") for x in data[-3:]]
        res["checks"]["a27_router_no_cordoned_residue"] = all(not x["data"].get("cordoned") for x in data[-3:])
    res["verdict"] = "PASS" if all(v is True for v in res["checks"].values()) else "FAIL"
    if res["verdict"] == "FAIL":
        res["marker"] = "recovery_failed"
    return res


_judge_d2_base = judge_d2


def judge_d2(journal, tape, known, dkill_log=None, probe_after=None, router_samples=None, launch_log=None):  # noqa: F811
    res = _judge_d2_base(journal, tape, known, dkill_log)
    return judge_d2_a27(res, journal, tape, known,
                        probe_after if probe_after is not None else _A27_EXTRA["probe_after"],
                        router_samples if router_samples is not None else _A27_EXTRA["router_samples"],
                        launch_log if launch_log is not None else _A27_EXTRA["launch_log"])


_main_base = main


def main(argv):  # noqa: F811
    """Stash --probe-after / --router-samples for the appended judge_d2 (main's d2 dispatch passes neither)."""
    import argparse
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--probe-after"); ap.add_argument("--router-samples"); ap.add_argument("--launch-log")
    a, _ = ap.parse_known_args(argv)
    _A27_EXTRA["probe_after"] = load_probe(a.probe_after)
    _A27_EXTRA["launch_log"] = (open(a.launch_log, errors="replace").read().splitlines()
                                if a.launch_log and os.path.exists(a.launch_log) else None)
    _A27_EXTRA["router_samples"] = ([json.loads(l) for l in open(a.router_samples) if l.strip()]
                                    if a.router_samples and os.path.exists(a.router_samples) else None)
    return _main_base(argv)

# ------------------------------------------------------------------------------------------------------- 3.6 ledger
# T36-REVIEW §3.2 L1-L8 (2026-10-02). Appended only: the judges above keep their checks; the wrappers below add the
# ledger checks to d4 / d2 / e1b / wd / r5-r6-r7-r5c and recompute the verdict. `ledger` None = not provided (CPU
# synthetic tests: "not provided", no ledger checks); [] = the --ledger file is missing/empty -> INVALID(evidence_missing).
# §5.1: judge_d4's old no_prepared_after_recovery read tape events the driver never emits (always True); L4 reads the
# ledger. §5.2: _ledger_duplicates is group-level and attempt-aware now (L2).
LEDGER_TRIPLE = ("prepared", "optimizer_applied", "outer_recorded")
_LEDGER_EXTRA = {"ledger": None}


def _lt(r):
    return r.get("wall_time") or r.get("time_unix") or r.get("ts") or 0


def _by_rollout(ledger):
    by = {}
    for r in ledger:
        rid = r.get("rollout_id")
        if rid is None or rid < 0:
            continue
        by.setdefault(rid, []).append(r)
    return by


def _complete_rounds(ledger):
    """rollout_id -> time of the prepared of its last attempt, for rollouts whose last attempt reached outer_recorded."""
    out = {}
    for rid, recs in _by_rollout(ledger).items():
        att = max(r.get("attempt") or 0 for r in recs)
        last = [r for r in recs if (r.get("attempt") or 0) == att]
        kinds = [r["kind"] for r in last]
        if "prepared" in kinds and "optimizer_applied" in kinds and "outer_recorded" in kinds:
            out[rid] = _lt(next(r for r in last if r["kind"] == "prepared"))
    return out


def ledger_L1_round_triples(ledger, tape):
    """L1: per rollout the last attempt has exactly one prepared, optimizer_applied and outer_recorded; every earlier
    attempt has one prepared, at most one optimizer_applied, no outer_recorded and ends with discarded/superseded;
    attempts increase; seq strictly increasing; rollout ids contiguous 0..N-1; tape rl_round_trained count == number of
    optimizer_applied (a superseded attempt was trained too). A trailing prepared with no train yet (process ended
    between generate and train) is reported as `dangling_prepared` and left to L3."""
    issues = []
    seqs = [r.get("seq") for r in ledger if r.get("seq") is not None]
    if seqs != sorted(seqs) or len(set(seqs)) != len(seqs):
        issues.append("seq not strictly increasing")
    by = _by_rollout(ledger)
    rids = sorted(by)
    dangling = None
    for rid in rids:
        recs = by[rid]
        attempts = sorted({r.get("attempt") or 0 for r in recs})
        if attempts != list(range(attempts[0], attempts[0] + len(attempts))):
            issues.append("rollout %s: attempts %r not consecutive" % (rid, attempts))
        for i, att in enumerate(attempts):
            kinds = [r["kind"] for r in recs if (r.get("attempt") or 0) == att]
            n = {k: kinds.count(k) for k in ("prepared", "optimizer_applied", "outer_recorded", "discarded", "superseded")}
            last = i == len(attempts) - 1
            if last and rid == rids[-1] and n["prepared"] == 1 and n["optimizer_applied"] == 0 and n["outer_recorded"] == 0 and n["discarded"] + n["superseded"] == 0:
                dangling = rid
                continue
            if n["prepared"] != 1:
                issues.append("rollout %s attempt %s: prepared x%d" % (rid, att, n["prepared"]))
            if last:
                if n["optimizer_applied"] != 1 or n["outer_recorded"] != 1:
                    issues.append("rollout %s attempt %s: optimizer_applied x%d outer_recorded x%d" % (rid, att, n["optimizer_applied"], n["outer_recorded"]))
            else:
                if n["optimizer_applied"] > 1 or n["outer_recorded"] or kinds[-1] not in ("discarded", "superseded"):
                    issues.append("rollout %s attempt %s: not closed by discarded/superseded (kinds %r)" % (rid, att, kinds))
    complete = [rid for rid in rids if rid != dangling]
    if complete != list(range(len(complete))):
        issues.append("rollout ids not contiguous: %r" % complete[:12])
    n_trained = len(named(tape, "rl_round_trained"))
    n_applied = sum(1 for r in ledger if r.get("kind") == "optimizer_applied")
    if n_trained and n_trained != n_applied:
        issues.append("tape rl_round_trained=%d, ledger optimizer_applied=%d" % (n_trained, n_applied))
    return not issues, {"rollouts": len(complete), "dangling_prepared": dangling, "tape_rounds_trained": n_trained,
                        "optimizer_applied": n_applied, "issues": issues}


def ledger_L2_group_reuse(ledger):
    """L2: the group_ids of every consumed batch (prepared of the same rollout+attempt as an optimizer_applied) are
    pairwise disjoint across rollouts, attempts and restarts. Returns (ok, {"duplicates": [group ids]})."""
    prepared = {}
    for r in ledger:
        if r.get("kind") == "prepared":
            prepared[(r.get("rollout_id"), r.get("attempt"))] = r.get("group_ids") or []
    seen, dup = {}, set()
    for r in ledger:
        if r.get("kind") != "optimizer_applied":
            continue
        for g in prepared.get((r.get("rollout_id"), r.get("attempt")), []):
            if g in seen and seen[g] != (r.get("rollout_id"), r.get("attempt")):
                dup.add(g)
            seen.setdefault(g, (r.get("rollout_id"), r.get("attempt")))
    return not dup, {"duplicates": sorted(dup), "consumed_groups": len(seen)}


def ledger_L3_continues_after(ledger, t, min_rounds_after):
    """L3: >= min_rounds_after complete rollouts prepared after time t; the last rollout before t ended outer_recorded,
    or its prepared was discarded/superseded later and its groups reappear in a later prepared (nothing lost)."""
    if t is None:
        return False, {"error": "terminal time unknown"}
    complete = _complete_rounds(ledger)
    after = sorted(rid for rid, tp in complete.items() if tp > t)
    info = {"rounds_after": after, "min_rounds_after": min_rounds_after}
    before = [r for r in ledger if _lt(r) <= t and r.get("rollout_id") is not None and r.get("rollout_id") >= 0]
    ok_before = True
    if before:
        last = max(r["rollout_id"] for r in before)
        kinds = [r["kind"] for r in before if r["rollout_id"] == last]
        info["last_before"] = {"rollout_id": last, "kinds": kinds}
        if "outer_recorded" not in kinds:
            later = [r for r in ledger if r.get("rollout_id") == last and _lt(r) > t]
            closed = any(r["kind"] in ("discarded", "superseded", "outer_recorded") for r in later)
            groups = set()
            for r in before:
                if r["rollout_id"] == last and r["kind"] == "prepared":
                    groups |= set(r.get("group_ids") or [])
            reappear = set()
            for r in ledger:
                if r.get("kind") == "prepared" and _lt(r) > t:
                    reappear |= set(r.get("group_ids") or [])
            lost = sorted(groups - reappear)
            info["last_before"]["closed_later"] = closed; info["last_before"]["groups_lost"] = lost
            ok_before = closed and (not groups or not lost) if "optimizer_applied" not in kinds else closed
    return len(after) >= min_rounds_after and ok_before, info


def ledger_L4_silent_after(ledger, tape, t_rec):
    """L4: after RECOVERY_REQUIRED (t_rec) the ledger has no record at all and the tape no rl_round_trained and no
    rl_driver_phase train/publish."""
    if t_rec is None:
        return False, {"error": "RECOVERY_REQUIRED time unknown"}
    led = [(r.get("kind"), r.get("rollout_id")) for r in ledger if _lt(r) > t_rec]
    tp = [(e.get("event"), e.get("phase"), e.get("rollout_id")) for e in tape if _lt(e) > t_rec
          and (e.get("event") == "rl_round_trained" or (e.get("event") == "rl_driver_phase" and e.get("phase") in ("train", "publish")))]
    return not led and not tp, {"ledger_after": led[:10], "tape_after": tp[:10], "t_rec": t_rec}


def ledger_L5_silent_during_transactions(ledger, journal, tape):
    """L5: inside every transaction's destructive window (first TRANSFERRING/INITIALIZING/REBUILD_OLD record ->
    terminal record) the ledger is silent (transactions run at a safe point); after a REBUILT_OLD the first ledger
    record is a prepared."""
    windows, noisy, first_after = [], [], []
    for tx in {r.get("tx_id") for r in phases(journal) if r.get("request_id")}:
        ph = [r for r in phases(journal) if r.get("tx_id") == tx]
        start = next((r for r in ph if r.get("phase") in ("TRANSFERRING", "INITIALIZING", "REBUILD_OLD")), None)
        term = next((r for r in ph if r.get("phase") in TERMINAL), None)
        if start is None or term is None:
            continue
        if start.get("wall_time") is None or term.get("wall_time") is None:
            continue  # synthetic records without times: no window to judge
        t0, t1 = _lt(start), _lt(term)
        windows.append({"tx_id": tx, "from": t0, "to": t1, "terminal": term.get("phase")})
        noisy += [(r.get("kind"), r.get("rollout_id")) for r in ledger if t0 <= _lt(r) <= t1]
        if term.get("phase") == "REBUILT_OLD":
            nxt = next((r for r in ledger if _lt(r) > t1), None)
            first_after.append(nxt.get("kind") if nxt else None)
    ok = not noisy and all(k in (None, "prepared") for k in first_after)
    return ok, {"windows": windows, "ledger_in_window": noisy[:10], "first_kind_after_rebuilt_old": first_after}


def ledger_L6_restart_consistent(ledger, tape):
    """L6 (r cases): after the second rl_driver_start, a leading superseded/discarded names the restart rollout
    (restart_rollout_id or 'restart at rollout N') equal to the first rl_publication.policy_version after the restart;
    the first prepared after the restart has that rollout_id; any outer_recorded before it is recovered=true."""
    starts = sorted(_lt(e) for e in named(tape, "rl_driver_start"))
    if len(starts) < 2:
        return None, {"restarts": len(starts) - 1}
    t_restart = starts[1]
    pubs = [e.get("policy_version") for e in named(tape, "rl_publication") if _lt(e) >= t_restart]
    after = [r for r in ledger if _lt(r) > t_restart]
    info = {"t_restart": t_restart, "first_publication_version": pubs[0] if pubs else None,
            "first_kinds_after": [r.get("kind") for r in after[:4]]}
    if not pubs or not after:
        return False, dict(info, error="no publication or no ledger record after the restart")
    v = pubs[0]
    ok = True
    for r in after:
        if r["kind"] == "prepared":
            info["first_prepared_rollout_id"] = r.get("rollout_id")
            ok = ok and r.get("rollout_id") == v
            break
        if r["kind"] in ("superseded", "discarded"):
            rr = r.get("restart_rollout_id")
            if rr is None and "restart at rollout" in str(r.get("error") or ""):
                try: rr = int(str(r["error"]).rsplit(" ", 1)[1])
                except Exception: rr = None
            info.setdefault("restart_rollout_ids", []).append(rr)
            ok = ok and rr == v
        elif r["kind"] == "outer_recorded":
            ok = ok and bool(r.get("recovered"))
    return ok, info


def ledger_L7_verified_unconsumed(journal):
    """L7: every recovery `verified` record has checks.unconsumed_batches == []."""
    ver = [r for r in journal if r.get("kind") == "recovery" and r.get("status") == "verified"]
    if not ver:
        return None, {"verified_records": 0}
    bad = [r.get("tx_id") for r in ver if (r.get("checks") or {}).get("unconsumed_batches") not in ([], None) or "unconsumed_batches" not in (r.get("checks") or {})]
    return not bad, {"verified_records": len(ver), "with_unconsumed_or_missing": bad}


def ledger_L8_report_fields(ledger):
    """L8: each prepared is followed by a carried_over_report of the same rollout/attempt; every filtered record names a
    mechanism; sums of filtered.groups / engine_discarded.groups and carried_over_reported are recorded (not judged)."""
    reports = {(r.get("rollout_id"), r.get("attempt")) for r in ledger if r.get("kind") == "carried_over_report"}
    missing = [(r.get("rollout_id"), r.get("attempt")) for r in ledger if r.get("kind") == "prepared" and (r.get("rollout_id"), r.get("attempt")) not in reports]
    filt = [r for r in ledger if r.get("kind") == "filtered"]
    no_mech = [r.get("rollout_id") for r in filt if not (r.get("detail") or {}).get("mechanism")]
    info = {"prepared_without_report": missing, "filtered_without_mechanism": no_mech,
            "filtered_groups": sum(int((r.get("detail") or {}).get("groups") or 0) for r in filt),
            "engine_discarded_groups": sum(int(r.get("groups") or 0) for r in ledger if r.get("kind") == "engine_discarded"),
            "carried_over_reported_all": all(r.get("carried_over_reported") for r in ledger if r.get("kind") == "carried_over_report") if reports else None}
    return not missing and not no_mech, info


def ledger_checks(ledger, tape, journal, *, continues_after=None, min_rounds_after=0, silent_after=None,
                  restart=False, verified=False, tx_windows=True):
    """L1-L8 on one run. Returns {"checks": {name: bool}, "info": {...}}; checks that do not apply are left out."""
    checks, info = {}, {}
    def put(name, pair):
        ok, detail = pair
        info[name] = detail
        if ok is not None:
            checks[name] = bool(ok)
    put("L1_round_triples", ledger_L1_round_triples(ledger, tape))
    put("L2_no_group_reuse", ledger_L2_group_reuse(ledger))
    if continues_after is not None or min_rounds_after:
        put("L3_continues_after_terminal", ledger_L3_continues_after(ledger, continues_after, min_rounds_after))
    if silent_after is not None:
        put("L4_silent_after_recovery", ledger_L4_silent_after(ledger, tape, silent_after))
    if tx_windows:
        put("L5_silent_during_transactions", ledger_L5_silent_during_transactions(ledger, journal, tape))
    if restart:
        put("L6_restart_consistent", ledger_L6_restart_consistent(ledger, tape))
    if verified:
        put("L7_verified_unconsumed_empty", ledger_L7_verified_unconsumed(journal))
    put("L8_report_fields", ledger_L8_report_fields(ledger))
    return {"checks": checks, "info": info}


def _ledger_duplicates(ledger):  # noqa: F811 - L2 (group-level, attempt-aware) replaces the rollout-level check
    return ledger_L2_group_reuse(ledger or [])[1]["duplicates"]


def _with_ledger(res, ledger, tape, journal, required=False, **kw):
    """Merge ledger_checks into a judge result and recompute the verdict (INVALID results are left alone)."""
    if res.get("verdict") == "INVALID_TEST":
        return res
    if ledger is None:
        if required:
            return _invalid(res, "ledger journal not provided: ledger evidence missing", "evidence_missing")
        res["ledger"] = "not provided"; return res
    if not ledger:
        return _invalid(res, "ledger journal missing or empty (--ledger): ledger evidence missing", "evidence_missing")
    lc = ledger_checks(ledger, tape, journal, **kw)
    res["checks"].update(lc["checks"]); res["ledger"] = lc["info"]
    res["verdict"] = "PASS" if all(v is True for v in res["checks"].values()) else "FAIL"
    if res["verdict"] == "FAIL" and "marker" not in res:
        res["marker"] = "recovery_failed"
    return res


def _terminal_time(journal, request_id, phase):
    return next((r.get("wall_time") for r in phases(journal) if r.get("request_id") == request_id and r.get("phase") == phase), None)


_judge_d4_base, _judge_d2_a27, _judge_e1b_base, _judge_wd_base, _judge_recovery_base = judge_d4, judge_d2, judge_e1b, judge_wd, judge_recovery


def judge_d4(journal, tape, known, ledger=None, recovery_timeout_s=None, margin_s=60.0):  # noqa: F811
    """d4 + L1/L2/L4/L8 (ledger required: it is the evidence that nothing was consumed after RECOVERY_REQUIRED)."""
    res = _judge_d4_base(journal, tape, known, ledger, recovery_timeout_s, margin_s)
    rec = [r.get("wall_time") for r in phases(journal) if r.get("phase") == "RECOVERY_REQUIRED" and r.get("wall_time") is not None]
    return _with_ledger(res, ledger, tape, journal, required=True, silent_after=min(rec) if rec else None, tx_windows=False)


def judge_d2(journal, tape, known, dkill_log=None, probe_after=None, router_samples=None, ledger=None, launch_log=None):  # noqa: F811
    """d2 + A27 + L1/L2/L3(>=2 complete rounds after REBUILT_OLD)/L5/L8. A27's own a27_training_continued (tape rounds /
    publications) is kept as information; the continuation verdict is L3 (ledger triples), per T36-REVIEW §3.1."""
    res = _judge_d2_a27(journal, tape, known, dkill_log, probe_after, router_samples, launch_log)
    if ledger is None: ledger = _LEDGER_EXTRA["ledger"]
    if "a27_training_continued" in res.get("checks", {}) and ledger is not None:
        res.setdefault("a27_after_terminal", {})["a27_training_continued"] = res["checks"].pop("a27_training_continued")
    return _with_ledger(res, ledger, tape, journal, continues_after=_terminal_time(journal, "up1", "REBUILT_OLD"), min_rounds_after=A27_MIN_ROUNDS_AFTER)


def judge_e1b(journal, tape, known, router_samples=None, probes=None, ledger=None):  # noqa: F811
    res = _judge_e1b_base(journal, tape, known, router_samples, probes)
    if ledger is None: ledger = _LEDGER_EXTRA["ledger"]
    return _with_ledger(res, ledger, tape, journal, continues_after=_terminal_time(journal, "up1", "REBUILT_OLD"), min_rounds_after=2)


def judge_wd(journal, tape, known, gpu_samples=None, probe_after=None, gpu_release_s=60.0, unblock_s=60.0, ledger=None):  # noqa: F811
    res = _judge_wd_base(journal, tape, known, gpu_samples, probe_after, gpu_release_s, unblock_s)
    if ledger is None: ledger = _LEDGER_EXTRA["ledger"]
    return _with_ledger(res, ledger, tape, journal, continues_after=_terminal_time(journal, "up1", "REBUILT_OLD"), min_rounds_after=2)


def judge_recovery(case, journal, tape, known, ledger=None, launch_log=None, probe_after=None, min_rounds_after=3, up="up1", down="dn1"):  # noqa: F811
    """r5/r6/r7/r5c + L1/L2/L3(>= min_rounds_after after the recovery verified / terminal)/L5/L6/L7 (r5, r5c, r7)/L8."""
    res = _judge_recovery_base(case, journal, tape, known, ledger, launch_log, probe_after, min_rounds_after, up, down)
    ver = [r.get("wall_time") for r in journal if r.get("kind") == "recovery" and r.get("status") == "verified"]
    t = max(ver) if ver else max([r.get("wall_time") or 0 for r in phases(journal) if r.get("phase") in TERMINAL] or [None])
    return _with_ledger(res, ledger, tape, journal, continues_after=t, min_rounds_after=min_rounds_after, restart=True, verified=case in ("r5", "r5c", "r7"))


_main_a27 = main


def main(argv):  # noqa: F811
    """Stash --ledger for the ledger-aware e1b / wd / d2 wrappers (the base dispatch passes it only to d4 and r*)."""
    import argparse
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--ledger")
    a, _ = ap.parse_known_args(argv)
    _LEDGER_EXTRA["ledger"] = load(a.ledger) if a.ledger else None
    return _main_a27(argv)



# ----------------------------------------------------------------------------------------------- s0 (strict-avg single island + head syncer smoke)
# Appended only (2026-10-01). Criteria fixed in E1D-RECOVERY-PROGRESS.md "s0 结果" before the rerun: rc=0; syncer "learner connected" and
# "training complete"; publication versions contiguous 0..N; journal up1/dn1 exactly one SUCCEEDED each, epochs 0->1->2; inbox without
# a rejected status file; epochs.json config_epoch=2; tape two rl_reconfiguration SUCCEEDED; >= S0_MIN_ROUNDS_AFTER rounds trained after the
# last reconfiguration; one outer_recorded per rollout; no lora_unverifiable; host syncer cleanup rc=0; ledger L1-L3.  Missing evidence ->
# INVALID_TEST(evidence_missing).  The first s0 (a4s7-20261001-1) FAILs here: both requests were refused at the inbox (pause budget 450 s).
S0_MIN_ROUNDS = 6
S0_MIN_ROUNDS_AFTER = 2


def judge_s0(journal, tape, ledger, epochs, inbox_statuses, syncer_log, launch_log=None, rc=None, syncer_clean_rc=None,
             up="up1", down="dn1"):
    res = {"case": "s0", "checks": {}, "invalid_reasons": []}
    for name, val in (("tape", tape), ("journal", journal), ("ledger", ledger), ("epochs.json", epochs),
                      ("inbox listing", inbox_statuses), ("syncer log", syncer_log)):
        if val is None or (name == "tape" and not val):
            return _invalid(res, f"{name} missing: s0 evidence missing", "evidence_missing")
    if not any(e.get("event") == "rl_driver_start" for e in tape):
        return _invalid(res, "tape has no rl_driver_start: the island never started", "evidence_missing")
    c = res["checks"]
    c["rc_zero"] = rc == 0; res["rc"] = rc
    c["syncer_learner_connected"] = any("learner connected" in l for l in syncer_log)
    c["syncer_training_complete"] = any("training complete" in l for l in syncer_log)
    pubs = [e.get("policy_version") for e in tape if e.get("event") == "rl_publication"]
    pubs = [0 if v is None and i == 0 else v for i, v in enumerate(pubs)]   # the initial publication carries no version
    rounds = [e.get("rollout_id") for e in tape if e.get("event") == "rl_round_trained"]
    res["publications"] = pubs; res["rounds_trained"] = rounds
    c["publication_versions_contiguous"] = len(pubs) >= 2 and pubs == list(range(len(pubs)))
    c["rounds_trained"] = len(rounds) >= S0_MIN_ROUNDS
    for rid, epoch_to in ((up, 1), (down, 2)):
        term = _tx_terminal(journal, rid)
        rec = [r for r in phases(journal) if r.get("request_id") == rid and r.get("phase") == "SUCCEEDED"]
        c[f"{rid}_SUCCEEDED_once"] = term == ["SUCCEEDED"]
        c[f"{rid}_epoch_to_{epoch_to}"] = len(rec) == 1 and rec[0].get("config_epoch_to", rec[0].get("config_epoch")) == epoch_to
    res["terminals"] = {rid: _tx_terminal(journal, rid) for rid in (up, down)}
    rejected = {k: v.get("rejected") for k, v in inbox_statuses.items() if isinstance(v, dict) and v.get("rejected")}
    res["inbox_rejected"] = rejected; c["inbox_no_rejected"] = not rejected
    c["epochs_config_epoch_2"] = epochs.get("config_epoch") == 2; res["epochs"] = {k: epochs.get(k) for k in ("config_epoch", "config_id", "members")}
    rcf = [e for e in tape if e.get("event") == "rl_reconfiguration"]
    ok_rcf = [(e.get("config_epoch_from"), e.get("config_epoch")) for e in rcf if e.get("result") == "SUCCEEDED"]
    res["reconfigurations"] = [(e.get("result"), e.get("config_epoch_from"), e.get("config_epoch")) for e in rcf]
    c["tape_two_reconfig_succeeded"] = ok_rcf == [(0, 1), (1, 2)]
    t_last = max([e.get("time_unix") or 0 for e in rcf if e.get("result") == "SUCCEEDED"] or [None])
    after = [e for e in tape if t_last is not None and e.get("event") == "rl_round_trained" and (e.get("time_unix") or 0) > t_last]
    c["rounds_after_reconfig"] = len(after) >= S0_MIN_ROUNDS_AFTER; res["rounds_after_reconfig"] = [e.get("rollout_id") for e in after]
    orec = {}
    for r in ledger:
        if r.get("kind") == "outer_recorded": orec[r.get("rollout_id")] = orec.get(r.get("rollout_id"), 0) + 1
    c["outer_recorded_once_per_rollout"] = bool(rounds) and sorted(orec) == sorted(set(rounds)) and all(n == 1 for n in orec.values())
    res["outer_recorded"] = orec
    texts = [json.dumps(tape, default=str), json.dumps(journal, default=str)] + ([ "\n".join(launch_log)] if launch_log else [])
    c["no_lora_unverifiable"] = not any("lora_unverifiable" in t for t in texts)
    c["syncer_clean_rc_zero"] = syncer_clean_rc == 0; res["syncer_clean_rc"] = syncer_clean_rc
    lc = ledger_checks(ledger, tape, journal, continues_after=t_last, min_rounds_after=S0_MIN_ROUNDS_AFTER, tx_windows=False)
    for k in ("L1_round_triples", "L2_no_group_reuse", "L3_continues_after_terminal"):
        if k in lc["checks"]: c[k] = lc["checks"][k]
    res["ledger"] = {k: lc["info"].get(k) for k in ("L1_round_triples", "L2_no_group_reuse", "L3_continues_after_terminal")}
    res["verdict"] = "PASS" if all(v is True for v in c.values()) else "FAIL"
    if res["verdict"] == "FAIL": res["marker"] = "s0_failed"
    return res


def load_inbox_statuses(d):
    """{request_id: parsed <id>.status.json}; None when the inbox dir is not there (evidence missing); {} when empty."""
    if not d or not os.path.isdir(d):
        return None
    out = {}
    for n in sorted(os.listdir(d)):
        if n.endswith(".status.json"):
            try: out[n[:-len(".status.json")]] = json.load(open(os.path.join(d, n)))
            except Exception as e: out[n[:-len(".status.json")]] = {"unparsable": str(e)}
    return out


def _rc_from_file(p, prefix="rc="):
    """'rc=0' (rc.txt) or 'syncer_host_clean rc=0' (syncer_clean.txt) -> int, else None."""
    if not p or not os.path.exists(p): return None
    vals = re.findall(prefix + r"(\d+)", open(p).read())
    return int(vals[-1]) if vals else None


_main_ledger = main


def main(argv):  # noqa: F811
    """Case s0 with its own evidence flags; every other case is passed on unchanged (the s0-only flags are stripped)."""
    import argparse
    ap = argparse.ArgumentParser(add_help=False)
    for f in ("--epochs", "--inbox-dir", "--syncer-log", "--rc-file", "--syncer-clean"): ap.add_argument(f)
    a, rest = ap.parse_known_args(argv)
    if not rest or rest[0] != "s0":
        return _main_ledger(rest)
    bp = argparse.ArgumentParser(add_help=False)
    bp.add_argument("case"); bp.add_argument("journal"); bp.add_argument("tape")
    for f in ("--ledger", "--launch-log", "--out", "--marker-dir"): bp.add_argument(f)
    b, _ = bp.parse_known_args(rest)
    def jl(p): return [json.loads(l) for l in open(p) if l.strip()] if p and os.path.exists(p) else None
    def lines(p): return open(p).read().splitlines() if p and os.path.exists(p) else None
    epochs = json.load(open(a.epochs)) if a.epochs and os.path.exists(a.epochs) else None
    res = judge_s0(jl(b.journal), jl(b.tape), jl(b.ledger), epochs, load_inbox_statuses(a.inbox_dir), lines(a.syncer_log),
                   lines(b.launch_log), _rc_from_file(a.rc_file), _rc_from_file(a.syncer_clean, prefix=r"syncer_host_clean rc="))
    if b.out: Path(b.out).write_text(json.dumps(res, indent=1, default=str))
    if b.marker_dir and res.get("marker"):
        Path(b.marker_dir, res["marker"]).write_text(json.dumps({"case": "s0", "verdict": res["verdict"], "why": res.get("invalid_reasons") or [k for k, v in res["checks"].items() if v is not True]}))
    print(json.dumps(res, default=str)); return EXIT[res["verdict"]]

# entry point: must stay last (later appended main() definitions shadow earlier ones)
if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
