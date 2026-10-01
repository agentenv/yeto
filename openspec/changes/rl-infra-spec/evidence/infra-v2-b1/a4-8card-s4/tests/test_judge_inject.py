"""CPU tests of judge_inject (v2): synthetic journals/tapes/samplers/probes plus replays of real GPU runs.
run: python3 tests/test_judge_inject.py"""
import json, os, sys, tempfile, unittest
sys.path[:0] = [os.path.join(os.path.dirname(__file__), ".."), os.path.join(os.path.dirname(__file__), "..", "scripts")]
import judge_inject as J

FX = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures")
T = ["engine:c2", "engine:c3"]
URLS = ["http://10.0.0.1:20002", "http://10.0.0.1:20003"]
OLD = ["http://10.0.0.1:20000", "http://10.0.0.1:20001"]


def jl(name):
    return [json.loads(l) for l in open(os.path.join(FX, name)) if l.strip()]


def ph(p, **k): return {"kind": "phase", "phase": p, "tx_id": "tx", "request_id": "up1", **k}
def inj(**k): return {"event": "test_injection", **{"kind": "lora_perturb", "target_members": T, "phase": "publish", "ts": 10.0, "applied": True, **k}}
ADD = {"kind": "add_intent", "members": T, "tx_id": "tx"}
ME = {"kind": "member_engines", "target_members": T, "engine_urls": URLS}
RB = {"kind": "lora_readback", "target_members": T, "reference_lora_keys": 224, "blind": [],
      "engines": [{"engine": "engine0", "keys": 450, "lora_keys": 224, "lora_keys_differ": 0}, {"engine": "engine1", "keys": 450, "lora_keys": 224, "lora_keys_differ": 0},
                  {"engine": "engine2", "keys": 450, "lora_keys": 224, "lora_keys_differ": 112}, {"engine": "engine3", "keys": 450, "lora_keys": 224, "lora_keys_differ": 112}]}
HOLD = [{"event": "test_hold", "stage": "start", "ts": 100.0}, {"event": "test_hold", "stage": "end", "ts": 110.0, "start_ts": 100.0, "end_ts": 110.0}]


def samples(new_busy=0, list_new=True, n=6):
    pre = [{"t": 90.0, "data": {"inflight": {u: 1 for u in OLD}}}]
    win = [{"t": 101 + i, "data": {"inflight": {**{u: 1 for u in OLD}, **({u: new_busy for u in URLS} if list_new else {})}}} for i in range(n)]
    return pre + win


def probes(stale_equal=True, refused=True, c2="Stopped", c3="Stopped", versions=None):
    v = versions or {"c0": "v2", "c1": "v2"}
    return {"after": {"probe_attested": True, "membership": {"epoch": 2, "members": ["c0", "c1"]}, "versions": v, "cell_statuses": {"c0": "Serving", "c1": "Serving", "c2": c2, "c3": c3}},
            "stale": {"probe_attested": True, "versions_equal": stale_equal, "stale_url": URLS[0], "stale_ack": {"status": 200}},
            "oldepoch": {"probe_attested": True, "refused": refused}}


def e1b_journal(cause="payload_mismatch", **k):
    return [ADD, ME, RB, ph("REBUILD_OLD", cause=cause, inconsistent_engines=["engine2", "engine3"]), ph("REBUILT_OLD", cause=cause, inconsistent_engines=["engine2", "engine3"])]


class E1B(unittest.TestCase):
    def e1b(self, journal=None, tape=None, s=None, p=None):
        return J.judge_e1b(journal if journal is not None else e1b_journal(), tape if tape is not None else [inj()] + HOLD,
                           {"engine:c0", "engine:c1", *T}, (samples() if s is None else (None if s == "MISSING" else s)), probes() if p is None else p)

    def test_pass(self):
        r = self.e1b(); self.assertEqual(r["verdict"], "PASS", r); self.assertEqual(r["hold_window"]["new_urls"], URLS)

    def test_injection_not_applied_is_invalid(self):
        r = self.e1b(tape=[inj(applied=False)] + HOLD); self.assertEqual((r["verdict"], r["marker"]), ("INVALID_TEST", "injection_not_reached"))
        self.assertEqual(self.e1b(tape=HOLD)["verdict"], "INVALID_TEST")

    def test_readback_blind_or_unchanged_is_invalid_not_pass(self):
        blind = dict(RB, blind=["engine2", "engine3"], engines=[dict(e, lora_keys=0, lora_keys_differ=0) for e in RB["engines"]])
        r = self.e1b(journal=[ADD, ME, blind, ph("REBUILD_OLD", cause="lora_unverifiable"), ph("REBUILT_OLD", cause="lora_unverifiable")])
        self.assertEqual((r["verdict"], r["marker"]), ("INVALID_TEST", "evidence_missing"))
        same = dict(RB, engines=[dict(e, lora_keys_differ=0) for e in RB["engines"]])
        r = self.e1b(journal=[ADD, ME, same, ph("SUCCEEDED")]); self.assertEqual(r["verdict"], "INVALID_TEST"); self.assertIn("did not change", r["invalid_reasons"][-1])
        r = self.e1b(journal=[ADD, ME, ph("SUCCEEDED")]); self.assertEqual(r["verdict"], "INVALID_TEST")  # no read-back record at all

    def test_wrong_cause_or_success_fails(self):
        r = self.e1b(journal=e1b_journal(cause="lora_unverifiable")); self.assertEqual(r["verdict"], "FAIL"); self.assertFalse(r["checks"]["cause_payload_mismatch"])
        r = self.e1b(journal=[ADD, ME, RB, ph("SUCCEEDED")]); self.assertEqual(r["verdict"], "FAIL"); self.assertFalse(r["checks"]["terminal_REBUILT_OLD"])

    def test_new_engines_must_be_identified_independently_and_listed(self):
        r = self.e1b(journal=[ADD, RB] + e1b_journal()[3:])   # no member_engines
        self.assertEqual((r["verdict"], r["marker"]), ("INVALID_TEST", "evidence_missing"))
        r = self.e1b(s=samples(list_new=False))               # router never listed them: the old "new_urls=[] -> pass" bug
        self.assertEqual(r["verdict"], "INVALID_TEST"); self.assertEqual(r["hold_window"]["new_urls"], [])
        r = self.e1b(s=samples(n=2)); self.assertEqual(r["verdict"], "INVALID_TEST")   # window too short
        r = self.e1b(s="MISSING"); self.assertEqual(r["verdict"], "INVALID_TEST")
        r = self.e1b(s=samples(new_busy=2)); self.assertEqual(r["verdict"], "FAIL"); self.assertFalse(r["checks"]["a_cordoned_in_hold_window"])
        r = self.e1b(tape=[inj()]); self.assertEqual(r["verdict"], "INVALID_TEST")      # no hold window

    def test_probes_b_c_d(self):
        for key in ("after", "stale", "oldepoch"):
            p = probes(); p[key] = None
            self.assertEqual(self.e1b(p=p)["verdict"], "INVALID_TEST", key)
        r = self.e1b(p=probes(c2="Serving")); self.assertEqual(r["verdict"], "FAIL"); self.assertFalse(r["checks"]["b_injected_cells_not_serving"])
        absent = probes(); absent["after"]["cell_statuses"] = {"c0": "Serving", "c1": "Serving"}   # stopped cells deregistered = not serving
        self.assertEqual(self.e1b(p=absent)["verdict"], "PASS")
        r = self.e1b(p=probes(versions={"c0": "v2", "c1": "v1"})); self.assertFalse(r["checks"]["b_old_members_same_version"])
        r = self.e1b(p=probes(stale_equal=False)); self.assertEqual(r["verdict"], "FAIL"); self.assertFalse(r["checks"]["c_stale_ack_changes_nothing"])
        r = self.e1b(p=probes(refused=False)); self.assertEqual(r["verdict"], "FAIL"); self.assertFalse(r["checks"]["d_old_epoch_refused"])

    def test_recovery_required_is_recovery_failed(self):
        r = self.e1b(journal=[ADD, ME, RB, ph("RECOVERY_REQUIRED")]); self.assertEqual((r["verdict"], r["marker"]), ("FAIL", "recovery_failed"))

    def test_replay_a8e1b3_old_run_is_invalid_not_pass(self):
        """real run infra-v2-b1-a8e1b-20261001-3 (b19b781): the v1 judge scored (a) true with new_urls=[]; no read-back
        evidence existed, so v2 refuses to judge it (evidence_missing) rather than pass or fail the product."""
        journal, samples_ = jl("a8e1b3_journal.jsonl"), jl("a8e1b3_router_samples.jsonl")
        r = J.judge_e1b(journal, [], set(), samples_, probes())
        self.assertEqual((r["verdict"], r["marker"]), ("INVALID_TEST", "evidence_missing"))
        self.assertNotIn("a_cordoned_in_hold_window", r["checks"])


# ----------------------------------------------------------------------------------------------- watchdog
U = {i: "GPU-%d" % i for i in range(8)}
def gs(t, apps): return {"t": t, "apps": [[U[g], p] for g, p in apps]}
OLDAPPS = [(0, 100), (1, 101), (2, 102), (3, 103), (4, 200), (5, 201)]


def wd_journal(killed=True, errors=(), release="target dead: cell c2 has no workers (stopped)", t_fire=1000.0, terminal="REBUILT_OLD"):
    k = [{"cell": c, "fork_cell": c.split(":")[1], "worker": c.split(":")[1] + "/w0", "generation": 1} for c in T] if killed else []
    return [ph("INITIALIZING", wall_time=800.0), ADD, ph("VERIFYING", wall_time=950.0),
            {"kind": "test_injection", "injection_kind": "block_update", "applied": True, "reached_ts": 951.0, "target_members": T, "tx_id": "tx"},
            {"kind": "watchdog", "tx_id": "tx", "phase": "VERIFYING", "classification": "FIRED_ON_BLOCKED_UPDATE", "injection_reached": True, "target_cells": T, "wall_time": t_fire},
            {"kind": "watchdog_action", "tx_id": "tx", "killed": k, "errors": list(errors)},
            {"kind": "test_injection", "injection_kind": "block_update", "applied": True, "reached_ts": 951.0, "released_ts": t_fire + 3, "outcome": release, "target_members": T, "tx_id": "tx"},
            ph(terminal, wall_time=t_fire + 20)]


def wd_gpu(release_after=True, horizon=1100.0):
    out = [gs(t, OLDAPPS) for t in (700.0, 790.0)]
    out += [gs(t, OLDAPPS + [(6, 300), (7, 301)]) for t in (900.0, 960.0, 1000.0, 1010.0)]
    out += [gs(t, OLDAPPS + ([] if release_after else [(6, 300), (7, 301)])) for t in (1030.0, 1060.0, 1080.0, horizon)]
    return out


PA = {"probe_attested": True, "cell_statuses": {"c0": "Serving", "c1": "Serving", "c2": "Stopped", "c3": "Stopped"}}


class WD(unittest.TestCase):
    def wd(self, journal=None, gpu=None, probe=PA):
        return J.judge_wd(journal if journal is not None else wd_journal(), [], set(), wd_gpu() if gpu is None else gpu, probe)

    def test_pass(self):
        r = self.wd(); self.assertEqual(r["verdict"], "PASS", r); self.assertEqual(r["gpus"]["target"], ["GPU-6", "GPU-7"])

    def test_killed_empty_or_errors_fail(self):
        r = self.wd(wd_journal(killed=False, errors=[{"cell": T[0], "kind": "unknown_target", "error": "matches=[]"}]))
        self.assertEqual(r["verdict"], "FAIL"); self.assertFalse(r["checks"]["killed_covers_all_targets"]); self.assertFalse(r["checks"]["no_kill_errors"])
        r = self.wd(wd_journal(errors=[{"cell": T[1], "kind": "no_workers", "error": "no live worker actors"}])); self.assertEqual(r["verdict"], "FAIL")

    def test_block_elapsed_on_its_own_timer_fails(self):
        r = self.wd(wd_journal(release="elapsed")); self.assertEqual(r["verdict"], "FAIL"); self.assertFalse(r["checks"]["unblocked_by_the_kill"])

    def test_not_fired_on_block_is_invalid(self):
        j = wd_journal(); j[4] = dict(j[4], classification="FIRED_AFTER_BLOCK_RELEASED")
        self.assertEqual(self.wd(j)["verdict"], "INVALID_TEST")
        j = wd_journal(); j[4] = dict(j[4], classification="INJECTION_NOT_REACHED")
        self.assertEqual((self.wd(j)["verdict"], self.wd(j)["marker"]), ("INVALID_TEST", "injection_not_reached"))

    def test_gpu_evidence(self):
        r = self.wd(gpu=wd_gpu(release_after=False)); self.assertEqual(r["verdict"], "FAIL"); self.assertFalse(r["checks"]["target_gpus_released"])
        r = self.wd(gpu=[g for g in wd_gpu() if g["t"] <= 1025.0]); self.assertEqual((r["verdict"], r["marker"]), ("INVALID_TEST", "evidence_missing"))   # <10 s after REBUILT_OLD
        # orderly run end 30 s after REBUILT_OLD (all old pids vanish together): evaluated just before the end, target GPUs empty -> PASS
        g = [x for x in wd_gpu() if x["t"] <= 1030.0] + [gs(t, []) for t in (1050.0, 1080.0, 1100.0)]
        r = self.wd(gpu=g); self.assertEqual(r["verdict"], "PASS", r); self.assertEqual(r["gpus"]["run_ended_at"], 30.0)
        g = [x for x in wd_gpu() if x["t"] <= 1030.0] + [gs(t, [(6, 300)]) for t in (1050.0, 1080.0, 1100.0)]   # ended but a target process survived
        r = self.wd(gpu=g); self.assertEqual(r["verdict"], "FAIL"); self.assertFalse(r["checks"]["target_gpus_released"])
        g = [x for x in wd_gpu() if x["t"] <= 1030.0] + [gs(t, [(0, 100), (1, 101), (2, 102), (3, 103), (4, 200)]) for t in (1050.0, 1080.0, 1100.0)]   # one old member died, others stayed
        r = self.wd(gpu=g); self.assertEqual(r["verdict"], "FAIL"); self.assertFalse(r["checks"]["old_member_pids_unchanged"])
        r = self.wd(gpu=None if False else []); self.assertEqual(r["verdict"], "INVALID_TEST")
        g = wd_gpu(); g[-2] = gs(1080.0, [(0, 100), (1, 999), (2, 102), (3, 103), (4, 200), (5, 201)])   # an old member restarted (new pid) by +60 s
        r = self.wd(gpu=g); self.assertEqual(r["verdict"], "FAIL"); self.assertFalse(r["checks"]["old_member_pids_unchanged"])
        g = [gs(t, OLDAPPS) for t in (700.0, 790.0, 900.0, 1000.0, 1100.0)]   # the targets never started on any GPU
        self.assertEqual(self.wd(gpu=g)["verdict"], "INVALID_TEST")

    def test_probe_restart(self):
        self.assertEqual(self.wd(probe=None)["verdict"], "INVALID_TEST")
        r = self.wd(probe={"probe_attested": True, "cell_statuses": {"c2": "Serving", "c3": "Stopped"}}); self.assertEqual(r["verdict"], "FAIL")

    def test_replay_a8ch2_wd_is_fail_not_pass(self):
        """real run infra-v2-b1-a8ch-20261001-2-wd (6f6dcb9): watchdog fired on the block, nothing killed (matches=[]),
        block elapsed on its own 600 s timer, REBUILT_OLD only via lora_unverifiable. Must be FAIL on (1)(2)."""
        r = J.judge_wd(jl("a8ch2_wd_journal.jsonl"), [], set(), jl("a8ch2_wd_gpu_samples.jsonl"), PA)
        self.assertEqual(r["verdict"], "FAIL", r)
        self.assertFalse(r["checks"]["killed_covers_all_targets"]); self.assertFalse(r["checks"]["unblocked_by_the_kill"]); self.assertFalse(r["checks"]["unblocked_within_60s"])
        self.assertIn("matches=[]", json.dumps(r["errors"]))
        self.assertEqual(len(r["gpus"]["target"]), 2)   # the target generation did start on two previously empty GPUs


# ----------------------------------------------------------------------------------------------- A4b
DT = {"kind": "drain_timeout", "tx_id": "dn", "active_requests": 0, "tool_wait": 1, "blockers": ["1 trajectories waiting on tools"], "target_members": ["engine:c3"]}
TW = {"event": "test_injection", "kind": "tool_wait", "applied": True, "ts": 1.0, "target_members": ["engine:c3"]}
PA4 = {"probe_attested": True, "membership": {"epoch": 1, "members": ["c0", "c1", "c2", "c3"]}, "cell_statuses": {c: "Serving" for c in ("c0", "c1", "c2", "c3")}}


def dn(p, **k): return {"kind": "phase", "phase": p, "tx_id": "dn", "request_id": "dn1", **k}


class A4B(unittest.TestCase):
    def test_cancel_variant(self):
        j = [DT, dn("CANCELLED")]
        self.assertEqual(J.judge_a4b(j, [TW], set(), "cancel", PA4)["verdict"], "PASS")
        self.assertEqual(J.judge_a4b(j, [dict(TW, applied=False)], set(), "cancel", PA4)["verdict"], "INVALID_TEST")
        self.assertEqual(J.judge_a4b([dn("CANCELLED")], [TW], set(), "cancel", PA4)["verdict"], "INVALID_TEST")   # drain never blocked on the tool wait
        self.assertEqual(J.judge_a4b([DT, dn("SUCCEEDED")], [TW], set(), "cancel", PA4)["verdict"], "FAIL")
        self.assertEqual(J.judge_a4b(j, [TW], set(), "cancel", None)["verdict"], "INVALID_TEST")
        r = J.judge_a4b([DT, {"kind": "fork_op", "op": "stop", "tx_id": "dn"}, dn("CANCELLED")], [TW], set(), "cancel", PA4)
        self.assertEqual(r["verdict"], "FAIL"); self.assertFalse(r["checks"]["no_stop_issued"])
        r = J.judge_a4b([DT, {"kind": "undrain_failed", "tx_id": "dn", "error": "x"}, dn("RECOVERY_REQUIRED")], [TW], set(), "cancel", PA4)
        self.assertEqual((r["verdict"], r["marker"]), ("FAIL", "recovery_failed"))
        bad = dict(PA4, cell_statuses=dict(PA4["cell_statuses"], c3="Cordoned")); self.assertEqual(J.judge_a4b(j, [TW], set(), "cancel", bad)["verdict"], "FAIL")
        # router sampler as the restoration evidence (no probe): cordoned during the drain, empty after CANCELLED, served again
        jc = [DT, dn("CANCELLED", wall_time=100.0)]
        def rs(t, cord, busy): return {"t": t, "data": {"inflight": {"http://a": 1, "http://b": busy}, "cordoned": cord}}
        good = [rs(96.0, ["http://b"], 0), rs(97.0, ["http://b"], 0), rs(101.0, [], 0), rs(102.0, [], 2), rs(103.0, [], 0)]
        r = J.judge_a4b(jc, [TW], set(), "cancel", None, good); self.assertEqual(r["verdict"], "PASS", r)
        still = good[:2] + [rs(t, ["http://b"], 0) for t in (101.0, 102.0, 103.0)]
        r = J.judge_a4b(jc, [TW], set(), "cancel", None, still); self.assertEqual(r["verdict"], "FAIL"); self.assertFalse(r["checks"]["routing_restored"])
        never_busy = good[:3] + [rs(102.0, [], 0), rs(103.0, [], 0)]
        self.assertEqual(J.judge_a4b(jc, [TW], set(), "cancel", None, never_busy)["verdict"], "FAIL")
        self.assertEqual(J.judge_a4b(jc, [TW], set(), "cancel", None, good[:3])["verdict"], "INVALID_TEST")   # <3 samples after
        self.assertEqual(J.judge_a4b(jc, [TW], set(), "cancel", None, [rs(t, [], 0) for t in (96.0, 101.0, 102.0, 103.0)])["verdict"], "INVALID_TEST")   # never saw the cordon
        self.assertEqual(J.judge_a4b(jc, [TW], set(), "cancel", bad, good)["verdict"], "FAIL")   # probe contradicts the router

    def test_recovery_variant(self):
        uf = {"event": "test_injection", "kind": "undrain_fail", "applied": True, "target_members": ["engine:c3"]}
        j = [DT, {"kind": "undrain_failed", "tx_id": "dn", "error": "injected"}, dn("RECOVERY_REQUIRED")]
        t = [TW, uf, {"event": "rl_reconfiguration", "result": "RECOVERY_REQUIRED"}]
        self.assertEqual(J.judge_a4b(j, t, set(), "recovery")["verdict"], "PASS")
        self.assertEqual(J.judge_a4b(j, [TW], set(), "recovery")["verdict"], "INVALID_TEST")
        r = J.judge_a4b([DT, dn("CANCELLED")], t, set(), "recovery"); self.assertEqual(r["verdict"], "FAIL")   # wrapped as a successful cancel


# ----------------------------------------------------------------------------------------------- E1-D
def tx(p, rid, **k): return {"kind": "phase", "phase": p, "request_id": rid, "tx_id": rid, **k}


def d4_journal():
    """2026-10-02 ruling shape: dn1 requested at t=100 with deadline 600, REBUILD_OLD at 700 (bounded to 820), island + request
    RECOVERY_REQUIRED at 790 (<= 600 + 120)."""
    return [tx("SUCCEEDED", "up1"),
            {"kind": "request", "request_id": "dn1", "tx_id": "dn1", "wall_time": 100.0, "deadline_wall": 700.0, "body": {"deadline_s": 600.0}},
            {"kind": "fork_op", "op": "stop", "status": "incomplete", "tx_id": "dn1", "wall_time": 101.0},
            tx("REBUILD_OLD", "dn1", wall_time=700.0, cause="stop_retry_deadline", deadline_wall=700.0, recovery_deadline_wall=820.0),
            {"kind": "phase", "phase": "RECOVERY_REQUIRED", "tx_id": "dn1", "request_id": None, "scope": "island", "seq": 9,
             "wall_time": 790.0, "error": "rebuild of the old engine set failed", "cause": "rebuild_old_failed", "config_epoch": 1, "fork_epoch": 1},
            tx("RECOVERY_REQUIRED", "dn1", wall_time=790.0, scope="request", seq=10, island_record_seq=9, error="rebuild of the old engine set failed",
               cause="rebuild_old_failed", config_epoch=1, fork_epoch=1, deadline_wall=700.0, recovery_deadline_wall=820.0)]


TID = "yeto-test-injected-tool-wait"
CID = TID + ":drain:engine:c3"
TWC = dict(TW, side_effect_log=True, tool_call_id=CID, attempt=1)


def se(kind, t, seq, tid=TID, cid=CID, **k): return {"kind": kind, "seq": seq, "trajectory_id": tid, "tool_call_id": cid, "wall_time": t, **k}
def a4bc_journal(t_req=90.0, t_cancel=100.0): return [{"kind": "request", "tx_id": "dn", "request_id": "dn1", "wall_time": t_req}, DT, dn("CANCELLED", wall_time=t_cancel)]
SE_OK = [se("tool_side_effect", 95.0, 1, attempt=1), se("tool_complete", 125.0, 2, attempt=1)]


class A4BC(unittest.TestCase):
    """a4bc = a4b + the tool side-effect journal: exactly one tool_side_effect per (trajectory, tool call) inside the cancelled
    transaction, none repeated after CANCELLED (3.3 X5 (b)); no_stop + CANCELLED alone is INVALID, not PASS."""
    def j(self, side_effects, journal=None, tape=None, probe=PA4):
        return J.judge_a4bc(journal if journal is not None else a4bc_journal(), [TWC] if tape is None else tape, set(), side_effects, probe)

    def test_pass_and_each_check(self):
        r = self.j(SE_OK); self.assertEqual((r["case"], r["verdict"]), ("a4bc", "PASS"), r)
        self.assertEqual(r["side_effects"]["in_tx_pairs"], {f"{TID}|{CID}": 1}); self.assertEqual(r["side_effects"]["tool_complete_pairs"], {f"{TID}|{CID}": 1})
        self.assertTrue(all(r["checks"][k] for k in ("one_side_effect_per_call_in_tx", "no_replay_after_cancel", "no_duplicate_pairs", "injection_matches_journal", "seq_monotonic", "no_stop_issued", "terminal_CANCELLED", "routing_restored")))
        # replay after the cancel: the same pair executed again -> FAIL
        r = self.j(SE_OK + [se("tool_side_effect", 101.0, 3, attempt=2)]); self.assertEqual(r["verdict"], "FAIL")
        self.assertFalse(r["checks"]["no_replay_after_cancel"]); self.assertFalse(r["checks"]["no_duplicate_pairs"]); self.assertEqual(r["side_effects"]["replayed_after_cancel"], [f"{TID}|{CID}"])
        # replay inside the transaction (a second execution before CANCELLED) -> FAIL
        r = self.j([se("tool_side_effect", 95.0, 1), se("tool_side_effect", 97.0, 2)]); self.assertEqual(r["verdict"], "FAIL"); self.assertFalse(r["checks"]["one_side_effect_per_call_in_tx"])
        # new ids after the cancel (new rollouts) are allowed and listed
        r = self.j(SE_OK + [se("tool_side_effect", 130.0, 3, tid="traj-9", cid="call-9")]); self.assertEqual(r["verdict"], "PASS", r)
        self.assertEqual(r["side_effects"]["new_pairs_after_cancel"], ["traj-9|call-9"])
        # missing/empty journal, injection without the switch, no record inside the tx -> INVALID (never PASS on empty evidence)
        r = self.j(None); self.assertEqual((r["verdict"], r["marker"]), ("INVALID_TEST", "evidence_missing"))
        self.assertEqual(self.j([])["verdict"], "INVALID_TEST")
        r = self.j(SE_OK, tape=[TW]); self.assertEqual((r["verdict"], r["marker"]), ("INVALID_TEST", "injection_not_reached"))
        r = self.j([se("tool_side_effect", 50.0, 1)]); self.assertEqual(r["verdict"], "INVALID_TEST")   # executed before the request: clocks disagree / not this drain
        r = self.j(SE_OK, journal=[DT, dn("CANCELLED", wall_time=100.0)]); self.assertEqual((r["verdict"], r["marker"]), ("INVALID_TEST", "evidence_missing"))   # no request record
        # the a4b part still decides: no probe/samples -> INVALID; a stop -> FAIL; RECOVERY_REQUIRED -> recovery_failed
        self.assertEqual(self.j(SE_OK, probe=None)["verdict"], "INVALID_TEST")
        r = self.j(SE_OK, journal=a4bc_journal() + [{"kind": "fork_op", "op": "stop", "tx_id": "dn"}]); self.assertEqual(r["verdict"], "FAIL"); self.assertFalse(r["checks"]["no_stop_issued"])
        r = self.j(SE_OK, journal=[a4bc_journal()[0], DT, {"kind": "undrain_failed", "tx_id": "dn", "error": "x"}, dn("RECOVERY_REQUIRED")]); self.assertEqual((r["verdict"], r["marker"]), ("FAIL", "recovery_failed"))
        # injection tool_call_id not in the journal, non-monotonic seq -> FAIL
        r = self.j(SE_OK, tape=[dict(TWC, tool_call_id="other")]); self.assertEqual(r["verdict"], "FAIL"); self.assertFalse(r["checks"]["injection_matches_journal"])
        r = self.j([se("tool_side_effect", 95.0, 2), se("tool_complete", 125.0, 1)]); self.assertEqual(r["verdict"], "FAIL"); self.assertFalse(r["checks"]["seq_monotonic"])

    def test_replay_chain1_a4b_real_journal_with_synthetic_side_effects(self):
        """Chain 1 a4b (infra-v2-b1-a4s5-20261001-1-a4b, 8xH100): real journal / tape fragment / router samples. On its own it is
        INVALID for a4bc (the run had no side-effect journal: no_stop + CANCELLED cannot prove no replay); with the journal
        switch in the injection record and a synthetic side_effects.jsonl aligned to the real wall clock it decides PASS/FAIL."""
        jr, tape, rs = jl("a4s5_1_a4b_journal.jsonl"), jl("a4s5_1_a4b_tape_fragment.jsonl"), jl("a4s5_1_a4b_router_samples.jsonl")
        r = J.judge_a4bc(jr, tape, set(), [], None, rs); self.assertEqual((r["verdict"], r["marker"]), ("INVALID_TEST", "injection_not_reached"), r["invalid_reasons"])
        inj = next(e for e in tape if e.get("event") == "test_injection"); members = inj["target_members"]
        cid = f"{TID}:drain:{','.join(sorted(members))}"
        tape2 = [dict(e, side_effect_log=True, tool_call_id=cid, attempt=1) if e is inj else e for e in tape]
        t_inj = inj["ts"]; t_cancel = next(x["wall_time"] for x in jr if x.get("phase") == "CANCELLED"); self.assertLess(t_inj, t_cancel)
        good = [se("tool_side_effect", t_inj - 0.002, 1, cid=cid, attempt=1, seconds=30.0, target_members=members), se("tool_complete", t_inj + 30.0, 2, cid=cid, attempt=1)]
        r = J.judge_a4bc(jr, tape2, set(), good, None, rs); self.assertEqual(r["verdict"], "PASS", r)
        self.assertEqual(r["side_effects"]["in_tx_pairs"], {f"{TID}|{cid}": 1}); self.assertEqual(r["side_effects"]["t_cancel"], t_cancel)
        r = J.judge_a4bc(jr, tape2, set(), [], None, rs); self.assertEqual((r["verdict"], r["marker"]), ("INVALID_TEST", "evidence_missing"))
        replay = good + [se("tool_side_effect", t_cancel + 0.5, 3, cid=cid, attempt=2, target_members=members)]
        r = J.judge_a4bc(jr, tape2, set(), replay, None, rs); self.assertEqual(r["verdict"], "FAIL"); self.assertEqual(r["side_effects"]["replayed_after_cancel"], [f"{TID}|{cid}"])
        later = good + [se("tool_side_effect", t_cancel + 20.0, 3, tid="traj-r3-0", cid="tool-7")]
        r = J.judge_a4bc(jr, tape2, set(), later, None, rs); self.assertEqual(r["verdict"], "PASS"); self.assertEqual(r["side_effects"]["new_pairs_after_cancel"], ["traj-r3-0|tool-7"])


class D(unittest.TestCase):
    def test_d123(self):
        j = [tx("SUCCEEDED", "up1"), {"kind": "fork_op", "op": "stop", "status": "incomplete", "tx_id": "dn1"}, tx("SUCCEEDED", "dn1"), tx("REBUILT_OLD", "up2"), tx("REBUILT_OLD", "up3")]
        k = [{"event": "kill", "rule": "d1", "gpu": 6, "res": {"300": "killed"}}, {"event": "kill", "rule": "d2", "gpu": 6, "res": {"301": "killed"}}]
        self.assertEqual(J.judge_d123(j, [], set(), k)["verdict"], "PASS")
        self.assertEqual(J.judge_d123(j, [], set(), [])["verdict"], "INVALID_TEST")
        self.assertEqual(J.judge_d123(j, [], set(), k[:1])["verdict"], "FAIL")
        self.assertEqual(J.judge_d123([j[0], j[2], j[3], j[4]], [], set(), k)["verdict"], "FAIL")   # no incomplete stop
        self.assertEqual(J.judge_d123(j[:3] + [tx("SUCCEEDED", "up2"), j[4]], [], set(), k)["verdict"], "FAIL")
        r = J.judge_d123(j[:4] + [tx("INITIALIZING", "up3")], [], set(), k)   # up3 submitted, run ended before its terminal
        self.assertEqual((r["verdict"], r["marker"]), ("INVALID_TEST", "evidence_missing")); self.assertTrue(r["checks"]["up2_REBUILT_OLD"])

    def test_d2(self):
        # 2026-10-02: judge_d2 = base + A27 (discovery, REBUILD_OLD->REBUILT_OLD, probe, router) + T36 ledger L1/L2/L3/L5/L8
        j = [{"kind": "add_intent", "members": T, "tx_id": "up1"}, tx("INITIALIZING", "up1", wall_time=95.0), tx("VERIFYING", "up1", wall_time=100.0),
             {"kind": "target_workers_lost", "tx_id": "up1", "lost_members": T}, tx("REBUILD_OLD", "up1", wall_time=101.0), tx("REBUILT_OLD", "up1", wall_time=130.0)]
        k = [{"event": "kill", "rule": "d2", "gpu": 6, "res": {"300": "killed"}, "wall": 100.5}]
        tape = [{"event": "rl_round_trained", "rollout_id": i, "time_unix": t} for i, t in ((0, 41.0), (1, 81.0), (2, 146.0), (3, 166.0))]
        tape += [pub(v, "yeto:%d:h" % v, OLD, t) for v, t in ((1, 45.0), (2, 85.0), (3, 150.0), (4, 170.0))]
        tape += [{"event": "rl_driver_phase", "phase": "sync", "time_unix": t} for t in (44.0, 84.0, 149.0, 169.0)]
        probe = {"probe_attested": True, "cell_statuses": {"c0": "Serving", "c1": "Serving", "c2": "Stopped", "c3": "Stopped"}, "versions": {"c0": "v2", "c1": "v2"}}
        samples = [{"t": 131.0 + i, "data": {"cordoned": []}} for i in range(3)]
        led = ledger_rounds([], 0, 40.0, ["a", "b"]); ledger_rounds(led, 1, 80.0, ["c", "d"]); ledger_rounds(led, 2, 145.0, ["e", "f"]); ledger_rounds(led, 3, 165.0, ["g", "h"]); seqd(led)
        r = J.judge_d2(j, tape, set(), k, probe, samples, led); self.assertEqual(r["verdict"], "PASS", r)
        self.assertNotIn("a27_training_continued", r["checks"]); self.assertTrue(r["a27_after_terminal"]["a27_training_continued"])   # moved to info: L3 judges continuation
        self.assertTrue(r["checks"]["L3_continues_after_terminal"] and r["checks"]["L5_silent_during_transactions"])
        self.assertEqual(J.judge_d2(j, tape, set(), [], probe, samples, led)["verdict"], "INVALID_TEST")   # kill not applied
        J._A27_EXTRA["probe_after"] = None   # (main() of an earlier test may have stashed one)
        self.assertEqual(J.judge_d2(j, tape, set(), k, None, samples, led)["verdict"], "INVALID_TEST")   # no probe after the terminal state
        self.assertEqual(J.judge_d2([j[0], tx("VERIFYING", "up1", wall_time=100.0), tx("SUCCEEDED", "up1")], tape, set(), k, probe, samples, led)["verdict"], "FAIL")
        self.assertEqual(J.judge_d2([j[0], tx("VERIFYING", "up1", wall_time=100.0)], tape, set(), k, probe, samples, led)["verdict"], "INVALID_TEST")
        r = J.judge_d2(j, tape, set(), k, probe, samples, led[:-4]); self.assertEqual(r["verdict"], "FAIL"); self.assertFalse(r["checks"]["L3_continues_after_terminal"])   # only 1 round after
        noisy = seqd(led + [L("prepared", 9, 120.0, group_ids=["z"])]); self.assertFalse(J.judge_d2(j, tape, set(), k, probe, samples, noisy)["checks"]["L5_silent_during_transactions"])
        self.assertEqual(J.judge_d2(j, tape, set(), k, probe, samples, [])["verdict"], "INVALID_TEST")   # ledger file missing/empty
        self.assertEqual(J.judge_d2(j, tape, set(), k, probe, samples)["ledger"], "not provided")

    def test_d4(self):
        # 2026-10-02 ruling: request-level + island-level RECOVERY_REQUIRED, <= 1 REBUILD_OLD, deadline + recovery_timeout bound
        j = d4_journal()
        t = [{"event": "rl_reconfiguration", "result": "RECOVERY_REQUIRED"}]
        led = seqd(ledger_rounds(ledger_rounds([], 0, 10.0, ["a", "b"]), 1, 60.0, ["c", "d"]))
        r = J.judge_d4(j, t, set(), led, recovery_timeout_s=120.0)
        self.assertEqual(r["verdict"], "PASS", r); self.assertTrue(r["checks"]["L4_silent_after_recovery"])
        self.assertEqual(J.judge_d4(j, t, set(), recovery_timeout_s=120.0)["verdict"], "INVALID_TEST")   # ledger required (T36 §5.1)
        self.assertEqual(J.judge_d4(j, t + [{"event": "rl_batch_prepared", "ts": 800.0}], set(), led, recovery_timeout_s=120.0)["verdict"], "FAIL")
        r = J.judge_d4(j, t, set(), ledger=seqd(led + [L("prepared", 2, 800.0, group_ids=["e"])]), recovery_timeout_s=120.0)
        self.assertEqual(r["verdict"], "FAIL"); self.assertFalse(r["checks"]["L4_silent_after_recovery"]); self.assertFalse(r["checks"]["no_prepared_after_recovery"])
        r = J.judge_d4(j, t + [{"event": "rl_driver_phase", "phase": "train", "time_unix": 800.0}], set(), led, recovery_timeout_s=120.0)
        self.assertFalse(r["checks"]["L4_silent_after_recovery"])   # trained after RECOVERY_REQUIRED
        self.assertEqual(J.judge_d4([j[0], tx("SUCCEEDED", "dn1")], t, set(), led)["verdict"], "INVALID_TEST")
        self.assertEqual(J.judge_d4(j, [], set(), led, recovery_timeout_s=120.0)["verdict"], "FAIL")
        # no request-level terminal (pre-fix controller) -> FAIL on dn1_RECOVERY_REQUIRED only
        r = J.judge_d4([x for x in j if not (x.get("phase") == "RECOVERY_REQUIRED" and x.get("request_id"))], t, set(), led, recovery_timeout_s=120.0)
        self.assertEqual(r["verdict"], "FAIL"); self.assertFalse(r["checks"]["dn1_RECOVERY_REQUIRED"]); self.assertFalse(r["checks"]["island_record_consistent"])
        # a second REBUILD_OLD, a reset deadline, or running past deadline + recovery_timeout + margin all fail
        extra = dict(j[3]); self.assertEqual(J.judge_d4(j[:4] + [extra] + j[4:], t, set(), led, recovery_timeout_s=120.0)["checks"]["at_most_one_REBUILD_OLD"], False)
        reset = [dict(x, deadline_wall=5000.0) if x.get("phase") == "REBUILD_OLD" else x for x in j]
        self.assertFalse(J.judge_d4(reset, t, set(), led, recovery_timeout_s=120.0)["checks"]["deadline_not_reset"])
        late = [dict(x, wall_time=900.0) if x.get("phase") == "RECOVERY_REQUIRED" else x for x in j]
        self.assertFalse(J.judge_d4(late, t, set(), led, recovery_timeout_s=120.0)["checks"]["within_time_limit"])
        # recovery_timeout from REBUILD_OLD.recovery_deadline_wall when not given
        self.assertEqual(J.judge_d4(j, t, set(), led)["time"]["recovery_timeout_s"], 120.0)

    def test_d4_replay_chain_6r2(self):
        """Real journal of chain 6r2 d4 (pre-fix controller): island-level RECOVERY_REQUIRED only -> FAIL on the request-level
        terminal; everything else (one REBUILD_OLD, 721 s <= 600 + 120 + 60, no prepared after) holds. With the request-level
        record the fixed controller writes appended (synthetic), the same run PASSes."""
        j, t, l = jl("a4s6_6r2_d4_journal.jsonl"), jl("a4s6_6r2_d4_tape.jsonl"), jl("a4s6_6r2_d4_ledger.jsonl")
        r = J.judge_d4(j, t, set(), l, recovery_timeout_s=120.0)
        self.assertEqual((r["verdict"], r["marker"]), ("FAIL", "recovery_failed"))
        self.assertEqual({k for k, v in r["checks"].items() if v is not True}, {"dn1_RECOVERY_REQUIRED", "island_record_consistent"})
        self.assertTrue(700 < r["time"]["elapsed_s"] < 780)
        isl = next(x for x in j if x.get("phase") == "RECOVERY_REQUIRED")
        fixed = j + [dict(isl, seq=isl["seq"] + 1, request_id="dn1", scope="request", cause="rebuild_old_failed", island_record_seq=isl["seq"])]
        r = J.judge_d4(fixed, t, set(), l, recovery_timeout_s=120.0)
        self.assertEqual(r["verdict"], "PASS", r)
        # the ledger of that run: anything prepared after the recovery time would fail it
        r = J.judge_d4(fixed, t, set(), l + [{"kind": "prepared", "wall_time": isl["wall_time"] + 5}], recovery_timeout_s=120.0)
        self.assertEqual(r["verdict"], "FAIL")


# ----------------------------------------------------------------------------------------------- E1-A (c)
class E1A(unittest.TestCase):
    def test_e1a_c(self):
        mem = lambda rid, n, k: {"event": "rl_membership", "round": rid, "members": ["m%d" % i for i in range(n)], "kind": k, "config_epoch": 0, "tx_id": "t"}
        tape = [{"event": "rl_publication", "sync/publication_members": ["m0"]}, mem(2, 3, "up"), {"event": "rl_publication", "sync/publication_members": ["m0", "m1", "m2"]},
                mem(4, 1, "down"), {"event": "rl_publication", "sync/publication_members": ["m0"]}]
        r = J.judge_e1a_c([], tape, [1, 1, 3, 3, 1, 1]); self.assertEqual(r["verdict"], "PASS", r)
        r = J.judge_e1a_c([], tape, [1, 1, 3, 3, 3, 1]); self.assertEqual(r["verdict"], "FAIL")
        tape[-1]["sync/publication_members"] = ["m0", "m1", "m2"]
        r = J.judge_e1a_c([], tape, [1, 1, 3, 3, 1, 1]); self.assertEqual(r["verdict"], "FAIL"); self.assertFalse(r["checks"]["publication_after_down_has_final_member_count"])
        self.assertEqual(J.judge_e1a_c([], [], [1])["verdict"], "INVALID_TEST")
        empty = [{"event": "rl_membership", "round": 2, "members": [], "kind": "up"}]
        self.assertEqual(J.judge_e1a_c([], empty, [1, 1])["verdict"], "INVALID_TEST")   # empty member list is no observation
        tape[-1]["sync/publication_members"] = ["m0"]
        blind = [dict(RB, blind=["engine2"], engines=[dict(e, lora_keys=0) for e in RB["engines"]])]
        r = J.judge_e1a_c(blind, tape, [1, 1, 3, 3, 1, 1]); self.assertEqual(r["verdict"], "FAIL"); self.assertFalse(r["checks"]["lora_readback_never_blind"])
        r = J.judge_e1a_c([RB], tape, [1, 1, 3, 3, 1, 1]); self.assertEqual(r["verdict"], "PASS")

    def test_e1a_c_real_8card_tape(self):
        """real tape fragment of infra-v2-b1-a8e1a-20261001-1 (8xH100, 12 rounds): up -> rl_membership.round=2, down -> round=7 (0-based rollout ids)."""
        tape = jl("a8e1a_tape_fragment.jsonl")
        exp = [2, 2, 4, 4, 4, 4, 4, 2, 2, 2, 2, 2]
        r = J.judge_e1a_c([], tape, exp); self.assertEqual(r["verdict"], "PASS", r); self.assertEqual(r["members_per_round"], exp)
        old = J.judge_e1a_c([], tape, exp, round_is_rollout_id=False)
        self.assertEqual(old["verdict"], "FAIL"); self.assertEqual(old["members_per_round"], [None, 4, 4, 4, 4, 4, 2, 2, 2, 2, 2, 2])
        self.assertEqual(J.judge_e1a_c([], tape, [2, 2, 2, 4, 4, 4, 4, 2, 2, 2, 2, 2])["verdict"], "FAIL")


class CLI(unittest.TestCase):
    def test_cli_writes_marker_and_reads_probes(self):
        d = tempfile.mkdtemp(); jp, tp = os.path.join(d, "j.jsonl"), os.path.join(d, "t.jsonl")
        open(jp, "w").write(json.dumps(ADD) + "\n" + json.dumps(ph("SUCCEEDED")) + "\n"); open(tp, "w").write(json.dumps(inj(applied=False)) + "\n")
        rc = J.main(["e1b", jp, tp, "--marker-dir", d, "--out", os.path.join(d, "o.json")])
        self.assertEqual(rc, 4); self.assertTrue(os.path.exists(os.path.join(d, "injection_not_reached")))
        pf = os.path.join(d, "p.txt"); open(pf, "w").write("noise\n" + json.dumps({"probe_attested": True, "x": 1}) + "\n")
        self.assertEqual(J.load_probe(pf)["x"], 1); self.assertIsNone(J.load_probe(os.path.join(d, "none")))
        open(pf, "w").write(json.dumps({"probe_attested": False}) + "\n"); self.assertIsNone(J.load_probe(pf))
        # a4bc through the CLI: --side-effects file (judge_after.sh passes elastic-state/side_effects.jsonl); missing file -> INVALID (evidence_missing)
        jp2, tp2, sp, pp = (os.path.join(d, n) for n in ("j2.jsonl", "t2.jsonl", "se.jsonl", "pa.txt"))
        open(jp2, "w").write("".join(json.dumps(x) + "\n" for x in a4bc_journal())); open(tp2, "w").write(json.dumps(TWC) + "\n")
        open(sp, "w").write("".join(json.dumps(x) + "\n" for x in SE_OK)); open(pp, "w").write(json.dumps(PA4) + "\n")
        self.assertEqual(J.main(["a4bc", jp2, tp2, "--side-effects", sp, "--probe-after", pp, "--out", os.path.join(d, "o2.json")]), 0)
        self.assertEqual(json.load(open(os.path.join(d, "o2.json")))["checks"]["no_replay_after_cancel"], True)
        rc = J.main(["a4bc", jp2, tp2, "--side-effects", os.path.join(d, "absent.jsonl"), "--probe-after", pp, "--marker-dir", d])
        self.assertEqual(rc, 4); self.assertTrue(os.path.exists(os.path.join(d, "evidence_missing")))





# ----------------------------------------------------------------------------------------------- E1-D ⑤⑥⑦ restart recovery (r5/r6/r7/r5c)
C4 = ["engine:c0", "engine:c1", "engine:c2", "engine:c3"]


def rec(status, **k): return {"kind": "recovery", "tx_id": "rec-1-1-abc", "status": status, **k}
def start(t): return {"event": "rl_driver_start", "time_unix": t}
def pub(v, tok, members, t): return {"event": "rl_publication", "policy_version": v, "rl/policy_token": tok, "sync/publication_members": members, "time_unix": t}
def train(t): return {"event": "rl_driver_phase", "phase": "train", "time_unix": t}
VER_OK = {"members": C4, "fork_epoch": 2, "policy_token": "verified", "router": {"not_admitted": []}, "trainer_layout": {"world": 4, "tp": 1, "pp": 1, "cp": 1, "ep": 1, "dp": 4}, "unconsumed_batches": [], "published_version": 1}
def L(kind, rid, t, attempt=0, **k): return {"kind": kind, "rollout_id": rid, "attempt": attempt, "wall_time": t, **k}


def ledger_rounds(recs, rid, t, groups, attempt=0, outer=True):
    recs += [L("prepared", rid, t, attempt, group_ids=groups), L("carried_over_report", rid, t + 0.1, attempt, carried_over_reported=True),
             L("optimizer_applied", rid, t + 1.0, attempt)] + ([L("outer_recorded", rid, t + 2.0, attempt)] if outer else [])
    return recs


def seqd(recs):
    for i, r in enumerate(recs): r["seq"] = i + 1
    return recs


def ledger_ok():
    """T36 L1-L8 shape for the r cases (restart at 110, restart publication version 1): rollouts 0 and 1 trained before the
    kill, rollout 1 never outer_recorded -> superseded at the restart (restart_rollout_id 1), regenerated as attempt 1 with
    new groups, then rollouts 2 and 3."""
    recs = ledger_rounds([], 0, 5.0, ["r0-g0", "r0-g1"])
    ledger_rounds(recs, 1, 25.0, ["r1-g0", "r1-g1"], outer=False)
    recs.append(L("superseded", 1, 139.5, 0, restart_rollout_id=1))
    ledger_rounds(recs, 1, 150.0, ["r1b-g0", "r1b-g1"], attempt=1); ledger_rounds(recs, 2, 160.0, ["r2-g0", "r2-g1"]); ledger_rounds(recs, 3, 170.0, ["r3-g0", "r3-g1"])
    return seqd(recs)


LEDGER_OK = ledger_ok()


def r5_journal(verified=True, seq_ok=True):
    j = [tx("VALIDATING", "up1"), tx("QUIESCING", "up1"), tx("INITIALIZING", "up1"), tx("VERIFYING", "up1"), tx("COMMITTED", "up1", wall_time=50.0),
         tx("SUCCEEDED", "up1", recovered_after_restart=True, wall_time=100.0),
         rec("planned", attempt=1, target=C4, actual=C4[:2], start=C4[2:], stop=[], wall_time=101.0),
         {"kind": "fork_op", "tx_id": "rec-1-1-abc", "op": "start", "status": "issued", "cells": C4[2:]},
         {"kind": "fork_op", "tx_id": "rec-1-1-abc", "op": "start", "status": "done", "cells": C4[2:], "result_fork_epoch": 2},
         rec("membership_restored", members=C4, fork_epoch=2, wall_time=130.0)]
    if verified: j.append(rec("verified", members=C4, fork_epoch=2, checks=VER_OK, wall_time=140.0))
    if not seq_ok: j.insert(6, rec("superseded"))
    return j


def r5_tape(restart=True, same_hash=True, rounds=3, recovered=True):
    t = [start(1.0), pub(0, "yeto:0:h0", C4[:2], 2.0), train(10.0), pub(1, "yeto:1:h1", C4[:2], 20.0), train(30.0)]
    if restart:
        t += [start(110.0), pub(1, "yeto:1:h1" if same_hash else "yeto:1:other", C4, 139.0)]
        if recovered: t.append({"event": "rl_reconfiguration", "result": "RECOVERED", "members": C4, "time_unix": 140.0})
        t += [train(150.0 + 10 * i) for i in range(rounds)]
    return t


class Recovery(unittest.TestCase):
    def test_r5_pass_and_each_check(self):
        r = J.judge_recovery("r5", r5_journal(), r5_tape(), set(C4), LEDGER_OK); self.assertEqual(r["verdict"], "PASS", r)
        self.assertEqual(J.judge_recovery("r5", r5_journal(), r5_tape(restart=False), set(C4), LEDGER_OK)["verdict"], "INVALID_TEST")   # kill not applied
        self.assertEqual(J.judge_recovery("r5", r5_journal(), r5_tape(restart=False), set(C4), LEDGER_OK, ["[yeto] learner exited 86; in-place restart 1/2"])["verdict"], "INVALID_TEST")   # loop line but no second driver start: identity unobservable
        r = J.judge_recovery("r5", r5_journal(verified=False), r5_tape(), set(C4), LEDGER_OK); self.assertEqual(r["verdict"], "INVALID_TEST")   # recovery never finished
        r = J.judge_recovery("r5", r5_journal(verified=False) + [rec("failed", errors=["x"]), {"kind": "phase", "phase": "RECOVERY_REQUIRED", "tx_id": "rec-1-1-abc"}], r5_tape(recovered=False), set(C4), LEDGER_OK)
        self.assertEqual((r["verdict"], r["marker"]), ("FAIL", "recovery_failed"))
        r = J.judge_recovery("r5", r5_journal(seq_ok=False), r5_tape(), set(C4), LEDGER_OK); self.assertEqual(r["verdict"], "FAIL"); self.assertFalse(r["checks"]["recovery_sequence"])
        r = J.judge_recovery("r5", r5_journal(), r5_tape(same_hash=False), set(C4), LEDGER_OK); self.assertEqual(r["verdict"], "FAIL"); self.assertFalse(r["checks"]["trainer_identity_same_policy_hash"])
        r = J.judge_recovery("r5", r5_journal(), r5_tape(rounds=2), set(C4), LEDGER_OK); self.assertEqual(r["verdict"], "FAIL"); self.assertFalse(r["checks"]["rounds_after_restart_ge_3"])
        dup = ledger_ok(); dup[-4]["group_ids"] = ["r0-g0", "r0-g1"]   # rollout 3 consumes rollout 0's groups again (L2, group-level)
        r = J.judge_recovery("r5", r5_journal(), r5_tape(), set(C4), dup); self.assertEqual(r["verdict"], "FAIL"); self.assertEqual(r["ledger_duplicates"], ["r0-g0", "r0-g1"])
        self.assertFalse(r["checks"]["L2_no_group_reuse"])
        # L1: a second optimizer_applied for one attempt; L6: the restart record names another rollout; L3: too few rounds after
        bad = ledger_ok() + [L("optimizer_applied", 3, 180.0)]; self.assertFalse(J.judge_recovery("r5", r5_journal(), r5_tape(), set(C4), seqd(bad))["checks"]["L1_round_triples"])
        bad = ledger_ok(); bad[7]["restart_rollout_id"] = 2; self.assertFalse(J.judge_recovery("r5", r5_journal(), r5_tape(), set(C4), bad)["checks"]["L6_restart_consistent"])
        self.assertFalse(J.judge_recovery("r5", r5_journal(), r5_tape(), set(C4), ledger_ok()[:-4])["checks"]["L3_continues_after_terminal"])
        self.assertEqual(J.judge_recovery("r5", r5_journal(), r5_tape(), set(C4), [])["verdict"], "INVALID_TEST")   # --ledger file empty/missing
        self.assertEqual(J.judge_recovery("r5", r5_journal(), r5_tape(), set(C4), None)["verdict"], "INVALID_TEST")   # ledger missing
        for bad in ({"policy_token": "unavailable"}, {"router": "unavailable"}, {"trainer_layout": "unavailable"}, {"unconsumed_batches": [3]}):
            j = r5_journal(verified=False) + [rec("verified", members=C4, fork_epoch=2, checks=dict(VER_OK, **bad), wall_time=140.0)]
            self.assertEqual(J.judge_recovery("r5", j, r5_tape(), set(C4), LEDGER_OK)["verdict"], "FAIL", bad)
        # tx SUCCEEDED but not marked recovered_after_restart (the commit happened after the restart: not ⑤)
        j = r5_journal(); j[5] = tx("SUCCEEDED", "up1", wall_time=100.0)
        r = J.judge_recovery("r5", j, r5_tape(), set(C4), LEDGER_OK); self.assertFalse(r["checks"]["up1_SUCCEEDED_recovered_after_restart"])
        # probe: epoch must match the verified fork epoch and all 4 cells Running
        good = {"probe_attested": True, "membership": {"epoch": 2}, "cell_statuses": {c: "phase='Running'" for c in ("c0", "c1", "c2", "c3")}}
        self.assertEqual(J.judge_recovery("r5", r5_journal(), r5_tape(), set(C4), LEDGER_OK, None, good)["verdict"], "PASS")
        self.assertEqual(J.judge_recovery("r5", r5_journal(), r5_tape(), set(C4), LEDGER_OK, None, dict(good, membership={"epoch": 1}))["verdict"], "FAIL")

    def test_r7(self):
        j = [tx("VALIDATING", "up1"), tx("COMMITTED", "up1"), tx("SUCCEEDED", "up1", wall_time=50.0),
             {"kind": "reconcile", "action": "restore_membership_state", "epoch": 1}] + r5_journal()[6:] + \
            [tx("VALIDATING", "dn1", wall_time=200.0), tx("SUCCEEDED", "dn1", wall_time=260.0)]
        t = r5_tape(); t.append(pub(2, "yeto:2:h2", C4[:2], 270.0))
        r = J.judge_recovery("r7", j, t, set(C4), LEDGER_OK); self.assertEqual(r["verdict"], "PASS", r)
        r = J.judge_recovery("r7", [x for x in j if x.get("kind") != "reconcile"], t, set(C4), LEDGER_OK); self.assertEqual(r["verdict"], "FAIL")
        r = J.judge_recovery("r7", j[:-1], t, set(C4), LEDGER_OK); self.assertEqual(r["verdict"], "INVALID_TEST")   # dn1 never terminal

    def test_r6(self):
        j = [tx("VALIDATING", "up1"), tx("WAIT_SAFE", "up1"), tx("QUIESCING", "up1", wall_time=40.0), tx("CANCELLED", "up1", error="learner restarted before release", wall_time=100.0)]
        t = r5_tape(recovered=False)
        self.assertEqual(J.judge_recovery("r6", j, t, set(C4[:2]), LEDGER_OK)["verdict"], "PASS")
        r = J.judge_recovery("r6", j + [rec("planned")], t, set(C4[:2]), LEDGER_OK); self.assertEqual(r["verdict"], "FAIL"); self.assertFalse(r["checks"]["no_recovery"])
        r = J.judge_recovery("r6", j[:3], t, set(C4[:2]), LEDGER_OK); self.assertEqual(r["verdict"], "INVALID_TEST")
        r = J.judge_recovery("r6", j[:2] + [tx("CANCELLED", "up1")], t, set(C4[:2]), LEDGER_OK); self.assertFalse(r["checks"]["killed_at_QUIESCING"])

    def test_r5c(self):
        j = [tx("VALIDATING", "up1"), tx("SUCCEEDED", "up1"), tx("VALIDATING", "dn1"), tx("COMMITTED", "dn1"), tx("SUCCEEDED", "dn1", recovered_after_restart=True)]
        t = [start(1.0), pub(0, "yeto:0:h0", C4[:2], 2.0), pub(1, "yeto:1:h1", C4, 20.0), train(30.0), start(110.0), pub(1, "yeto:1:h1", C4[:2], 139.0)] + [train(150.0 + 10 * i) for i in range(3)]
        self.assertEqual(J.judge_recovery("r5c", j, t, set(C4), LEDGER_OK)["verdict"], "PASS")
        r = J.judge_recovery("r5c", j + [rec("planned")], t, set(C4), LEDGER_OK); self.assertFalse(r["checks"]["no_recovery"])
        t2 = list(t); t2[5] = pub(1, "yeto:1:h1", C4, 139.0)
        r = J.judge_recovery("r5c", j, t2, set(C4), LEDGER_OK); self.assertFalse(r["checks"]["startup_shape_after_restart"])

    def test_replay_cpu_r5(self):
        """Replay of the CPU recovery test (tests/test_rl_reconfig_recovery.py::test_restart_after_commit_recovers_committed_members,
        infra-e1-recovery 0e68962): journal + tape + ledger of a fake island whose up committed, learner restarted, members rebuilt.
        Shape: the up committed and finished normally in the first process, then the learner restarted with the committed
        config (not a kill at COMMITTED), the fake trainer has no actual_layout and the CPU restart ran without a ledger:
        exactly those three §10.4 checks fail by design; everything else (recovery sequence, verified checks, RECOVERED,
        identity, rounds, no duplicate consumption) passes on the real journal/tape."""
        j, t, l = jl("cpu_r5_journal.jsonl"), jl("cpu_r5_tape.jsonl"), jl("cpu_r5_ledger.jsonl")
        r = J.judge_recovery("r5", j, t, set(C4), l, up="up")   # the CPU test's request id
        self.assertEqual(r["verdict"], "FAIL", r)
        failed = sorted(k for k, v in r["checks"].items() if v is not True)
        # 2026-10-02 T36 ledger checks: the restarted CPU process ran without a ledger, so L1 (tape 6 rounds trained vs 3
        # optimizer_applied), L6 (no ledger record after the restart) and L7 (verified.checks has no unconsumed_batches)
        # fail on this fixture by the same design as no_unconsumed_batches.
        self.assertEqual(failed, ["L1_round_triples", "L6_restart_consistent", "L7_verified_unconsumed_empty",
                                  "no_unconsumed_batches", "trainer_layout_world_4", "up_SUCCEEDED_recovered_after_restart"], r["checks"])
        self.assertTrue(r["checks"]["L2_no_group_reuse"] and r["checks"]["L8_report_fields"] and r["checks"]["L3_continues_after_terminal"])
        self.assertTrue(r["checks"]["recovery_sequence"] and r["checks"]["tape_RECOVERED"] and r["checks"]["router_all_admitted"])
        self.assertEqual(r["identity"]["version"], 0)   # LocalOnlySync restarts at 0: same initial policy hash
        self.assertEqual(r["ledger_duplicates"], [])
        # main() path with files
        with tempfile.TemporaryDirectory() as d:
            for n in ("cpu_r5_journal.jsonl", "cpu_r5_tape.jsonl", "cpu_r5_ledger.jsonl"):
                open(os.path.join(d, n), "w").write(open(os.path.join(FX, n)).read())
            rc = J.main(["r5", os.path.join(d, "cpu_r5_journal.jsonl"), os.path.join(d, "cpu_r5_tape.jsonl"), "--ledger", os.path.join(d, "cpu_r5_ledger.jsonl"), "--out", os.path.join(d, "o.json"), "--marker-dir", d])
            self.assertEqual(rc, 4); self.assertTrue(os.path.exists(os.path.join(d, "evidence_missing")))   # main() judges request id up1 (a8go's)


if __name__ == "__main__":
    unittest.main()
