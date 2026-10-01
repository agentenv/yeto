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
        j = [tx("VERIFYING", "up1", wall_time=100.0), tx("REBUILT_OLD", "up1", wall_time=130.0)]
        k = [{"event": "kill", "rule": "d2", "gpu": 6, "res": {"300": "killed"}, "wall": 100.5}]
        self.assertEqual(J.judge_d2(j, [], set(), k)["verdict"], "PASS")
        self.assertEqual(J.judge_d2(j, [], set(), [])["verdict"], "INVALID_TEST")
        self.assertEqual(J.judge_d2([tx("VERIFYING", "up1", wall_time=100.0), tx("SUCCEEDED", "up1")], [], set(), k)["verdict"], "FAIL")
        self.assertEqual(J.judge_d2([tx("VERIFYING", "up1", wall_time=100.0)], [], set(), k)["verdict"], "INVALID_TEST")

    def test_d4(self):
        j = [tx("SUCCEEDED", "up1"), {"kind": "fork_op", "op": "stop", "status": "incomplete", "tx_id": "dn1"}, tx("RECOVERY_REQUIRED", "dn1", wall_time=50.0)]
        t = [{"event": "rl_reconfiguration", "result": "RECOVERY_REQUIRED"}]
        self.assertEqual(J.judge_d4(j, t, set())["verdict"], "PASS")
        self.assertEqual(J.judge_d4(j, t + [{"event": "rl_batch_prepared", "ts": 60.0}], set())["verdict"], "FAIL")
        self.assertEqual(J.judge_d4([j[0], tx("SUCCEEDED", "dn1")], t, set())["verdict"], "INVALID_TEST")
        self.assertEqual(J.judge_d4(j, [], set())["verdict"], "FAIL")


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


if __name__ == "__main__":
    unittest.main()
