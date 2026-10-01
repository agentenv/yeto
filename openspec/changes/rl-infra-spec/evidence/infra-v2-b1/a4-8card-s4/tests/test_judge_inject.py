"""CPU test of judge_inject with synthetic journal/tape. run: python3 tests/test_judge_inject.py"""
import json, os, sys, tempfile, unittest
sys.path[:0] = [os.path.join(os.path.dirname(__file__), ".."), os.path.join(os.path.dirname(__file__), "..", "scripts")]
import judge_inject as J


def ph(p, **k): return {"kind": "phase", "phase": p, **k}
def inj(**k): return {"event": "test_injection", **{"kind": "lora_perturb", "target_members": ["c1"], "phase": "publish", "ts": 10.0, "applied": True, **k}}
ADD = {"kind": "add_intent", "members": ["c1", "c2"]}


class T(unittest.TestCase):
    def e1b(self, tape, journal, samples=None):
        return J.judge_e1b(journal, tape, {"c0", "c1", "c2"}, samples)

    def test_e1b_pass(self):
        r = self.e1b([inj()], [ADD, ph("REBUILD_OLD", cause="payload_mismatch"), ph("REBUILT_OLD", cause="payload_mismatch")])
        self.assertEqual(r["verdict"], "PASS")

    def test_e1b_not_applied_is_invalid_not_failure(self):
        r = self.e1b([inj(applied=False)], [ADD, ph("SUCCEEDED")])
        self.assertEqual((r["verdict"], r["marker"]), ("INVALID_TEST", "injection_not_reached"))

    def test_e1b_missing_injection_event_invalid(self):
        self.assertEqual(self.e1b([], [ADD, ph("SUCCEEDED")])["verdict"], "INVALID_TEST")

    def test_e1b_target_absent_invalid(self):
        r = self.e1b([inj(target_members=["c9"])], [ADD, ph("REBUILT_OLD", cause="payload_mismatch")])
        self.assertEqual(r["verdict"], "INVALID_TEST"); self.assertIn("does not exist", r["invalid_reasons"][0])

    def test_e1b_wrong_cause_fails(self):
        r = self.e1b([inj()], [ADD, ph("REBUILT_OLD", cause="start_failed")])
        self.assertEqual(r["verdict"], "FAIL"); self.assertFalse(r["checks"]["cause_payload_mismatch"])

    def test_e1b_recovery_required_is_recovery_failed(self):
        r = self.e1b([inj()], [ADD, ph("RECOVERY_REQUIRED")])
        self.assertEqual((r["verdict"], r["marker"]), ("FAIL", "recovery_failed"))

    def test_e1b_hold_window_sampling(self):
        hold = {"event": "test_hold", "stage": "end", "ts": 110.0, "start_ts": 100.0, "end_ts": 110.0}
        hold_start = {"event": "test_hold", "stage": "start", "ts": 100.0}
        s = lambda t, n: {"t": t, "data": {"inflight": {"http://a": 1, "http://new": n}}}
        good = [{"t": 90.0, "data": {"inflight": {"http://a": 1}}}] + [s(101 + i, 0) for i in range(5)]
        j = [ADD, ph("REBUILT_OLD", cause="payload_mismatch")]
        self.assertEqual(self.e1b([inj(), hold], j, good)["verdict"], "PASS")
        bad = good[:3] + [s(104, 2)] + good[3:]
        self.assertEqual(self.e1b([inj(), hold], j, bad)["verdict"], "FAIL")
        self.assertEqual(self.e1b([inj(), hold], j, good[:2])["verdict"], "INVALID_TEST")   # window not sampled enough
        self.assertEqual(self.e1b([inj()], j, good)["verdict"], "INVALID_TEST")             # no test_hold event
        self.assertEqual(self.e1b([inj(), hold_start], j, good)["verdict"], "INVALID_TEST")  # start only, hold never completed

    def wd(self, wdrec, tape):
        return J.judge_wd([ADD, {"kind": "watchdog", **wdrec}, {"kind": "watchdog_action", "killed": True}, ph("REBUILT_OLD")], tape, set())

    def test_wd_ok_and_invalid(self):
        blk = lambda **k: {"event": "test_injection", "kind": "block_update", "applied": True, "target_members": None, "ts": 1.0, **k}
        self.assertEqual(self.wd({"classification": "FIRED_ON_BLOCKED_UPDATE", "injection_reached": True}, [blk(reached_ts=5.0)])["verdict"], "PASS")
        r = self.wd({"classification": "INJECTION_NOT_REACHED", "injection_reached": False}, [blk()])
        self.assertEqual((r["verdict"], r["marker"]), ("INVALID_TEST", "injection_not_reached"))
        self.assertEqual(self.wd({"classification": "FIRED_ON_BLOCKED_UPDATE"}, [blk()])["verdict"], "INVALID_TEST")  # no reached_ts

    def test_wd_fired_after_release_is_invalid_and_journal_form(self):
        blk = {"event": "test_injection", "kind": "block_update", "applied": True, "target_members": None, "ts": 1.0, "reached_ts": 5.0, "released_ts": 9.0}
        self.assertEqual(self.wd({"classification": "FIRED_AFTER_BLOCK_RELEASED", "injection_reached": True}, [blk])["verdict"], "INVALID_TEST")
        jb = {"kind": "test_injection", "injection_kind": "block_update", "applied": True, "reached_ts": 5.0, "ts": 1.0}   # journal form
        r = J.judge_wd([ADD, {"kind": "watchdog", "classification": "FIRED_ON_BLOCKED_UPDATE", "injection_reached": True}, {"kind": "watchdog_action", "killed": True}, ph("REBUILT_OLD"), jb], [], set())
        self.assertEqual(r["verdict"], "PASS")

    def test_a4b(self):
        j = [ph("CANCELLED")]
        t = {"event": "test_injection", "kind": "tool_wait", "applied": True, "ts": 1.0, "target_members": None}
        self.assertEqual(J.judge_a4b(j, [t], set())["verdict"], "PASS")
        self.assertEqual(J.judge_a4b(j, [dict(t, applied=False)], set())["verdict"], "INVALID_TEST")
        self.assertEqual(J.judge_a4b([ph("SUCCEEDED")], [t], set())["verdict"], "FAIL")

    def test_e1a_c(self):
        # synthetic, rollout_id units: up serves rollout 2 (= round 3), down serves rollout 4 (= round 5); 4-card counts [1,1,3,3,1,1]
        mem = lambda rid, n, k: {"event": "rl_membership", "round": rid, "members": ["m%d" % i for i in range(n)], "kind": k, "config_epoch": 0, "tx_id": "t"}
        tape = [{"event": "rl_publication", "sync/publication_members": ["m0"]}, mem(2, 3, "up"), {"event": "rl_publication", "sync/publication_members": ["m0", "m1", "m2"]},
                mem(4, 1, "down"), {"event": "rl_publication", "sync/publication_members": ["m0"]}]
        r = J.judge_e1a_c([], tape, [1, 1, 3, 3, 1, 1]); self.assertEqual(r["verdict"], "PASS", r)
        r = J.judge_e1a_c([], tape, [1, 1, 3, 3, 3, 1]); self.assertEqual(r["verdict"], "FAIL")
        tape[-1]["sync/publication_members"] = ["m0", "m1", "m2"]
        r = J.judge_e1a_c([], tape, [1, 1, 3, 3, 1, 1]); self.assertEqual(r["verdict"], "FAIL"); self.assertFalse(r["checks"]["publication_after_down_has_final_member_count"])
        self.assertEqual(J.judge_e1a_c([], [], [1])["verdict"], "INVALID_TEST")

    def test_e1a_c_real_8card_tape(self):
        """real tape fragment of infra-v2-b1-a8e1a-20261001-1 (8xH100, 12 rounds): up -> rl_membership.round=2, down -> round=7 (0-based rollout ids)."""
        fx = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures", "a8e1a_tape_fragment.jsonl")
        tape = [json.loads(l) for l in open(fx) if l.strip()]
        exp = [2, 2, 4, 4, 4, 4, 4, 2, 2, 2, 2, 2]
        r = J.judge_e1a_c([], tape, exp); self.assertEqual(r["verdict"], "PASS", r); self.assertEqual(r["members_per_round"], exp)
        old = J.judge_e1a_c([], tape, exp, round_is_rollout_id=False)   # the first reading (`round <= r`, no initial members): fails on the same data
        self.assertEqual(old["verdict"], "FAIL"); self.assertEqual(old["members_per_round"], [None, 4, 4, 4, 4, 4, 2, 2, 2, 2, 2, 2])
        bad = J.judge_e1a_c([], tape, [2, 2, 2, 4, 4, 4, 4, 2, 2, 2, 2, 2]); self.assertEqual(bad["verdict"], "FAIL")   # a genuinely wrong sequence is still caught

    def test_cli_writes_marker(self):
        d = tempfile.mkdtemp(); jp, tp = os.path.join(d, "j.jsonl"), os.path.join(d, "t.jsonl")
        open(jp, "w").write(json.dumps(ADD) + "\n" + json.dumps(ph("SUCCEEDED")) + "\n"); open(tp, "w").write(json.dumps(inj(applied=False)) + "\n")
        rc = J.main(["e1b", jp, tp, "--marker-dir", d, "--out", os.path.join(d, "o.json")])
        self.assertEqual(rc, 4); self.assertTrue(os.path.exists(os.path.join(d, "injection_not_reached")))


if __name__ == "__main__":
    unittest.main()
