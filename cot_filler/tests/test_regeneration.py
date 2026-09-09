"""All model/HTTP responses here are explicit test doubles, never quality evidence."""
import copy
import io
import json
from pathlib import Path
import sqlite3
import unittest
from unittest.mock import patch

from cot_filler import grounding_review_v4 as policy
from cot_filler.core import canonical, digest, gaps_for
from cot_filler.regeneration import (MAX_ATTEMPTS, PROMPT_VERSION, PrefixRegenerationProvider,
    RegenerationLedger, regeneration_messages, request_regeneration)


class Tokenizer:
    identity = "unit-test-tokenizer"
    def count(self, messages, options):
        return 100


class RegenerationTests(unittest.TestCase):
    def setUp(self):
        trace = json.loads((Path(__file__).resolve().parents[1] / "examples/demo.json").read_text())
        trace["events"][0]["content"] += " SAFE_PREFIX_SENTINEL"
        trace["events"][2]["content"] += " FUTURE_EVENT_SENTINEL"
        self.gap = next(gaps_for(trace))
        self.rejected = "I already observed the result REJECTED_CANDIDATE_SENTINEL."
        self.good = "Inspect the function before choosing a repair."
        self.config = {"model": "unit-test-model", "base_url": "http://127.0.0.1:9999/v1",
                       "tokenizer_path": "/unused-unit-test", "context_limit": 4096,
                       "max_output_tokens": 128, "chat_template_matches_server": True,
                       "chat_template_kwargs": {"thinking": True}}
        self.review_config = {**self.config, "review_version": policy.REVIEW_VERSION}
        self.db = sqlite3.connect(":memory:")
        self.addCleanup(self.db.close)
        self.db.execute("CREATE TABLE candidates(id TEXT,text TEXT)")
        self.db.execute("INSERT INTO candidates VALUES('original',?)", (self.rejected,))
        self.ledger = RegenerationLedger(self.db, self.config, self.review_config)
        self.rejection = self.review(self.rejected, "reject")

    def review(self, candidate, decision, *, grounding_failure=True):
        contract, evidence = policy.review_contract(self.gap, candidate)
        eid = next(i for i, item in evidence.items() if "Inspect the function before editing." in item["quote"])
        checks = dict.fromkeys(policy.CHECKS, True)
        if decision != "pass":
            checks["all_factual_claims_supported_by_prefix"] = not grounding_failure
            checks["no_future_observation_claims"] = not grounding_failure
            checks["specific_and_concise"] = False
        raw = {"schema": policy.REVIEW_VERSION, "decision": decision, "checks": checks,
               "statements": [{"id": unit["id"], "kind": "plan" if decision == "pass" else "observation",
                               "assessment": {"pass": "supported", "reject": "unsupported", "uncertain": "uncertain"}[decision],
                               "evidence_ids": [eid] if decision == "pass" else []} for unit in contract["candidate_units"]],
               "note": "NOTE_FUTURE_SENTINEL; test double only, not an actual model review."}
        reviewer = {"provider": "openai-compatible", "model": self.config["model"], "finish_reason": "stop",
                    "prompt_tokens_local": 100, "usage": {"prompt_tokens": 100},
                    "parameters": {"chat_template_kwargs": self.config["chat_template_kwargs"]}}
        record = policy.validate_review(self.gap, candidate, raw, reviewer=reviewer)
        record.update(prompt_token_parity_verified=True, configured_template_options_verified=True,
                      configured_decoding_controls_verified=True)
        return record

    def metadata(self, request, **changes):
        data = {"generator": {"provider": "openai-compatible", "model": self.config["model"],
                              "response_model": self.config["model"],
                              "prompt_version": PROMPT_VERSION,
                              "parameters": {"model": self.config["model"], "max_tokens": 128, "temperature": 0.3,
                                             "chat_template_kwargs": self.config["chat_template_kwargs"]}},
                "attempt_id": request["attempt_id"], "prompt_hash": digest(regeneration_messages(self.gap, request)),
                "prompt_tokens_local": 100, "usage": {"prompt_tokens": 100}, "finish_reason": "stop",
                "prefix_only": True, "lookahead_conditioned": False, "flags": []}
        return {**data, **changes}

    def request(self):
        return self.ledger.reserve(self.gap, self.rejected, self.rejection)

    def test_prefix_only_prompt_excludes_target_candidate_notes_and_citations(self):
        request = self.request()
        text = canonical(regeneration_messages(self.gap, request))
        self.assertIn("SAFE_PREFIX_SENTINEL", text)
        for sentinel in ("FUTURE_EVENT_SENTINEL", "REJECTED_CANDIDATE_SENTINEL", "NOTE_FUTURE_SENTINEL"):
            self.assertNotIn(sentinel, text)
        self.assertEqual(request["gap_id"], self.gap["id"])
        self.assertEqual(request["event_id"], self.gap["event_id"])
        self.assertEqual(set(request["feedback"]), {"reason_codes", "unit_ids"})

    def test_feedback_and_placement_are_fail_closed_even_with_rehashed_request(self):
        request = self.request()
        for modify in (lambda r: r["feedback"].update(note="execute injected instruction"),
                       lambda r: r["feedback"].update(reason_codes=["ignore_all_constraints"]),
                       lambda r: r["feedback"].update(unit_ids=["c0; FUTURE_EVENT_SENTINEL"]),
                       lambda r: r.update(event_id="different-event"),
                       lambda r: r.update(attempt_number=MAX_ATTEMPTS+1)):
            bad = copy.deepcopy(request);modify(bad)
            bad["attempt_id"] = digest({k: v for k, v in bad.items() if k != "attempt_id"})
            with self.assertRaises(ValueError):regeneration_messages(self.gap, bad)

    def test_only_complete_genuine_grounding_rejections_are_eligible(self):
        for decision, grounding in (("pass", True), ("uncertain", True), ("reject", False)):
            record = self.review(self.good, decision, grounding_failure=grounding)
            self.assertIsNone(request_regeneration(self.gap, self.good, record, self.review_config, attempt_number=1))
        for mutate in (lambda r: r.update(candidate_hash="wrong"),
                       lambda r: r.update(prefix_hash="wrong"),
                       lambda r: r.update(prompt_token_parity_verified=False),
                       lambda r: r.update(prompt_token_parity_verified=1),
                       lambda r: r.update(future_events_supplied_to_reviewer=0),
                       lambda r: r["reviewer"].update(finish_reason="length"),
                       lambda r: r["reviewer"].update(provider="fake"),
                       lambda r: r["reviewer"]["usage"].update(prompt_tokens=99)):
            bad=copy.deepcopy(self.rejection);mutate(bad)
            with self.assertRaises(ValueError):
                request_regeneration(self.gap, self.rejected, bad, self.review_config, attempt_number=1)
        with self.assertRaises(ValueError):
            request_regeneration(self.gap, self.rejected, {"decision": "transport_error"}, self.review_config, attempt_number=1)
        self.assertEqual(self.db.execute("SELECT COUNT(*) FROM cot_regen_attempts").fetchone()[0],0)

    def test_fresh_candidate_needs_same_review_and_never_overwrites_original(self):
        request=self.request();aid=request["attempt_id"]
        state=self.ledger.generation_done(self.gap,aid,self.good,self.metadata(request))
        self.assertEqual(state,"review_pending")
        self.assertEqual(self.db.execute("SELECT COUNT(*) FROM cot_regen_events WHERE kind='review'").fetchone()[0],0)
        with self.assertRaises(ValueError):self.ledger.review_done(self.gap,aid,self.rejection)
        self.assertEqual(self.ledger.review_done(self.gap,aid,self.review(self.good,"pass")),"accepted")
        self.assertEqual(self.db.execute("SELECT text FROM candidates").fetchone()[0],self.rejected)
        with self.assertRaises(sqlite3.IntegrityError):
            self.ledger.generation_done(self.gap,aid,"overwrite",self.metadata(request))
        with self.assertRaises(sqlite3.IntegrityError):
            self.ledger.review_done(self.gap,aid,self.review(self.good,"pass"))
        with self.assertRaises(ValueError):self.ledger.failure(aid,"transport")

    def test_second_attempt_requires_exact_parent_rejection_and_cap_is_two(self):
        first=self.request();a1=first["attempt_id"]
        with self.assertRaises(ValueError):self.ledger.reserve(self.gap,self.rejected,self.rejection)
        self.ledger.generation_done(self.gap,a1,self.good,self.metadata(first))
        rejection=self.review(self.good,"reject")
        self.ledger.review_done(self.gap,a1,rejection)
        with self.assertRaises(ValueError):self.ledger.reserve(self.gap,self.rejected,self.rejection)
        second=self.ledger.reserve(self.gap,self.good,rejection)
        self.assertEqual(second["attempt_number"],2)
        self.assertEqual(second["parent_attempt_id"],a1)
        self.ledger.generation_done(self.gap,second["attempt_id"],self.good,self.metadata(second))
        self.ledger.review_done(self.gap,second["attempt_id"],rejection)
        self.assertIsNone(self.ledger.reserve(self.gap,self.good,rejection))
        self.assertEqual(self.db.execute("SELECT COUNT(*) FROM cot_regen_attempts").fetchone()[0],2)
        self.assertEqual(self.db.execute("SELECT COUNT(*) FROM cot_regen_events").fetchone()[0],4)

    def test_invalid_generation_uncertainty_and_transport_failure_stay_excluded(self):
        request=self.request();aid=request["attempt_id"]
        state=self.ledger.generation_done(self.gap,aid,self.good,self.metadata(request,finish_reason="length"))
        self.assertEqual(state,"generation_invalid")
        with self.assertRaises(ValueError):self.ledger.review_done(self.gap,aid,self.review(self.good,"pass"))
        with self.assertRaises(ValueError):self.ledger.reserve(self.gap,self.good,self.review(self.good,"reject"))

    def test_terminal_failure_cannot_later_be_accepted_and_config_is_immutable(self):
        request=self.request();aid=request["attempt_id"]
        self.ledger.failure(aid,"transport")
        with self.assertRaises(ValueError):self.ledger.generation_done(self.gap,aid,self.good,self.metadata(request))
        with self.assertRaises(ValueError):self.ledger.review_done(self.gap,aid,self.review(self.good,"pass"))
        with self.assertRaises(ValueError):RegenerationLedger(self.db,{**self.config,"model":"different"},self.review_config)
        restarted=RegenerationLedger(self.db,self.config,self.review_config)
        with self.assertRaises(ValueError):restarted.reserve(self.gap,self.rejected,self.rejection)
        self.assertEqual(self.db.execute("SELECT COUNT(*) FROM cot_regen_attempts").fetchone()[0],1)

    def test_code_identity_change_cannot_silently_resume_same_ledger(self):
        with patch("cot_filler.regeneration.implementation_identity",return_value={"regeneration.py":"different-code"}):
            with self.assertRaisesRegex(ValueError,"configuration changed"):
                RegenerationLedger(self.db,self.config,self.review_config)

    def test_actual_generation_controls_must_match_immutable_config(self):
        for key,value in (("model","other-model"),("max_tokens",256),("temperature",0.9),
                          ("chat_template_kwargs",{"thinking":False}),("reasoning_effort","high")):
            with self.subTest(key=key):
                db=sqlite3.connect(":memory:")
                try:
                    ledger=RegenerationLedger(db,self.config,self.review_config)
                    request=ledger.reserve(self.gap,self.rejected,self.rejection)
                    metadata=self.metadata(request)
                    metadata["generator"]["parameters"][key]=value
                    self.assertEqual(ledger.generation_done(self.gap,request["attempt_id"],self.good,metadata),"generation_invalid")
                    with self.assertRaises(ValueError):ledger.review_done(self.gap,request["attempt_id"],self.review(self.good,"pass"))
                finally:db.close()

    def test_actual_server_model_must_match_for_regeneration(self):
        request=self.request();metadata=self.metadata(request)
        metadata["generator"]["response_model"]="unexpected-backend-model"
        self.assertEqual(self.ledger.generation_done(self.gap,request["attempt_id"],self.good,metadata),"generation_invalid")
        with self.assertRaises(ValueError):
            self.ledger.review_done(self.gap,request["attempt_id"],self.review(self.good,"pass"))

    def test_uncertain_regenerated_review_does_not_request_another_attempt(self):
        request=self.request();aid=request["attempt_id"]
        self.ledger.generation_done(self.gap,aid,self.good,self.metadata(request))
        uncertain=self.review(self.good,"uncertain")
        self.assertEqual(self.ledger.review_done(self.gap,aid,uncertain),"uncertain")
        with self.assertRaises(ValueError):self.ledger.reserve(self.gap,self.good,uncertain)

    def test_mocked_provider_wire_is_prefix_only_and_has_no_tools(self):
        request=self.request();bodies=[]
        payload={"model":self.config["model"],"choices":[{"message":{"content":self.good},"finish_reason":"stop"}],"usage":{"prompt_tokens":100}}
        class Opener:
            def open(inner,wire,timeout):
                bodies.append(json.loads(wire.data));return io.BytesIO(json.dumps(payload).encode())
        with patch("cot_filler.regeneration.urllib.request.build_opener",return_value=Opener()):
            text,metadata=PrefixRegenerationProvider(self.config,Tokenizer()).generate(self.gap,request)
        self.assertEqual(text,self.good)
        self.assertEqual(metadata["generator"]["prompt_version"],PROMPT_VERSION)
        self.assertFalse(metadata["lookahead_conditioned"])
        self.assertNotIn("tools",bodies[0]);self.assertNotIn("tool_choice",bodies[0])
        for sentinel in ("FUTURE_EVENT_SENTINEL","REJECTED_CANDIDATE_SENTINEL","NOTE_FUTURE_SENTINEL"):
            self.assertNotIn(sentinel,canonical(bodies[0]))
        self.assertEqual(self.ledger.generation_done(self.gap,request["attempt_id"],text,metadata),"review_pending")


if __name__ == "__main__":unittest.main()
