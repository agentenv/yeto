"""Synthetic protocol tests only: no real generation or semantic quality claim."""
import copy
import fcntl
import io
import json
from pathlib import Path
import sqlite3
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

from cot_filler.core import canonical, digest
from cot_filler.corpus_worker import Journal, _code_identity
from cot_filler.corpus_source import file_sha256
from cot_filler.provider import ContextOverflow
from cot_filler.regeneration import RegenerationLedger
from cot_filler.regeneration_worker import (ReadOnlyRejectedSource, _cohort_result, _strict_receipt,
    accepted_selections, main, run_regenerations)
from cot_filler.tests import test_regeneration as fixtures


class Generator:
    def __init__(self, fixture, stop=None):
        self.fixture = fixture
        self.config = fixture.config
        self.tokenizer = fixtures.Tokenizer()
        self.calls = 0
        self.stop = stop

    def generate(self, gap, request):
        self.calls += 1
        metadata = self.fixture.metadata(request)
        metadata["generator"]["tokenizer_sha256"] = self.tokenizer.identity
        if self.stop is not None:
            self.stop.set()
        return self.fixture.good, metadata


class Reviewer:
    def __init__(self, fixture, decisions=("pass",)):
        self.fixture = fixture
        self.config = fixture.review_config
        self.tokenizer = fixtures.Tokenizer()
        self.decisions = list(decisions)
        self.calls = 0

    def review(self, gap, candidate):
        decision = self.decisions[min(self.calls, len(self.decisions)-1)]
        self.calls += 1
        record = self.fixture.review(candidate, decision)
        raw = {key: record[key] for key in ("schema", "decision", "checks", "note")}
        raw["statements"] = [{key: s[key] for key in ("id", "kind", "assessment", "evidence_ids")}
                             for s in record["statements"]]
        metadata = record["reviewer"]
        metadata["response_model"] = self.config["model"]
        metadata["parameters"].update(model=self.config["model"], max_tokens=self.config["max_output_tokens"], temperature=0.0)
        metadata["tokenizer_sha256"] = self.tokenizer.identity
        return json.dumps(raw), metadata


class WorkerTests(unittest.TestCase):
    def setUp(self):
        self.f = fixtures.RegenerationTests()
        self.f.setUp()
        self.addCleanup(self.f.doCleanups)
        self.generator = Generator(self.f)
        self.reviewer = Reviewer(self.f)
        self.entries = [(self.f.gap, self.f.rejected, self.f.rejection)]

    def run_worker(self, **kwargs):
        return run_regenerations(self.entries, self.f.ledger, self.generator, self.reviewer, max_gaps=1, **kwargs)

    def test_real_contract_double_accepts_only_fresh_review_without_changing_original(self):
        result = self.run_worker()
        self.assertEqual(result["states"], {"accepted": 1})
        self.assertEqual((self.generator.calls, self.reviewer.calls), (1, 1))
        self.assertEqual(len(list(accepted_selections(self.f.ledger))), 1)
        self.assertEqual(self.f.db.execute("SELECT text FROM candidates").fetchone()[0], self.f.rejected)
        self.run_worker()
        self.assertEqual((self.generator.calls, self.reviewer.calls), (1, 1))

    def test_only_two_grounding_attempts_then_stop(self):
        self.reviewer = Reviewer(self.f, ("reject",))
        result = self.run_worker()
        self.assertEqual(result["states"], {"rejected": 1})
        self.assertEqual((self.generator.calls, self.reviewer.calls), (2, 2))
        self.run_worker()
        self.assertEqual((self.generator.calls, self.reviewer.calls), (2, 2))
        self.assertEqual(list(accepted_selections(self.f.ledger)), [])

    def test_uncertain_does_not_retry(self):
        self.reviewer = Reviewer(self.f, ("uncertain",))
        self.assertEqual(self.run_worker()["states"], {"uncertain": 1})
        self.assertEqual(self.generator.calls, 1)

    def test_approved_style_only_and_malformed_originals_do_not_call_model(self):
        for decision, grounding in (("pass", True), ("uncertain", True), ("reject", False)):
            self.entries = [(self.f.gap, self.f.good, self.f.review(self.f.good, decision, grounding_failure=grounding))]
            self.assertEqual(self.run_worker()["states"], {"ineligible": 1})
        self.entries = [(self.f.gap, self.f.rejected, {"decision": "reject"})]
        self.assertEqual(self.run_worker()["states"], {"invalid_original_review": 1})
        self.assertEqual((self.generator.calls, self.reviewer.calls), (0, 0))

    def test_drain_committed_generation_resumes_at_review_not_regeneration(self):
        stop = threading.Event()
        self.generator.stop = stop
        self.assertEqual(self.run_worker(stop=stop)["states"], {"pending": 1})
        self.assertEqual((self.generator.calls, self.reviewer.calls), (1, 0))
        self.generator.stop = None
        self.assertEqual(self.run_worker()["states"], {"accepted": 1})
        self.assertEqual((self.generator.calls, self.reviewer.calls), (1, 1))

    def test_unknown_interrupted_request_is_excluded_without_retry(self):
        request = self.f.request()
        self.f.ledger._append(request["attempt_id"], "generate_started", {"at": "test"})
        self.assertEqual(self.run_worker()["states"], {"excluded": 1})
        self.assertEqual((self.generator.calls, self.reviewer.calls), (0, 0))
        self.assertEqual(self.f.ledger._event(request["attempt_id"], "failure")["category"], "interrupted")

    def test_persisted_review_response_is_validated_without_http_replay(self):
        request = self.f.request(); aid = request["attempt_id"]
        text, metadata = self.generator.generate(self.f.gap, request)
        self.f.ledger.generation_done(self.f.gap, aid, text, metadata)
        raw, receipt = self.reviewer.review(self.f.gap, text)
        self.f.ledger._append(aid, "review_started", {"at": "test"})
        self.f.ledger._append(aid, "review_response", {"raw": raw, "metadata": receipt})
        self.assertEqual(self.run_worker()["states"], {"accepted": 1})
        self.assertEqual((self.generator.calls, self.reviewer.calls), (1, 1))

    def test_generation_response_saved_before_validation_resumes_without_generation(self):
        request=self.f.request();aid=request["attempt_id"]
        text,metadata=self.generator.generate(self.f.gap,request)
        self.f.ledger._append(aid,"generate_started",{"at":"test"})
        self.f.ledger._append(aid,"generation_response",{"text":text,"metadata":metadata})
        self.assertEqual(self.run_worker()["states"],{"accepted":1})
        self.assertEqual((self.generator.calls,self.reviewer.calls),(1,1))

    def test_unknown_interrupted_review_preserves_generation_and_stays_excluded(self):
        request=self.f.request();aid=request["attempt_id"]
        text,metadata=self.generator.generate(self.f.gap,request)
        self.f.ledger.generation_done(self.f.gap,aid,text,metadata)
        self.f.ledger._append(aid,"review_started",{"at":"test"})
        self.assertEqual(self.run_worker()["states"],{"excluded":1})
        self.assertEqual(self.f.ledger._event(aid,"generation")["text"],text)
        self.assertEqual((self.generator.calls,self.reviewer.calls),(1,0))

    def test_transport_context_and_receipt_failures_are_terminal_not_retries(self):
        for exc in (TimeoutError(), ContextOverflow("test"), ValueError("malformed")):
            with self.subTest(exc=type(exc).__name__):
                db=sqlite3.connect(":memory:")
                ledger=RegenerationLedger(db,self.f.config,self.f.review_config)
                with patch.object(self.generator,"generate",side_effect=exc) as call:
                    result=run_regenerations(self.entries,ledger,self.generator,self.reviewer,max_gaps=1)
                    self.assertEqual(result["states"],{"excluded":1})
                    self.assertEqual(call.call_count,1)
                    run_regenerations(self.entries,ledger,self.generator,self.reviewer,max_gaps=1)
                    self.assertEqual(call.call_count,1)
                db.close()
        original=self.reviewer.review
        def mismatch(gap,candidate):
            raw,metadata=original(gap,candidate);metadata["usage"]["prompt_tokens"]=99
            return raw,metadata
        with patch.object(self.reviewer,"review",side_effect=mismatch):
            self.assertEqual(self.run_worker()["states"],{"excluded":1})
        self.assertEqual(list(accepted_selections(self.f.ledger)),[])

    def test_strict_review_must_be_actually_ready_before_any_generation(self):
        self.f.review_config["require_strict_thinking_server"]=True
        db=sqlite3.connect(":memory:");self.addCleanup(db.close)
        self.f.ledger=RegenerationLedger(db,self.f.config,self.f.review_config)
        with self.assertRaisesRegex(ValueError,"live readiness"):
            self.run_worker()
        self.assertEqual(self.generator.calls,0)
        self.assertEqual(self.run_worker(readiness=lambda config: True)["states"],{"excluded":1})
        self.assertEqual(self.generator.calls,0)

    def test_live_readiness_needs_fresh_complete_matching_actual_backend_arguments(self):
        config=self.f.review_config
        receipt={"schema":"cot.live-strict-backends/v1","config_hash":digest(config),
                 "checked_at_unix":time.time(),"routed_backend_ids":["n1","n2"],
                 "backends":[{"id":name,"server_args":{"enable_strict_thinking":True,
                    "grammar_backend":"xgrammar",
                    "served_model_name":config["model"],"context_length":4096}} for name in ("n1","n2")]}
        self.assertEqual(len(_strict_receipt(config,receipt)["server_args_hashes"]),2)
        for change in (lambda r:r.update(checked_at_unix=0),lambda r:r["backends"].pop(),
                       lambda r:r["backends"][0]["server_args"].update(enable_strict_thinking=False),
                       lambda r:r["backends"][0]["server_args"].update(grammar_backend="other")):
            bad=copy.deepcopy(receipt);change(bad)
            with self.assertRaises(ValueError):_strict_receipt(config,bad)

    def test_exact_config_and_implementation_resume_guard(self):
        self.run_worker()
        with patch("cot_filler.regeneration_worker.implementation_identity",return_value={"changed":True}):
            with self.assertRaisesRegex(ValueError,"identity changed"):self.run_worker()
        self.generator.config={**self.f.config,"temperature":0.9}
        with self.assertRaisesRegex(ValueError,"immutable regeneration"):self.run_worker()

    def test_cli_requires_explicit_activation_before_any_provider(self):
        with patch("cot_filler.regeneration_worker.PrefixRegenerationProvider") as provider, patch("sys.stderr",new_callable=io.StringIO):
            with self.assertRaises(SystemExit):
                main(["--original-journal","unused","--ledger","unused2","--inference-lock","unused3","--max-gaps","2"])
            provider.assert_not_called()

    def test_cli_cannot_start_while_actual_shared_inference_lock_is_owned(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp=Path(tmp);shared=tmp/"inference.lock";shared.touch()
            with shared.open("r+") as owned:
                fcntl.flock(owned,fcntl.LOCK_EX|fcntl.LOCK_NB)
                with patch("cot_filler.regeneration_worker.PrefixRegenerationProvider") as provider,patch("sys.stderr",new_callable=io.StringIO):
                    with self.assertRaises(SystemExit):main(["--original-journal",str(tmp/"original"),"--ledger",str(tmp/"ledger"),
                        "--inference-lock",str(shared),"--max-gaps","1","--confirm-run"])
                    provider.assert_not_called()
                self.assertFalse((tmp/"ledger").exists())

    def test_pagination_cursor_never_advances_past_interrupted_or_pending_work(self):
        queue=[{"ordinal":4},{"ordinal":9}]
        for result in ({"stopped":True,"selected":1,"states":{"accepted":1}},
                       {"stopped":False,"selected":2,"states":{"pending":1,"accepted":1}},
                       {"stopped":False,"selected":1,"states":{"accepted":1}}):
            final=_cohort_result(result,queue,0)
            self.assertTrue(final["resume_same_ledger_required"])
            self.assertIsNone(final["next_after_ordinal"])
        final=_cohort_result({"stopped":False,"selected":2,"states":{"accepted":2}},queue,0)
        self.assertEqual(final["next_after_ordinal"],9)

    def test_source_reader_is_read_only_and_checks_source_and_implementation(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp=Path(tmp);source=tmp/"source.jsonl";journal_path=tmp/"original.sqlite3"
            trace=json.loads((Path(__file__).parents[1]/"examples/demo.json").read_text())
            source.write_text(canonical(trace)+"\n")
            identity={"generator_config":self.f.config,"reviewer_config":self.f.review_config,"implementation_sha256":_code_identity()}
            journal=Journal(journal_path,source,file_sha256(source),identity)
            journal.close()
            before=file_sha256(journal_path)
            reader=ReadOnlyRejectedSource(journal_path)
            self.assertEqual(list(reader.rejected_rows(2)),[])
            row=reader.db.execute("SELECT * FROM gaps LIMIT 1").fetchone()
            reader.load_gap(row)
            with self.assertRaises(sqlite3.OperationalError):reader.db.execute("DELETE FROM gaps")
            source.write_text(source.read_text()+" ")
            with self.assertRaises(ValueError):reader.load_gap(row)
            reader.close()
            self.assertEqual(before,file_sha256(journal_path))

    def test_rejected_queue_pages_by_original_ordinal_without_repeating_first_batch(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp=Path(tmp);source=tmp/"source.jsonl";path=tmp/"original.sqlite3"
            trace=json.loads((Path(__file__).parents[1]/"examples/demo.json").read_text())
            other=copy.deepcopy(trace);other["trace_id"]+="-second"
            source.write_text(canonical(trace)+"\n"+canonical(other)+"\n")
            identity={"generator_config":self.f.config,"reviewer_config":self.f.review_config,"implementation_sha256":_code_identity()}
            journal=Journal(path,source,file_sha256(source),identity)
            rows=list(journal.db.execute("SELECT * FROM gaps ORDER BY ordinal"))
            for row in rows:
                with journal.db:
                    journal.db.execute("INSERT INTO candidates VALUES(?,?,?,?,?,?,?)",(row["id"],"test","{}","p","l","h","test"))
                    journal.db.execute("INSERT INTO reviews VALUES(?,?,?)",(row["id"],"{}","test"))
                    journal.db.execute("UPDATE gaps SET state='rejected' WHERE id=?",(row["id"],))
            journal.close()
            reader=ReadOnlyRejectedSource(path)
            try:
                first=list(reader.rejected_rows(1))
                second=list(reader.rejected_rows(1,first[0]["ordinal"]))
                self.assertEqual(len(first),1);self.assertEqual(len(second),1)
                self.assertGreater(second[0]["ordinal"],first[0]["ordinal"])
                self.assertNotEqual(second[0]["id"],first[0]["id"])
            finally:reader.close()

    def test_fresh_server_model_mismatch_is_excluded(self):
        original=self.reviewer.review
        def wrong_model(gap,candidate):
            raw,metadata=original(gap,candidate);metadata["response_model"]="different-server-model"
            return raw,metadata
        with patch.object(self.reviewer,"review",side_effect=wrong_model):
            self.assertEqual(self.run_worker()["states"],{"excluded":1})
        self.assertEqual(list(accepted_selections(self.f.ledger)),[])


if __name__ == "__main__":unittest.main()
