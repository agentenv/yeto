from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

from cot_filler.core import PROMPT_VERSION, canonical, digest, gaps_for
from cot_filler.corpus_source import file_sha256, normalize_archive, prepare_source
from cot_filler.corpus_worker import AdmissionOverflow, Journal, PrefixReviewProvider, SourceChanged, TokenAdmission, gap_at, run_corpus
from cot_filler.grounding_review_v3 import CHECKS, REVIEW_VERSION, review_contract


DEMO = Path(__file__).resolve().parents[1] / "examples" / "demo.json"


class Tokenizer:
    identity = "fake-offset-free-tokenizer-for-unit-tests"
    def count(self, messages, options):
        return 100


class Generator:
    """Deterministic unit-test double; no real provider or evaluator runs."""
    tokenizer = Tokenizer()
    config = {"context_limit": 1000, "max_output_tokens": 100, "safety_margin": 0}
    def __init__(self):
        self.calls = 0
    def generate(self, gap):
        self.calls += 1
        return "Inspect the function before choosing a repair.", {
            "generator": {"provider": "unit-test-double", "model": "test-generation", "prompt_version": PROMPT_VERSION},
            "prompt_hash": gap["prompt_hash"], "finish_reason": "stop", "flags": [],
        }


class Reviewer(Generator):
    def __init__(self, decision="pass"):
        super().__init__()
        self.decision = decision
    def review(self, gap, candidate):
        self.calls += 1
        _, evidence = review_contract(gap, candidate)
        evidence_id = next(key for key, item in evidence.items() if item["event_id"] == "u1" and "Inspect the function before editing." in item["quote"])
        result = {"schema": REVIEW_VERSION, "decision": self.decision, "checks": dict.fromkeys(CHECKS, self.decision == "pass"),
                  "statements": [{"id": "c0", "kind": "plan", "assessment": "supported" if self.decision == "pass" else "uncertain",
                                  "evidence_ids": [evidence_id]}],
                  "note": "Unit-test evaluator fixture, not actual semantic quality evidence."}
        return json.dumps(result), {"provider": "openai-compatible", "model": "mocked-provider-only-in-unit-test", "finish_reason": "stop",
                                    "prompt_tokens_local": 100, "usage": {"prompt_tokens": 100},
                                    "parameters": {"chat_template_kwargs": copy.deepcopy(self.config.get("chat_template_kwargs", {}))}}


class CorpusWorkerTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.source = Path(self.tmp.name) / "source.jsonl"
        self.trace = json.loads(DEMO.read_text())
        self.source.write_text(canonical(self.trace) + "\n")
        self.path = Path(self.tmp.name) / "journal.sqlite3"

    def journal(self, identity=None):
        result = Journal(self.path, self.source, file_sha256(self.source), identity or {"test": "deterministic-fixture-only", "reviewer_config": {}})
        self.addCleanup(result.close)
        return result

    def test_compact_gap_reconstruction_exactly_matches_original_v4(self):
        for gap in gaps_for(self.trace):
            self.assertEqual(gap_at(self.trace, gap["event_index"]), gap)

    def test_finite_run_is_reviewed_and_resume_does_not_regenerate(self):
        journal = self.journal()
        generator, reviewer = Generator(), Reviewer()
        first = run_corpus(journal, generator, reviewer, max_gaps=2, workers=2, token_budget=400)
        self.assertEqual(first["states"], {"approved": 2, "queued": 1})
        self.assertEqual(generator.calls, 2)
        self.assertEqual(reviewer.calls, 2)
        self.assertLessEqual(first["peak_reserved_tokens"], 400)
        self.assertLessEqual(first["peak_requests"], 2)
        second = run_corpus(journal, generator, reviewer, max_gaps=3, workers=2, token_budget=400)
        self.assertEqual(second["states"], {"approved": 3})
        self.assertEqual(generator.calls, 3)
        self.assertEqual(second["selected_gaps"], 1)
        self.assertFalse(second["semantic_certainty_claimed"])
        self.assertEqual(journal.db.execute("SELECT COUNT(*) FROM responses").fetchone()[0], 6)

    def test_generation_only_stages_candidates_without_review_then_resumes_exactly(self):
        journal = self.journal()
        generator, reviewer = Generator(), Reviewer()
        first = run_corpus(journal, generator, reviewer, max_gaps=2, workers=2,
                           token_budget=400, generation_only=True)
        self.assertEqual(first["states"], {"review_pending": 2, "queued": 1})
        self.assertEqual(first["mode"], "generation_only")
        self.assertEqual(generator.calls, 2)
        self.assertEqual(reviewer.calls, 0)
        self.assertEqual(journal.db.execute("SELECT COUNT(*) FROM reviews").fetchone()[0], 0)
        second = run_corpus(journal, generator, reviewer, max_gaps=3, workers=2,
                            token_budget=400, generation_only=True)
        self.assertEqual(second["states"], {"review_pending": 3})
        self.assertEqual(second["selected_gaps"], 1)
        self.assertEqual(generator.calls, 3)
        self.assertEqual(reviewer.calls, 0)
        staged = [tuple(row) for row in journal.db.execute("SELECT gap_id,text FROM candidates ORDER BY gap_id")]
        result = run_corpus(journal, generator, reviewer, max_gaps=3, workers=2, token_budget=400)
        self.assertEqual(result["states"], {"approved": 3})
        self.assertEqual(generator.calls, 3)
        self.assertEqual(reviewer.calls, 3)
        self.assertEqual(staged, [tuple(row) for row in journal.db.execute("SELECT gap_id,text FROM candidates ORDER BY gap_id")])

    def test_generation_only_still_rejects_truncated_outputs(self):
        class Truncated(Generator):
            def generate(self, gap):
                text, metadata = super().generate(gap)
                return text, {**metadata, "finish_reason": "length", "flags": ["truncated_output"]}
        journal = self.journal()
        reviewer = Reviewer()
        result = run_corpus(journal, Truncated(), reviewer, max_gaps=1, generation_only=True)
        self.assertEqual(result["states"], {"generation_invalid": 1, "queued": 2})
        self.assertEqual(reviewer.calls, 0)

    def test_uncertain_review_is_excluded_and_exact_candidate_kept(self):
        journal = self.journal()
        result = run_corpus(journal, Generator(), Reviewer("uncertain"), max_gaps=1, workers=1)
        self.assertEqual(result["states"], {"queued": 2, "uncertain": 1})
        record = json.loads(journal.db.execute("SELECT data FROM reviews").fetchone()[0])
        self.assertEqual(record["decision"], "uncertain")
        self.assertFalse(record["automatic_approval"])

    def test_context_overflow_does_not_call_provider(self):
        journal = self.journal()
        generator = Generator()
        generator.config = {**generator.config, "context_limit": 150}
        result = run_corpus(journal, generator, Reviewer(), max_gaps=1)
        self.assertEqual(result["states"], {"context_overflow": 1, "queued": 2})
        self.assertEqual(generator.calls, 0)

    def test_truncated_generator_and_invalid_review_never_approve(self):
        class Truncated(Generator):
            def generate(self, gap):
                text, metadata = super().generate(gap)
                return text, {**metadata, "finish_reason": "length", "flags": ["truncated_output"]}
        journal = self.journal()
        reviewer = Reviewer()
        result = run_corpus(journal, Truncated(), reviewer, max_gaps=1)
        self.assertEqual(result["states"]["generation_invalid"], 1)
        self.assertEqual(reviewer.calls, 0)
        class Invalid(Reviewer):
            def review(self, gap, candidate):
                return "not valid review JSON", {"provider": "openai-compatible", "model": "test-double", "finish_reason": "stop"}
        result = run_corpus(journal, Generator(), Invalid(), max_gaps=1)
        self.assertEqual(result["states"]["failed"], 1)
        self.assertNotIn("approved", result["states"])
        self.assertEqual(journal.db.execute("SELECT text FROM responses WHERE stage='review'").fetchone()[0], "not valid review JSON")

    def test_provider_failures_trip_circuit_with_bounded_retries(self):
        class Unavailable(Generator):
            def generate(self, gap):
                self.calls += 1
                raise ValueError("Teacher returned HTTP 503")
        journal = self.journal()
        generator = Unavailable()
        result = run_corpus(journal, generator, Reviewer(), max_gaps=3, workers=1, retries=0, provider_failure_limit=1)
        self.assertTrue(result["stopped"])
        self.assertEqual(result["stop_reason"], "provider_unavailable")
        self.assertEqual(generator.calls, 1)
        self.assertEqual(result["states"], {"failed": 1, "queued": 2})

    def test_transient_request_retries_are_recorded(self):
        class Flaky(Generator):
            def generate(self, gap):
                self.calls += 1
                if self.calls == 1:
                    raise ValueError("Teacher returned HTTP 429")
                text, data = Generator.generate(self, gap)
                return text, data
        journal = self.journal()
        result = run_corpus(journal, Flaky(), Reviewer(), max_gaps=1, retries=1, workers=1)
        self.assertEqual(result["states"]["approved"], 1)
        metadata = json.loads(journal.db.execute("SELECT data FROM candidates").fetchone()[0])
        self.assertEqual(metadata["request_attempts"], 2)

    def test_source_mutation_is_detected_before_loading_or_commit(self):
        journal = self.journal()
        row = dict(journal.db.execute("SELECT * FROM gaps LIMIT 1").fetchone())
        gap = journal.load_gap(row)
        self.source.write_text("{}\n")
        with self.assertRaises(SourceChanged):
            journal.load_gap(row)
        with self.assertRaises(SourceChanged):
            journal.generation_done(gap, *Generator().generate(gap))

    def test_prompt_version_or_config_identity_change_refuses_resume(self):
        journal = self.journal()
        with self.assertRaisesRegex(ValueError, "identity changed"):
            Journal(self.path, self.source, file_sha256(self.source), {"test": "changed-configuration"})
        self.assertEqual(journal.summary(), {"queued": 3})

    def test_review_receipt_matches_configured_thinking_or_nonthinking_mode(self):
        for index, options in enumerate(({}, {"thinking": True}, {"thinking": False})):
            with self.subTest(options=options):
                config = {"chat_template_kwargs": copy.deepcopy(options)}
                journal = Journal(self.path.with_name(f"mode-{index}.sqlite3"), self.source,
                                  file_sha256(self.source), {"reviewer_config": config})
                self.addCleanup(journal.close)
                # Mutating the original configuration cannot change the seal.
                config["chat_template_kwargs"] = {"thinking": "changed-after-indexing"}
                reviewer = Reviewer()
                reviewer.config = {**reviewer.config, "chat_template_kwargs": options}
                result = run_corpus(journal, Generator(), reviewer, max_gaps=1, workers=1)
                self.assertEqual(result["states"].get("approved"), 1)
                record = json.loads(journal.db.execute("SELECT data FROM reviews").fetchone()[0])
                self.assertTrue(record["prompt_token_parity_verified"])
                self.assertTrue(record["configured_template_options_verified"])

    def test_review_receipt_malformed_counts_and_option_mismatches_never_approve(self):
        journal = self.journal({"reviewer_config": {"chat_template_kwargs": {"thinking": True}}})
        row = dict(journal.db.execute("SELECT * FROM gaps LIMIT 1").fetchone())
        gap = journal.load_gap(row)
        text, generation = Generator().generate(gap)
        journal.generation_done(gap, text, generation)
        reviewer = Reviewer()
        reviewer.config = {**reviewer.config, "chat_template_kwargs": {"thinking": True}}
        response, metadata = reviewer.review(gap, text)
        cases = []
        for local, remote in [(None, 100), (100, None), (True, True), (100.0, 100),
                              (100, 100.0), (0, 0), (-1, -1), (100, 101)]:
            value = copy.deepcopy(metadata)
            value.update(prompt_tokens_local=local, usage={"prompt_tokens": remote})
            cases.append(value)
        for options in (None, [], {}, {"thinking": False}, {"thinking": 1},
                        {"thinking": True, "unconfigured_option": True}):
            value = copy.deepcopy(metadata)
            value["parameters"] = {"chat_template_kwargs": options}
            cases.append(value)
        for field in ("usage", "parameters", "prompt_tokens_local"):
            value = copy.deepcopy(metadata)
            del value[field]
            cases.append(value)
        for value in cases:
            with self.subTest(metadata=value), self.assertRaises(ValueError):
                journal.review_done(gap, text, response, value)
        self.assertEqual(journal.db.execute("SELECT count(*) FROM reviews").fetchone()[0], 0)
        self.assertEqual(journal.summary(), {"review_pending": 1, "queued": 2})
        self.assertEqual(journal.review_done(gap, text, response, metadata), "approved")

    def test_missing_reviewer_identity_fails_closed_with_response_retained(self):
        journal = self.journal({"test": "missing-reviewer-contract"})
        result = run_corpus(journal, Generator(), Reviewer(), max_gaps=1, workers=1)
        self.assertEqual(result["states"].get("failed"), 1)
        self.assertNotIn("approved", result["states"])
        self.assertEqual(journal.db.execute("SELECT count(*) FROM reviews").fetchone()[0], 0)
        self.assertEqual(journal.db.execute("SELECT count(*) FROM responses WHERE stage='review'").fetchone()[0], 1)

    def test_recovery_retains_durable_candidate_without_second_generation(self):
        journal = self.journal()
        row, _, _ = journal.claim("old-process", True)
        gap = journal.load_gap(row)
        journal.generation_done(gap, *Generator().generate(gap))
        with journal.db:
            journal.db.execute("UPDATE gaps SET state='reviewing' WHERE id=?", (gap["id"],))
        reopened = self.journal()
        generator, reviewer = Generator(), Reviewer()
        result = run_corpus(reopened, generator, reviewer, max_gaps=1)
        self.assertEqual(generator.calls, 0)
        self.assertEqual(reviewer.calls, 1)
        self.assertEqual(result["states"], {"approved": 1, "queued": 2})

    def test_token_admission_obeys_weight_and_cancellation(self):
        admission = TokenAdmission(4, 250)
        stop = threading.Event()
        occupied = threading.Event()
        release = threading.Event()
        def hold():
            with admission.reserve(200, stop):
                occupied.set()
                release.wait(1)
        thread = threading.Thread(target=hold)
        thread.start()
        self.assertTrue(occupied.wait(1))
        self.assertEqual(admission.tokens, 200)
        with self.assertRaises(AdmissionOverflow):
            with admission.reserve(251, stop):
                pass
        release.set()
        thread.join(1)
        self.assertEqual(admission.tokens, 0)


class ReviewWireTests(unittest.TestCase):
    def test_review_wire_has_no_tools_target_or_lookahead(self):
        config = {"model": "model-under-test", "base_url": "http://127.0.0.1:9999/v1", "tokenizer_path": "unused",
                  "context_limit": 10000, "max_output_tokens": 500, "chat_template_matches_server": True}
        provider = PrefixReviewProvider(config, Tokenizer())
        gap = next(gaps_for(json.loads(DEMO.read_text())))
        class Response:
            def __enter__(self): return self
            def __exit__(self, *args): pass
            def read(self, limit):
                return json.dumps({"choices": [{"message": {"content": "{}"}, "finish_reason": "stop"}]}).encode()
        with patch("urllib.request.build_opener") as opener:
            opener.return_value.open.return_value = Response()
            provider.review(gap, "Inspect the function.")
            body = json.loads(opener.return_value.open.call_args.args[0].data)
        self.assertNotIn("tools", body)
        self.assertNotIn("tool_choice", body)
        self.assertNotIn("read-1", canonical(body["messages"]))


class CorpusSourceTests(unittest.TestCase):
    def archive(self):
        return [
            {"event_id": "u1", "role": "user", "kind": "message", "content": "Inspect total.py", "data": {}},
            {"event_id": "r1", "role": "assistant", "kind": "reasoning", "content": "hidden source text", "data": {}},
            {"event_id": "a1", "role": "assistant", "kind": "tool_call", "content": "", "data": {"type": "custom_tool_call", "name": "run", "input": "read total.py", "call_id": "call1"}},
            {"event_id": "t1", "role": "tool", "kind": "tool_result", "content": "result=1", "data": {"call_id": "call1"}},
        ]

    def test_preserves_tools_and_exact_original_reasoning_boundary(self):
        archive = self.archive()
        trace = normalize_archive(archive, {"capture_id": "capture"})
        self.assertEqual(trace["events"][1]["data"], archive[2]["data"])
        self.assertEqual(trace["gap_targets"][0]["event_id"], "a1")
        self.assertEqual(trace["gap_targets"][0]["source_reasoning_events"], [{"event_id": "r1", "archive_position": 1}])
        self.assertNotIn("hidden source text", canonical(trace))
        self.assertEqual(archive[1]["content"], "hidden source text")

    def test_ambiguous_marker_or_unknown_event_fails_closed(self):
        for field, value in (("role", "user"), ("kind", "compaction")):
            archive = self.archive()
            archive[2][field] = value
            with self.assertRaises(ValueError):
                normalize_archive(archive, {"capture_id": "capture"})

    def test_selected_manifest_hash_and_scores_are_enforced(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            archive = root / "archive.jsonl"
            archive.write_text("\n".join(map(canonical, self.archive())) + "\n")
            manifest = root / "manifest.jsonl"
            record = {"capture_id": "capture", "archive_path": str(archive), "archive_file_sha256": file_sha256(archive), "quality_score": 4, "overall_confidence_score": 3}
            manifest.write_text(canonical(record) + "\n")
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                prepare_source(manifest, root / "source.jsonl", root / "report.json", expected_manifest_sha256="wrong")
            report = prepare_source(manifest, root / "source.jsonl", root / "report.json", expected_manifest_sha256=file_sha256(manifest))
            self.assertEqual(report["retained_traces"], 1)
            self.assertEqual(report["eligible_gaps"], 1)
            self.assertEqual(report["source_file_sha256"], file_sha256(root / "source.jsonl"))
            with self.assertRaisesRegex(ValueError, "immutable"):
                prepare_source(manifest, root / "source.jsonl", root / "report.json", expected_manifest_sha256=file_sha256(manifest))


if __name__ == "__main__":
    unittest.main()
