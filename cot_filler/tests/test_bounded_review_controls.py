import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from cot_filler.core import canonical, gaps_for
from cot_filler.corpus_source import file_sha256
from cot_filler.corpus_worker import Journal, PrefixReviewProvider, review_policy, validate_review_controls
from cot_filler.grounding_review_v4 import CHECKS, REVIEW_VERSION, response_format, review_contract
from cot_filler.tests.test_corpus_worker import DEMO, Generator, Tokenizer


def configuration():
    return {"model": "model-under-test", "base_url": "http://127.0.0.1:9999/v1",
            "tokenizer_path": "unused", "context_limit": 30000,
            "max_output_tokens": 12288, "chat_template_matches_server": True,
            "chat_template_kwargs": {"thinking": True}, "review_version": REVIEW_VERSION,
            "custom_params": {"thinking_budget": 8192}, "response_format": response_format(),
            "require_strict_thinking_server": True}


class BoundedReviewControlsTests(unittest.TestCase):
    def test_rejects_unsupported_unverified_or_ineffective_controls(self):
        cases = []
        for budget in [0, -1, True, 1.5, 12288]:
            config = configuration();config["custom_params"]["thinking_budget"] = budget;cases.append(config)
        config = configuration();config["custom_params"]["arbitrary"] = 1;cases.append(config)
        config = configuration();config.pop("require_strict_thinking_server");cases.append(config)
        config = configuration();config["chat_template_kwargs"]["thinking"] = False;cases.append(config)
        config = configuration();config["response_format"]["json_schema"]["strict"] = False;cases.append(config)
        for config in cases:
            with self.assertRaises(ValueError):validate_review_controls(config)
        with self.assertRaises(ValueError):review_policy({"review_version": "invented"})

    def test_wire_and_receipt_include_exact_controls_without_future_input(self):
        config = configuration();provider = PrefixReviewProvider(config, Tokenizer())
        trace = json.loads(DEMO.read_text());trace["events"][2]["content"] += " FUTURE_SENTINEL"
        gap = next(gaps_for(trace))
        class Response:
            def __enter__(self):return self
            def __exit__(self, *args):pass
            def read(self, limit):
                return json.dumps({"choices": [{"message": {"content": "{}"}, "finish_reason": "stop"}],
                                   "usage": {"prompt_tokens": 100, "reasoning_tokens": 42}}).encode()
        with patch("urllib.request.build_opener") as opener:
            opener.return_value.open.return_value = Response()
            _, receipt = provider.review(gap, "I will inspect the function.")
            body = json.loads(opener.return_value.open.call_args.args[0].data)
        for key in ("custom_params", "response_format", "chat_template_kwargs"):
            self.assertEqual(body[key], config[key]);self.assertEqual(receipt["parameters"][key], config[key])
        self.assertNotIn("FUTURE_SENTINEL", canonical(body))
        self.assertNotIn("max_thinking_tokens", body)
        self.assertNotIn("tools", body)
        self.assertEqual(receipt["review_version"], REVIEW_VERSION)

    def test_journal_rejects_changed_or_missing_decoding_controls(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "source.jsonl";source.write_text(canonical(json.loads(DEMO.read_text())) + "\n")
            config = configuration()
            journal = Journal(Path(tmp) / "journal.sqlite3", source, file_sha256(source), {"reviewer_config": config})
            self.addCleanup(journal.close)
            gap = journal.load_gap(dict(journal.db.execute("SELECT * FROM gaps ORDER BY ordinal LIMIT 1").fetchone()))
            text, metadata = Generator().generate(gap);journal.generation_done(gap, text, metadata)
            contract, evidence = review_contract(gap, text)
            evidence_id = next(key for key, item in evidence.items() if "Inspect the function before editing." in item["quote"])
            result = {"schema": REVIEW_VERSION, "decision": "pass", "checks": dict.fromkeys(CHECKS, True),
                      "statements": [{"id": unit["id"], "kind": "plan", "assessment": "supported", "evidence_ids": [evidence_id]}
                                     for unit in contract["candidate_units"]], "note": "Test fixture; no semantic model run."}
            receipt = {"provider": "openai-compatible", "model": "test-double", "finish_reason": "stop",
                       "usage": {"prompt_tokens": 100}, "prompt_tokens_local": 100,
                       "parameters": {key: copy.deepcopy(config[key]) for key in ("chat_template_kwargs", "custom_params", "response_format")}}
            for key in ("custom_params", "response_format"):
                bad = copy.deepcopy(receipt);bad["parameters"].pop(key)
                with self.assertRaisesRegex(ValueError, "decoding controls"):
                    journal.review_done(gap, text, result, bad)
            self.assertEqual(journal.review_done(gap, text, result, receipt), "approved")
            saved = json.loads(journal.db.execute("SELECT data FROM reviews").fetchone()[0])
            self.assertTrue(saved["configured_decoding_controls_verified"])
            self.assertEqual(journal.identity["review_version"], REVIEW_VERSION)


if __name__ == "__main__":unittest.main()
