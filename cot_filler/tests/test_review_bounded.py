import contextlib
import io
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import patch

from cot_filler.core import canonical
from cot_filler.corpus_source import file_sha256
from cot_filler.corpus_worker import Journal, PrefixReviewProvider, run_corpus
from cot_filler.review_bounded import main
from cot_filler.tests.test_bounded_review_controls import configuration
from cot_filler.tests.test_corpus_worker import DEMO, Reviewer, Tokenizer
from cot_filler.tests.test_review_comparison import GenerationFixture


class BoundedComparisonTests(unittest.TestCase):
    def test_cpu_preflight_reuses_candidates_including_prior_review_overflow(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp);source = root / "source.jsonl"
            source.write_text(canonical(json.loads(DEMO.read_text())) + "\n")
            original = root / "original.sqlite3"
            j = Journal(original, source, file_sha256(source),
                        {"reviewer_config": {}, "generator_config": {"model": "model-under-test"}})
            run_corpus(j, GenerationFixture(), Reviewer(), max_gaps=3, workers=1)
            with j.db:j.db.execute("UPDATE gaps SET state='context_overflow' WHERE ordinal=2")
            j.close();old_hash = file_sha256(original)
            config = configuration();config_path = root / "review.json"
            config_path.write_text(canonical(config))
            output = root / "comparison.sqlite3"
            args = ["--original-journal", str(original), "--original-count", "3", "--journal", str(output),
                    "--review-config", str(config_path), "--ordinals", "1,2", "--inference-lock", str(root / "inference.lock"),
                    "--preflight-only"]
            with patch("cot_filler.review_bounded.PrefixReviewProvider", side_effect=lambda c: PrefixReviewProvider(c, Tokenizer())), \
                 patch("urllib.request.build_opener") as network, contextlib.redirect_stdout(io.StringIO()) as stdout:
                main(args)
            network.assert_not_called()
            report = json.loads(stdout.getvalue())
            self.assertEqual(report["fitting_requests"], 2)
            self.assertEqual(report["inference_requests"], 0)
            self.assertFalse(report["automatic_bulk_approval"])
            self.assertEqual(file_sha256(original), old_hash)
            with sqlite3.connect(output) as db:
                self.assertEqual(dict(db.execute("SELECT state,count(*) FROM gaps GROUP BY state")),
                                 {"review_pending": 2, "recovery_not_selected": 1})
                self.assertEqual(db.execute("SELECT count(*) FROM reviews").fetchone()[0], 0)
                self.assertEqual(db.execute("SELECT count(*) FROM responses WHERE stage='generate'").fetchone()[0], 3)


if __name__ == "__main__":unittest.main()
