import json
from pathlib import Path
import sqlite3
import tempfile
import unittest

from cot_filler import corpus_worker
from cot_filler.core import canonical
from cot_filler.corpus_source import file_sha256
from cot_filler.corpus_worker import Journal, _code_identity
from cot_filler.derived_index import DerivedIndexJournal
from cot_filler.tests.test_corpus_worker import DEMO


class DerivedIndexTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name);self.source = self.root / "source.jsonl"
        self.source.write_text(canonical(json.loads(DEMO.read_text())) + "\n")
        self.original = self.root / "index.sqlite3"
        index = Journal(self.original, self.source, file_sha256(self.source),
                        {"reviewer_config": {}, "implementation_sha256": _code_identity()})
        index.close()

    def derive(self):
        return DerivedIndexJournal(self.root / "new.sqlite3", self.source, file_sha256(self.source),
                {"reviewer_config": {}}, index_template=self.original,
                template_worker_path=corpus_worker.__file__, expected_sources=1, expected_gaps=3)

    def test_new_identity_preserves_index_and_checks_original_gap_loading(self):
        old_sha = file_sha256(self.original)
        j = self.derive();self.addCleanup(j.close)
        self.assertEqual(j.summary(), {"queued": 3})
        row = dict(j.db.execute("SELECT * FROM gaps ORDER BY ordinal LIMIT 1").fetchone())
        self.assertEqual(j.load_gap(row)["id"], row["id"])
        self.assertEqual(j.identity["index_parent"]["file_sha256"], old_sha)
        self.assertEqual(file_sha256(self.original), old_sha)
        self.assertEqual(j.db.execute("SELECT count(*) FROM responses").fetchone()[0], 0)

    def test_claimed_parent_or_corrupt_gap_id_cannot_derive(self):
        for update in ("UPDATE gaps SET invocation='already-claimed' WHERE ordinal=1",
                       "UPDATE gaps SET id='corrupt' WHERE ordinal=1"):
            with self.subTest(update=update):
                with sqlite3.connect(self.original) as db:
                    saved = db.execute("SELECT id FROM gaps WHERE ordinal=1").fetchone()[0]
                    db.execute(update)
                with self.assertRaises(ValueError):self.derive()
                for path in self.root.glob('new.sqlite3*'):path.unlink()
                with sqlite3.connect(self.original) as db:
                    db.execute("UPDATE gaps SET id=?,invocation=NULL WHERE ordinal=1", (saved,))


if __name__ == "__main__":unittest.main()
