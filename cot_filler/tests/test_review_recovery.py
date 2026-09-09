import json
from pathlib import Path
import sqlite3
import tempfile
import unittest

from cot_filler.core import canonical
from cot_filler.corpus_source import file_sha256
from cot_filler.corpus_worker import Journal, run_corpus
from cot_filler.review_comparison import NoGeneration, import_candidates, snapshot_candidates
from cot_filler.review_recovery import revalidate, restrict_retry, saved_reviews
from cot_filler.tests.test_corpus_worker import DEMO, Reviewer
from cot_filler.tests.test_review_comparison import GenerationFixture


class ReviewRecoveryTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        root=Path(self.tmp.name);self.source=root/'source.jsonl'
        self.source.write_text(canonical(json.loads(DEMO.read_text()))+'\n')
        self.original=root/'original.sqlite3';self.output=root/'output.sqlite3'
        self.identity={"test":"fixture-only", "reviewer_config":{"chat_template_kwargs":{"thinking":True}}}
        reviewer=Reviewer();reviewer.config={**reviewer.config,"chat_template_kwargs":{"thinking":True}}
        old=Journal(self.original,self.source,file_sha256(self.source),self.identity)
        run_corpus(old,GenerationFixture(),reviewer,max_gaps=3,workers=1);old.close()

    def journal(self):
        result=Journal(self.output,self.source,file_sha256(self.source),self.identity)
        self.addCleanup(result.close)
        import_candidates(result,snapshot_candidates(self.original,expected_count=3))
        return result

    def test_offline_revalidation_preserves_raw_fences_and_excludes_empty_results(self):
        reviews=saved_reviews(self.original);reviews[0]['text']='```json\n'+reviews[0]['text']+'\n```'
        reviews[1]['text']='';metadata=json.loads(reviews[1]['data']);metadata['finish_reason']='length';reviews[1]['data']=canonical(metadata)
        old_hash=file_sha256(self.original);journal=self.journal();result=revalidate(journal,reviews)
        self.assertEqual(result['states'],{'approved':2,'failed':1});self.assertEqual(result['inference_requests'],0)
        self.assertFalse(result['automatic_bulk_approval'])
        recorded=journal.db.execute("select text from responses where stage='review' order by id").fetchall()
        self.assertEqual([r[0] for r in recorded],[r['text'] for r in reviews])
        self.assertEqual(file_sha256(self.original),old_hash)

    def test_retry_subset_never_regenerates_or_reviews_unselected_gaps(self):
        journal=self.journal();restrict_retry(journal,[2])
        reviewer=Reviewer('reject');reviewer.config={**reviewer.config,"chat_template_kwargs":{"thinking":True}}
        result=run_corpus(journal,NoGeneration(),reviewer,max_gaps=1,workers=1)
        self.assertEqual(result['states'],{'recovery_not_selected':2,'rejected':1})
        self.assertEqual(reviewer.calls,1)
        self.assertEqual(journal.db.execute("select count(*) from responses where stage='generate'").fetchone()[0],3)

    def test_duplicate_or_missing_retry_ordinals_and_multiple_review_receipts_fail(self):
        journal=self.journal()
        for ordinals in ([],[2,2],[99]):
            with self.assertRaises(ValueError):restrict_retry(journal,ordinals)
        reviews=saved_reviews(self.original)
        with self.assertRaises(ValueError):revalidate(journal,reviews+[reviews[0]])


if __name__=='__main__':unittest.main()
