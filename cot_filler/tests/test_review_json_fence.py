import copy
import json
import unittest

from cot_filler.grounding_review_v3 import REVIEW_VERSION, validate_review
from cot_filler.tests import test_grounding_review_ids as fixtures


class CompleteReviewJSONTests(unittest.TestCase):
    def setUp(self):
        fixture = fixtures.BoundReviewTests()
        fixture.setUp()
        self.gap, self.candidate = fixture.gap, fixture.candidate
        self.result = {**fixture.result, "schema": REVIEW_VERSION}
        self.provenance = fixture.provenance

    def validate(self, response):
        return validate_review(self.gap, self.candidate, response, reviewer=self.provenance)

    def test_only_outer_fence_differs_from_bare_result(self):
        raw = json.dumps(self.result)
        bare = self.validate(raw)
        for response in ('```json\n' + raw + '\n```', ' \n```\r\n' + raw + '\r\n```\n '):
            actual = self.validate(response)
            self.assertEqual(actual.pop('review_response_format'), 'whole_json_fence')
            expected = {k: v for k, v in bare.items() if k != 'review_response_format'}
            self.assertEqual(actual, expected)

    def test_prose_multiple_objects_and_partial_fences_are_not_extracted(self):
        raw = json.dumps(self.result)
        for response in ('Here is the result:\n```json\n' + raw + '\n```',
                         '```json\n' + raw + '\n```\nExtra prose',
                         '```json\n' + raw + '\n```\n```json\n' + raw + '\n```',
                         '```json\n' + raw, raw + raw):
            with self.subTest(response=response[:20]), self.assertRaises(ValueError):
                self.validate(response)

    def test_fence_does_not_repair_missing_fields_or_evidence(self):
        bad_values = []
        bad = copy.deepcopy(self.result); bad['checks'].pop(next(iter(bad['checks']))); bad_values.append(bad)
        bad = copy.deepcopy(self.result); bad['statements'][0]['evidence_ids'] = ['future-p0']; bad_values.append(bad)
        bad = copy.deepcopy(self.result); bad['statements'].pop(); bad_values.append(bad)
        for bad in bad_values:
            with self.assertRaises(ValueError):
                self.validate('```json\n' + json.dumps(bad) + '\n```')

    def test_truncated_receipt_cannot_pass_even_with_complete_fenced_json(self):
        self.provenance['finish_reason'] = 'length'
        with self.assertRaises(ValueError):
            self.validate('```json\n' + json.dumps(self.result) + '\n```')


if __name__ == '__main__':
    unittest.main()
