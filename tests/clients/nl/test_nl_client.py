"""Tests for the natural-language client (answer extraction + GT checking).

Pure-Python — no servers or models required.
"""

import asyncio
import json
import os
import tempfile
import unittest

from treethink.clients.nl.client import (
    AsyncNLClient,
    NLClient,
    default_compare,
    default_extract_answer,
)


class TestDefaultExtraction(unittest.TestCase):
    def test_boxed(self):
        self.assertEqual(
            default_extract_answer(r"reasoning ... so \boxed{42}"), "42"
        )

    def test_last_boxed_wins(self):
        self.assertEqual(
            default_extract_answer(r"\boxed{1} then \boxed{7}"), "7"
        )

    def test_answer_phrase(self):
        self.assertEqual(default_extract_answer("The answer is 3.14"), "3.14")

    def test_number_fallback(self):
        self.assertEqual(default_extract_answer("we get 5 apples"), "5")

    def test_none(self):
        self.assertIsNone(default_extract_answer("no number here"))
        self.assertIsNone(default_extract_answer(""))


class TestDefaultCompare(unittest.TestCase):
    """default_compare == math grade_answer (normalisation + sympy)."""

    def test_normalization(self):
        self.assertTrue(default_compare("42", " 42 "))
        self.assertTrue(default_compare("$1,000$", "1000"))

    def test_fraction_decimal_equivalence(self):
        self.assertTrue(default_compare("1/2", "0.5"))

    def test_symbolic_equivalence(self):
        self.assertTrue(default_compare("2x", "x+x"))

    def test_not_equal(self):
        self.assertFalse(default_compare("42", "43"))

    def test_none_prediction(self):
        self.assertFalse(default_compare(None, "42"))


class TestNLClientGroundTruthOverride(unittest.TestCase):
    def test_correct(self):
        c = NLClient()
        c.set_ground_truth("42")
        resp = c.check(snips=[r"long reasoning ... \boxed{42}"]).results[0]
        self.assertTrue(c.is_success_response(resp.response))
        self.assertEqual(resp.response["extracted"], "42")
        self.assertEqual(resp.response["ground_truth"], "42")
        self.assertEqual(resp.response["backend"], "nl")

    def test_incorrect(self):
        c = NLClient()
        c.set_ground_truth("42")
        resp = c.check(snips=[r"... \boxed{7}"]).results[0]
        self.assertFalse(c.is_success_response(resp.response))

    def test_no_ground_truth_is_failure(self):
        c = NLClient()  # no dataset, no override
        resp = c.check(snips=[r"\boxed{42}"]).results[0]
        self.assertFalse(c.is_success_response(resp.response))
        self.assertIsNone(resp.response["ground_truth"])


class TestNLClientDatasetLookup(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.NamedTemporaryFile(
            mode="w", suffix=".jsonl", delete=False
        )
        rows = [
            {"problem": "What is 2+2?", "answer": "4"},
            {"problem": "What is 2+2? Extended version.", "answer": "5"},
            {"problem": "Capital of France?", "answer": "Paris"},
        ]
        for r in rows:
            self.tmp.write(json.dumps(r) + "\n")
        self.tmp.close()

    def tearDown(self):
        os.unlink(self.tmp.name)

    def test_resolves_by_problem_prefix(self):
        c = NLClient(dataset_path=self.tmp.name)
        snip = r"What is 2+2? Let me think... \boxed{4}"
        resp = c.check(snips=[snip]).results[0].response
        self.assertEqual(resp["ground_truth"], "4")
        self.assertTrue(c.is_success_response(resp))

    def test_longest_prefix_wins(self):
        c = NLClient(dataset_path=self.tmp.name)
        # This snippet matches the longer problem statement → GT "5".
        snip = r"What is 2+2? Extended version. Answer: \boxed{5}"
        resp = c.check(snips=[snip]).results[0].response
        self.assertEqual(resp["ground_truth"], "5")

    def test_unknown_problem_no_gt(self):
        c = NLClient(dataset_path=self.tmp.name)
        resp = c.check(snips=[r"Unrelated problem \boxed{9}"]).results[0]
        self.assertIsNone(resp.response["ground_truth"])
        self.assertFalse(c.is_success_response(resp.response))

    def test_custom_keys(self):
        tmp = tempfile.NamedTemporaryFile(
            mode="w", suffix=".jsonl", delete=False
        )
        tmp.write(json.dumps({"q": "Q1?", "gt": "yes"}) + "\n")
        tmp.close()
        try:
            c = NLClient(
                dataset_path=tmp.name, problem_key="q", answer_key="gt"
            )
            resp = c.check(snips=[r"Q1? ... \boxed{yes}"]).results[0].response
            self.assertEqual(resp["ground_truth"], "yes")
        finally:
            os.unlink(tmp.name)


class TestPluggableFns(unittest.TestCase):
    def test_custom_extract_and_compare(self):
        c = NLClient(
            extract_fn=lambda t: t.split()[-1],
            compare_fn=lambda p, g: p == g,
        )
        c.set_ground_truth("DONE")
        resp = c.check(snips=["step one step DONE"]).results[0].response
        self.assertEqual(resp["extracted"], "DONE")
        self.assertTrue(c.is_success_response(resp))


class TestBatchAndOrder(unittest.TestCase):
    def test_batch_preserves_order(self):
        c = NLClient()
        c.set_ground_truth("42")
        resp = c.check(snips=[r"\boxed{42}", r"\boxed{0}", r"\boxed{42}"])
        outcomes = [c.is_success_response(r.response) for r in resp.results]
        self.assertEqual(outcomes, [True, False, True])

    def test_empty(self):
        self.assertEqual(NLClient().check(snips=[]).results, [])


class TestAsyncNLClient(unittest.TestCase):
    def test_async_check(self):
        c = AsyncNLClient()
        c.set_ground_truth("42")

        async def run():
            return await c.check(snips=[r"... \boxed{42}", r"... \boxed{1}"])

        resp = asyncio.run(run())
        outcomes = [c.is_success_response(r.response) for r in resp.results]
        self.assertEqual(outcomes, [True, False])


class TestFactoryWiring(unittest.TestCase):
    def test_create_client_routes_nl(self):
        from treethink.client_factory import create_client
        from treethink.utils.args import ClientArgs
        from treethink.utils.enums import ProofLanguage

        client = create_client(
            ProofLanguage.NL, ClientArgs(enable_cache=False)
        )
        self.assertIsInstance(client, NLClient)


if __name__ == "__main__":
    unittest.main()
