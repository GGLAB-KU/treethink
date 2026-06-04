import unittest

from treethink import TreeThinkOutputs


class TestTreeThinkOutputs(unittest.TestCase):
    def test_solution_to_outputs_populates_solution_and_outputs(self):
        result = TreeThinkOutputs()

        result.solution_to_outputs(["proof line 1", "proof line 2"])

        self.assertEqual(result.solution, "proof line 1")
        self.assertEqual(result.outputs, ["proof line 1", "proof line 2"])
        self.assertTrue(result.has_solution)

    def test_to_dict_and_from_dict_round_trip(self):
        result = TreeThinkOutputs(
            solution_text="final answer",
            outputs=["final answer"],
            graph_stats={"nodes": 3},
            checked_and_true=True,
            generation_meta={"problem_id": "demo"},
        )

        payload = result.to_dict()
        restored = TreeThinkOutputs.from_dict(payload)

        self.assertEqual(restored.solution, "final answer")
        self.assertEqual(restored.outputs, ["final answer"])
        self.assertEqual(restored.graph_stats, {"nodes": 3})
        self.assertTrue(restored.checked_and_true)
        self.assertEqual(restored.generation_meta, {"problem_id": "demo"})

    def test_summary_includes_basic_metrics(self):
        result = TreeThinkOutputs(solution_text="abc", outputs=["abc"])

        summary = result.summary()

        self.assertEqual(summary["has_solution"], True)
        self.assertEqual(summary["solution_length"], 3)
        self.assertEqual(summary["output_count"], 1)
        self.assertEqual(summary["checked_and_true"], False)


if __name__ == "__main__":
    unittest.main()
