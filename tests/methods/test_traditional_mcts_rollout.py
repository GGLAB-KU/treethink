"""Tests for the TraditionalMCTS rollout path and Node.answer wiring.

These exercise the branch where the policy supports raw rollout generation
(``_has_rollout_policy()`` is True) and a ``rollout_evaluator`` scores the
*full* candidate proof (proof-so-far + rollout) — unlike the fallback path
covered in ``test_traditional_mcts.py``.
"""

import asyncio
import unittest

from treethink.methods import AsyncTraditionalMCTS, Node, TraditionalMCTS


# ---------------------------------------------------------------------------
# Fakes that satisfy ``_has_rollout_policy()`` without needing vLLM.
# ---------------------------------------------------------------------------


class _FakeCompletion:
    def __init__(self, text):
        self.text = text


class _FakeResponse:
    def __init__(self, texts):
        self.outputs = [_FakeCompletion(t) for t in texts]


class _FakeModel:
    def __init__(self, completions):
        self._completions = completions

    def generate(self, prompts, sampling_params, use_tqdm=False):
        return [_FakeResponse(self._completions) for _ in prompts]


class _FakeTokenizer:
    def encode(self, text):
        return text.split()


class RolloutStubPolicy:
    """Minimal policy exposing the attributes ``_has_rollout_policy`` needs."""

    def __init__(self, completions, n_children=2):
        self.model = _FakeModel(completions)
        self.prompter = lambda messages: " ".join(
            m["content"] for m in messages
        )
        self.system_prompt = "sys"
        self._tokenizer = _FakeTokenizer()
        self._max_model_len = 4096
        self._n_children = n_children

    def __call__(self, node, method):
        for i in range(self._n_children):
            node.add_child(Node(text=f"step-{i}"))


def _qed_rollout_evaluator(nodes, method):
    """Score the *full* candidate proof: 1.0 when it contains 'QED'."""
    return [1.0 if "QED" in n.answer else 0.0 for n in nodes]


class AsyncRolloutStubPolicy(RolloutStubPolicy):
    async def __call__(self, node, method):
        for i in range(self._n_children):
            node.add_child(Node(text=f"step-{i}"))


# ---------------------------------------------------------------------------
# Node.answer
# ---------------------------------------------------------------------------


class TestNodeAnswer(unittest.TestCase):
    def test_answer_concatenates_path(self):
        root = Node("a")
        child = Node("b")
        grandchild = Node("c")
        root.add_child(child)
        child.add_child(grandchild)
        self.assertEqual(grandchild.answer, "abc")

    def test_answer_appends_rollout_output(self):
        root = Node("a")
        child = Node("b")
        root.add_child(child)
        child.rollout_output = "cd"
        self.assertEqual(child.answer, "abcd")

    def test_answer_handles_none_text(self):
        root = Node(None)
        child = Node("b")
        root.add_child(child)
        self.assertEqual(child.answer, "b")


# ---------------------------------------------------------------------------
# Rollout path
# ---------------------------------------------------------------------------


class TestTraditionalMCTSRollout(unittest.TestCase):
    def _make(self, method_cls=TraditionalMCTS, policy_cls=RolloutStubPolicy):
        root = Node("root\n")
        mcts = method_cls(
            root_node=None,
            policy=policy_cls(completions=["proof QED", "proof bad"]),
            evaluator=lambda nodes, method: [0.0 for _ in nodes],
            rollout_evaluator=_qed_rollout_evaluator,
            exploration_weight=0.0,
        )
        mcts.set_root_node(root)
        return root, mcts

    def test_rollout_path_scores_full_proof(self):
        root, mcts = self._make()
        mcts.simulate(expansion_count=1)

        self.assertGreater(len(root.children), 0)
        for child in root.children:
            # mean of [1.0 (QED), 0.0 (bad)] == 0.5 → proves rollout ran
            self.assertEqual(child.win_value, 0.5)
            # best completion retained
            self.assertEqual(child.rollout_output, "proof QED")
            # full proof = root + step + winning rollout
            self.assertTrue(child.answer.endswith("proof QED"))
            self.assertTrue(child.answer.startswith("root\n"))

    def test_temp_nodes_do_not_pollute_tree(self):
        root, mcts = self._make()
        mcts.simulate(expansion_count=1)
        for child in root.children:
            self.assertEqual(len(child.children), 0)

    def test_fallback_when_no_rollout_evaluator(self):
        """No rollout_evaluator → AlphaZero-style direct child evaluation."""
        root = Node("root\n")
        mcts = TraditionalMCTS(
            root_node=None,
            policy=RolloutStubPolicy(completions=["proof QED"]),
            evaluator=lambda nodes, method: [7.0 for _ in nodes],
            rollout_evaluator=None,
            exploration_weight=0.0,
        )
        mcts.set_root_node(root)
        mcts.simulate(expansion_count=1)
        for child in root.children:
            self.assertEqual(child.win_value, 7.0)
            self.assertIsNone(child.rollout_output)

    def test_async_rollout_path(self):
        root, mcts = self._make(
            method_cls=AsyncTraditionalMCTS,
            policy_cls=AsyncRolloutStubPolicy,
        )
        asyncio.run(mcts.async_simulate(expansion_count=1))

        self.assertGreater(len(root.children), 0)
        for child in root.children:
            self.assertEqual(child.win_value, 0.5)
            self.assertEqual(child.rollout_output, "proof QED")


if __name__ == "__main__":
    unittest.main()
