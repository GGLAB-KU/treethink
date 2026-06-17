"""Tests for the reward-model value-function evaluators (sync + async).

A fake reward model (exposing ``encode``) is injected so the evaluators can
be exercised without vLLM / a real pooling model.  Tests assert the exact
text each evaluator feeds the model (proof-level = whole proof; state-level =
problem + node step), plus score extraction/reduction, batching, and order.
"""

import asyncio
import unittest
from types import SimpleNamespace

from treethink.async_evaluators import (
    AsyncProofLevelRewardEvaluator,
    AsyncStateLevelRewardEvaluator,
)
from treethink.evaluators import (
    ProofLevelRewardEvaluator,
    StateLevelRewardEvaluator,
)
from treethink.methods import Node


def _pooled(data):
    """Mimic a vLLM pooling output: ``output.outputs.data``."""
    return SimpleNamespace(outputs=SimpleNamespace(data=data))


class _Tensorish:
    """Stands in for a torch tensor (has ``tolist``)."""

    def __init__(self, values):
        self._values = values

    def tolist(self):
        return self._values


class FakeRewardModel:
    """Fake pooling model: scores each prompt via an injected ``scorer``."""

    def __init__(self, scorer):
        self.scorer = scorer
        self.calls = []  # list of (prompts, pooling_task)

    def encode(self, prompts, pooling_task="classify"):
        self.calls.append((list(prompts), pooling_task))
        return [_pooled(self.scorer(p)) for p in prompts]


# Join chat messages into one string so we can assert on the fed text.
def _prompter(messages):
    return "|".join(m["content"] for m in messages)


class _FakeMethod:
    def __init__(self, root):
        self.root_node = root

    def traverse_to_root(self, node, include_root=True):
        parts = []
        cur = node
        while cur is not None:
            parts.append(cur.text or "")
            cur = cur.parent
        parts.reverse()
        return "".join(parts if include_root else parts[1:])


def _tree():
    root = Node("THM ")
    a = Node("s1 ", parent=root)
    root.add_child(a)
    b = Node("s2 ", parent=a)
    a.add_child(b)
    return root, b


class TestProofLevelRewardEvaluator(unittest.TestCase):
    def test_feeds_whole_proof(self):
        root, b = _tree()
        model = FakeRewardModel(scorer=lambda p: 1.0)
        ev = ProofLevelRewardEvaluator(model, prompter=_prompter)
        ev([b], _FakeMethod(root))
        # user = problem, assistant = full proof (root..b)
        assert model.calls[-1][0][0] == "THM |THM s1 s2 "
        assert model.calls[-1][1] == "classify"

    def test_score_returned(self):
        root, b = _tree()
        model = FakeRewardModel(scorer=lambda p: 0.42 if "s1" in p else 0.0)
        ev = ProofLevelRewardEvaluator(model, prompter=_prompter)
        assert ev([b], _FakeMethod(root)) == [0.42]


class TestStateLevelRewardEvaluator(unittest.TestCase):
    def test_feeds_problem_plus_node_only(self):
        root, b = _tree()
        model = FakeRewardModel(scorer=lambda p: 1.0)
        ev = StateLevelRewardEvaluator(model, prompter=_prompter)
        ev([b], _FakeMethod(root))
        # user = problem, assistant = node step only (no earlier steps)
        assert model.calls[-1][0][0] == "THM |s2 "
        assert "s1" not in model.calls[-1][0][0]

    def test_system_prompt_included(self):
        root, b = _tree()
        model = FakeRewardModel(scorer=lambda p: 1.0)
        ev = StateLevelRewardEvaluator(
            model, prompter=_prompter, system_prompt="SYS"
        )
        ev([b], _FakeMethod(root))
        assert model.calls[-1][0][0] == "SYS|THM |s2 "


class TestScoreExtraction(unittest.TestCase):
    def _eval(self, data, reduction="last"):
        root, b = _tree()
        model = FakeRewardModel(scorer=lambda p: data)
        ev = StateLevelRewardEvaluator(
            model, prompter=_prompter, score_reduction=reduction
        )
        return ev([b], _FakeMethod(root))[0]

    def test_scalar(self):
        assert self._eval(0.7) == 0.7

    def test_vector_last_default(self):
        assert self._eval([0.1, 0.9]) == 0.9

    def test_vector_mean(self):
        assert abs(self._eval([0.2, 0.4], reduction="mean") - 0.3) < 1e-9

    def test_vector_first(self):
        assert self._eval([0.1, 0.9], reduction="first") == 0.1

    def test_tensor_like(self):
        assert self._eval(_Tensorish([5.0]), reduction="last") == 5.0


class TestBatchingAndInput(unittest.TestCase):
    def test_batch_preserves_order(self):
        root, b = _tree()
        c = Node("s3 ", parent=root)
        root.add_child(c)
        model = FakeRewardModel(scorer=lambda p: 9.0 if "s2" in p else 1.0)
        ev = StateLevelRewardEvaluator(model, prompter=_prompter)
        assert ev([b, c], _FakeMethod(root)) == [9.0, 1.0]

    def test_single_node_input(self):
        root, b = _tree()
        model = FakeRewardModel(scorer=lambda p: 3.0)
        ev = StateLevelRewardEvaluator(model, prompter=_prompter)
        assert ev(b, _FakeMethod(root)) == [3.0]


class TestAsyncRewardEvaluators(unittest.TestCase):
    def test_async_proof_level(self):
        root, b = _tree()
        model = FakeRewardModel(scorer=lambda p: 1.0 if "s1" in p else 0.0)
        ev = AsyncProofLevelRewardEvaluator(model, prompter=_prompter)
        result = asyncio.run(ev([b], _FakeMethod(root)))
        assert result == [1.0]
        assert model.calls[-1][0][0] == "THM |THM s1 s2 "

    def test_async_state_level(self):
        root, b = _tree()
        model = FakeRewardModel(scorer=lambda p: 1.0 if "s1" in p else 0.0)
        ev = AsyncStateLevelRewardEvaluator(model, prompter=_prompter)
        result = asyncio.run(ev([b], _FakeMethod(root)))
        # state-level excludes earlier steps -> scorer sees no "s1"
        assert result == [0.0]
        assert model.calls[-1][0][0] == "THM |s2 "


if __name__ == "__main__":
    unittest.main()
