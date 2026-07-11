"""Tests for the RMaxTS intrinsic-reward evaluator (sync + async).

Covers the binary novelty signal (``R_intrinsic = 1[new node]``), the
per-tree reset lifecycle (manual + auto on new root), pluggable novelty
keys, and integration with an MCTS search.
"""

import asyncio
import unittest

from tests.common import SimpleChildPolicy
from treethink.async_evaluators import AsyncRMaxTSEvaluator
from treethink.evaluators import RMaxTSEvaluator
from treethink.methods import RFMCTS, Node


class _FakeMethod:
    """Minimal method exposing what RMaxTSEvaluator needs."""

    def __init__(self, root):
        self.root_node = root

    def traverse_to_root(self, node, include_root=True):
        parts = []
        cur = node
        while cur is not None:
            parts.append(cur.text or "")
            cur = cur.parent
        return "".join(reversed(parts))


def _by_text(node, method):
    """State key = node's own text (lets us force collisions in tests)."""
    return node.text


class TestRMaxTSEvaluator(unittest.TestCase):
    def test_novel_then_seen(self):
        root = Node("root")
        method = _FakeMethod(root)
        ev = RMaxTSEvaluator(state_fn=_by_text)

        a, b = Node("A"), Node("B")
        self.assertEqual(ev([a, b], method), [1.0, 1.0])
        # Same states again → already seen → 0.0
        self.assertEqual(ev([a, b], method), [0.0, 0.0])
        # A node with the SAME text collides with a previously-seen state
        self.assertEqual(ev([Node("A"), Node("C")], method), [0.0, 1.0])

    def test_default_state_fn_uses_full_proof(self):
        root = Node("root")
        method = _FakeMethod(root)
        ev = RMaxTSEvaluator()  # default: traverse_to_root

        child = Node("step", parent=root)
        # Two distinct paths → both novel
        other = Node("other", parent=root)
        self.assertEqual(ev([child, other], method), [1.0, 1.0])
        # Re-evaluating the same node (same full proof) → seen
        self.assertEqual(ev([child], method), [0.0])

    def test_custom_rewards(self):
        method = _FakeMethod(Node("root"))
        ev = RMaxTSEvaluator(
            state_fn=_by_text, novel_reward=5.0, seen_reward=-1.0
        )
        self.assertEqual(ev([Node("X")], method), [5.0])
        self.assertEqual(ev([Node("X")], method), [-1.0])

    def test_single_node_input(self):
        method = _FakeMethod(Node("root"))
        ev = RMaxTSEvaluator(state_fn=_by_text)
        self.assertEqual(ev(Node("solo"), method), [1.0])

    def test_manual_reset(self):
        method = _FakeMethod(Node("root"))
        ev = RMaxTSEvaluator(state_fn=_by_text)
        self.assertEqual(ev([Node("A")], method), [1.0])
        self.assertEqual(ev([Node("A")], method), [0.0])
        ev.reset()
        self.assertEqual(ev([Node("A")], method), [1.0])

    def test_auto_reset_on_new_root(self):
        ev = RMaxTSEvaluator(state_fn=_by_text)
        m1 = _FakeMethod(Node("root1"))
        self.assertEqual(ev([Node("A")], m1), [1.0])
        self.assertEqual(ev([Node("A")], m1), [0.0])
        # New search tree (different root) → seen-set auto-clears
        m2 = _FakeMethod(Node("root2"))
        self.assertEqual(ev([Node("A")], m2), [1.0])

    def test_integration_with_mcts(self):
        root = Node("root")
        mcts = RFMCTS(
            root_node=None,
            policy=SimpleChildPolicy(num_child=3),
            evaluator=RMaxTSEvaluator(),
            exploration_weight=0.0,
        )
        mcts.set_root_node(root)
        mcts.simulate(expansion_count=3)
        self.assertGreater(len(root.children), 0)
        self.assertIsNotNone(mcts.best_answer)
        # Each uniquely-texted child is a novel state → gets the novel
        # reward (1.0); backprop only adds further non-negative intrinsic
        # rewards from its subtree, so win_value is always >= 1.0.
        for child in root.children:
            self.assertGreaterEqual(child.win_value, 1.0)


class TestAsyncRMaxTSEvaluator(unittest.TestCase):
    def test_async_novel_then_seen(self):
        method = _FakeMethod(Node("root"))
        ev = AsyncRMaxTSEvaluator(state_fn=_by_text)

        async def run():
            first = await ev([Node("A"), Node("B")], method)
            second = await ev([Node("A")], method)
            return first, second

        first, second = asyncio.run(run())
        self.assertEqual(first, [1.0, 1.0])
        self.assertEqual(second, [0.0])

    def test_async_reset(self):
        method = _FakeMethod(Node("root"))
        ev = AsyncRMaxTSEvaluator(state_fn=_by_text)

        async def run():
            a = await ev([Node("A")], method)
            ev.reset()
            b = await ev([Node("A")], method)
            return a, b

        a, b = asyncio.run(run())
        self.assertEqual(a, [1.0])
        self.assertEqual(b, [1.0])


if __name__ == "__main__":
    unittest.main()
