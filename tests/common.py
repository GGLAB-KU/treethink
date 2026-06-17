"""Common functionality for tests."""

import itertools
import random

from treethink import BaseEvaluator, BasePolicy, Node

# -------------
# POLICIES
# -------------


class SimpleChildPolicy(BasePolicy):
    def __init__(self, num_child: int = 5, *args, **kwargs):
        super().__init__(name="simple_policy", *args, **kwargs)
        self.num_child = num_child

        # Create generators that maintain their own state
        self._id_gen = itertools.count(0)  # Infinite counter

    def __call__(self, node, method):
        def _policy(x: Node, y):
            for i in range(self.num_child - 1):
                child_id = next(self._id_gen)
                node.add_child(
                    Node(
                        text=f"ID{child_id}-child{i + 2}",
                        termination_str=node.termination_str,
                        max_children=node.max_children,
                        parent=node,
                    )
                )

        return _policy(node, method)


class SetStrPolicy(BasePolicy):
    def __init__(
        self, text: str, num_child: int = 5, sep="\n", *args, **kwargs
    ):
        super().__init__(name="set_str_policy", *args, **kwargs)
        self.num_child = num_child

        # Create generators that maintain their own state
        self._text_gen = iter([t + sep for t in text.split(sep=sep)])
        # self._text_gen = iter(text.split(sep=sep))
        self._id_gen = itertools.count(0)  # Infinite counter

    def __call__(self, node, method):
        # Get next values from generators
        try:
            first_child_text = next(self._text_gen)
            node.add_child(
                Node(
                    text=first_child_text,
                    termination_str=node.termination_str,
                    max_children=node.max_children,
                    parent=node,
                )
            )
        except StopIteration:
            pass

        # Add remaining children with generated IDs
        # if text is exhausted we will be giving one less children but it is OK.
        for i in range(self.num_child - 1):
            child_id = next(self._id_gen)
            node.add_child(
                Node(
                    text=f"ID{child_id}-child{i + 2}",
                    termination_str=node.termination_str,
                    max_children=node.max_children,
                    parent=node,
                )
            )


class PreferTerminationChildPolicy(BasePolicy):
    def __init__(
        self,
        termination_str: str = "```",
        start_prob: float = 0.0,
        increment_prob: float = 0.2,
        num_child: int = 5,
        *args,
        **kwargs,
    ):
        """Occasionally produce a node with text `termination_str` to test if
        we can REPL that proof trajectory."""
        super().__init__(name="prefer_termination_policy", *args, **kwargs)
        self.termination_str = termination_str
        self.num_child = num_child
        self.start_prob = start_prob
        self.increment_prob = increment_prob

        # Initialize generators
        self._prob_gen = self._create_prob_generator()
        self._id_gen = itertools.count(0)

    def _create_prob_generator(self):
        """Generator that yields increasing probabilities, cycling when > 1.0"""
        while True:
            prob = self.start_prob
            while prob <= 1.0:
                yield prob
                prob += self.increment_prob

    def _should_terminate(self) -> bool:
        """Determine if this call should produce a termination node."""
        prob = next(self._prob_gen)
        return random.random() < prob

    def __call__(self, node, method):
        # Create regular children (all but the last one)
        for i in range(self.num_child - 1):
            child_id = next(self._id_gen)
            node.add_child(
                Node(
                    f"ID{child_id}-child{i + 1}",
                    termination_str=self.termination_str,
                    max_children=node.max_children,
                    parent=node,
                )
            )

        # Last child: either termination or regular
        if self._should_terminate():
            node.add_child(
                Node(
                    text=self.termination_str,
                    termination_str=self.termination_str,
                    max_children=node.max_children,
                    parent=node,
                )
            )
        else:
            child_id = next(self._id_gen)
            node.add_child(
                Node(
                    f"ID{child_id}-child{self.num_child}",
                    termination_str=self.termination_str,
                    max_children=node.max_children,
                    parent=node,
                )
            )


# ---------------
# Node Evaluators
# ---------------


class RandomNodeEvaluator(BaseEvaluator):
    def __init__(self, min_val=-20, max_val=0, *args, **kwargs):
        self.min_val = min_val
        self.max_val = max_val

        super().__init__(name="random_evaluator", *args, **kwargs)

    def __call__(self, node, method):
        def _evaluator(x, y):
            if isinstance(x, Node):
                x = [x]

            return [
                random.randint(self.min_val, self.max_val)
                for i in range(len(x))
            ]

        return _evaluator(node, method)


class FirstPosOthersNegEvaluator(BaseEvaluator):
    """Score nodes in decreasing order from 0 to max_children, setting the first
    element to be the highest among all other children by making it positive."""

    def __init__(
        self,
        *args,
        **kwargs,
    ):
        self._val_gen = itertools.count(-1.0, -1.0)

        super().__init__(name="first_pos_others_neg_evaluator", *args, **kwargs)

    def __call__(self, node, method):
        def _evaluator(x, y):
            if isinstance(x, Node):
                x = [x]

            scores = [-next(self._val_gen) * 10000]
            scores.extend([next(self._val_gen) for i in range(len(x) - 1)])
            if len(scores) != len(x):
                breakpoint()
            return scores

        return _evaluator(node, method)
