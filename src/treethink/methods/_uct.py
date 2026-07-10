"""Shared UCT (Upper Confidence bounds applied to Trees) helper.

Used by both :class:`~treethink.methods.rf_mcts.RFMCTS`
and :class:`~treethink.methods.traditional_mcts.TraditionalMCTS`.
"""

from typing import List

import numpy as np

from .node import Node


def best_child_uct(node: Node, exploration_weight: float) -> Node:
    """Select the child of ``node`` with the highest UCB1 score.

    UCB = win_value/visits + exploration_weight * sqrt(ln(N) / n_i)

    Unvisited children receive infinite weight and are always explored
    first.

    Args:
        node: Parent node whose children are scored.
        exploration_weight: Exploration constant *c* in the UCB1 formula
            (often :math:`\\sqrt2`).

    Returns:
        The child node with the highest UCB1 weight.
    """
    choices_weights: List[float] = []
    for child in node.children:
        if child.visits == 0:
            weight: float = float("inf")
        else:
            exploitation_term: float = child.win_value / child.visits
            exploration_term: float = exploration_weight * np.sqrt(
                np.log(node.visits) / child.visits
            )
            weight = exploitation_term + exploration_term

        choices_weights.append(weight)

    return node.children[np.argmax(choices_weights)]
