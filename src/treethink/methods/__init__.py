from typing import TypeVar

from loguru import logger

from .base_method import BaseMethod
from .beam import BeamSearch
from .bfts import BFTS
from .mcts import MCTS
from .node import Node, get_total_child_num

# Import async methods
try:
    from .beam import AsyncBeamSearch  # AsyncBeamSearch is in beam.py
    from .bfts import AsyncBFTS  # AsyncBFTS is in bfts.py
    from .mcts import AsyncMCTS  # AsyncMCTS is in mcts.py

    ASYNC_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Async methods not available: {e}")
    ASYNC_AVAILABLE = False

IMPLEMENTED_METHODS = {
    "MCTS": MCTS,
    "BFTS": BFTS,
    "BeamSearch": BeamSearch,
}

# Add async methods if available
if ASYNC_AVAILABLE:
    IMPLEMENTED_METHODS.update(
        {
            "AsyncBeamSearch": AsyncBeamSearch,
            "AsyncBFTS": AsyncBFTS,
            "AsyncMCTS": AsyncMCTS,
        }
    )

METHODS = list(IMPLEMENTED_METHODS.keys())
METHOD_TYPE = TypeVar("METHOD_TYPE", bound=BaseMethod)


def get_method(treethink_config, root_node, policy, evaluator):
    try:
        return IMPLEMENTED_METHODS[treethink_config.method_name](
            root_node=root_node,
            policy=policy,
            evaluator=evaluator,
            **treethink_config,
        )
    except KeyError:
        logger.error(
            f"Could not found method: {treethink_config.method_name}. "
            + f"Available methods are: {IMPLEMENTED_METHODS.keys()}"
        )


__all__ = [
    "BFTS",
    "BaseMethod",
    "BeamSearch",
    "MCTS",
    "Node",
    "base_method",
    "beam",
    "bfts",
    "get_total_child_num",
    "mcts",
    "node",
]

# Add async methods to __all__ if available
if ASYNC_AVAILABLE:
    __all__.extend(
        [
            "AsyncBeamSearch",
            "AsyncBFTS",
            "AsyncMCTS",
        ]
    )
