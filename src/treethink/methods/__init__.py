from typing import TypeVar

from loguru import logger

# <AUTOGEN_INIT>
from treethink.methods import (
    alpha_zero_mcts,
    base_method,
    beam,
    bfts,
    node,
    traditional_mcts,
)
from treethink.methods.alpha_zero_mcts import (
    AlphaZeroMCTS,
    AsyncAlphaZeroMCTS,
)
from treethink.methods.base_method import (
    BaseMethod,
)
from treethink.methods.beam import (
    AsyncBeamSearch,
    BeamSearch,
)
from treethink.methods.bfts import (
    BFTS,
    AsyncBFTS,
)
from treethink.methods.node import (
    Node,
    get_total_child_num,
)
from treethink.methods.traditional_mcts import (
    AsyncTraditionalMCTS,
    TraditionalMCTS,
)

# </AUTOGEN_INIT>

# Import async methods
try:
    from .alpha_zero_mcts import AsyncAlphaZeroMCTS
    from .beam import AsyncBeamSearch
    from .bfts import AsyncBFTS
    from .traditional_mcts import AsyncTraditionalMCTS

    ASYNC_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Async methods not available: {e}")
    ASYNC_AVAILABLE = False

IMPLEMENTED_METHODS = {
    "AlphaZeroMCTS": AlphaZeroMCTS,
    "BFTS": BFTS,
    "BeamSearch": BeamSearch,
    "TraditionalMCTS": TraditionalMCTS,
}

# Add async methods if available
if ASYNC_AVAILABLE:
    IMPLEMENTED_METHODS.update(
        {
            "AsyncBeamSearch": AsyncBeamSearch,
            "AsyncBFTS": AsyncBFTS,
            "AsyncAlphaZeroMCTS": AsyncAlphaZeroMCTS,
            "AsyncTraditionalMCTS": AsyncTraditionalMCTS,
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
    "AlphaZeroMCTS",
    "AsyncAlphaZeroMCTS",
    "AsyncTraditionalMCTS",
    "BFTS",
    "BaseMethod",
    "BeamSearch",
    "Node",
    "TraditionalMCTS",
    "alpha_zero_mcts",
    "base_method",
    "beam",
    "bfts",
    "get_total_child_num",
    "node",
    "traditional_mcts",
    "get_method",
]
