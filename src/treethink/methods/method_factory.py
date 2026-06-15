from typing import TypeVar

from loguru import logger

from .base_method import BaseMethod
from .beam import AsyncBeamSearch, BeamSearch
from .bfts import BFTS, AsyncBFTS
from .mcts import MCTS, AsyncMCTS

IMPLEMENTED_METHODS = {
    "MCTS": MCTS,
    "BFTS": BFTS,
    "BeamSearch": BeamSearch,
    "AsyncMCTS": AsyncMCTS,
    "AsyncBFTS": AsyncBFTS,
    "AsyncBeamSearch": AsyncBeamSearch,
}

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
