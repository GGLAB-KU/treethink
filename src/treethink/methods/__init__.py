from enum import Enum
from typing import Callable, List, TypeVar

from loguru import logger

# ── Re-export key symbols for external consumers ────────────────────────
# Other modules (evaluators, graph, etc.) import BaseMethod, Node, etc.
# from treethink.methods – keep these re-exports in sync.
from .base_method import BaseMethod  # noqa: F401
from .node import Node, get_total_child_num  # noqa: F401


class MethodType(Enum):
    """Enum mapping method config names to their implementation classes.

    Members are accessed via ``from_str()`` which normalises the config
    ``method_name`` (e.g. ``"AlphaZeroMCTS"`` → ``MethodType.ALPHA_ZERO_MCTS``).

    To add a new method, add a member here and ensure the value is a
    ``BaseMethod`` subclass.  Both sync and async variants are listed
    as enum members.
    """

    # ── Sync methods ───────────────────────────────────────────────────
    ALPHA_ZERO_MCTS = "AlphaZeroMCTS"
    BEAM_SEARCH = "BeamSearch"
    BFTS = "BFTS"
    TRADITIONAL_MCTS = "TraditionalMCTS"

    # ── Async methods ──────────────────────────────────────────────────
    ASYNC_ALPHA_ZERO_MCTS = "AsyncAlphaZeroMCTS"
    ASYNC_BEAM_SEARCH = "AsyncBeamSearch"
    ASYNC_BFTS = "AsyncBFTS"
    ASYNC_TRADITIONAL_MCTS = "AsyncTraditionalMCTS"

    @classmethod
    def from_str(cls, name: str) -> "MethodType":
        """Resolve a config ``method_name`` to a ``MethodType`` member."""
        normalized = name.strip()
        for member in cls:
            if member.value == normalized:
                return member
        valid_keys = [m.value for m in cls]
        raise ValueError(
            f"Unknown method '{name}'. Valid options: {valid_keys}"
        )

    def initialize(
        self, root_node, policy, evaluator, *args, **kwargs
    ) -> BaseMethod:
        """Instantiate the method class with the given arguments."""
        return _METHOD_CLASSES[self](
            root_node, policy, evaluator, *args, **kwargs
        )

    def async_variant(self) -> "MethodType":
        """Return the async version of this method, if available.

        Uses the convention ``"Async" + <sync_name>``.  Raises
        ``ValueError`` if no async variant is registered.
        """
        async_name = f"Async{self.value}"
        return MethodType.from_str(async_name)

    @property
    def is_async(self) -> bool:
        """Whether this method is an async variant."""
        return self.value.startswith("Async")


# ── Internal: mapping from MethodType → implementation classes ──────────

try:
    from .alpha_zero_mcts import AlphaZeroMCTS, AsyncAlphaZeroMCTS
    from .beam import AsyncBeamSearch, BeamSearch
    from .bfts import BFTS, AsyncBFTS
    from .traditional_mcts import AsyncTraditionalMCTS, TraditionalMCTS

    _ASYNC_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Async methods not available: {e}")
    _ASYNC_AVAILABLE = False

_METHOD_CLASSES = {
    MethodType.ALPHA_ZERO_MCTS: AlphaZeroMCTS,
    MethodType.BEAM_SEARCH: BeamSearch,
    MethodType.BFTS: BFTS,
    MethodType.TRADITIONAL_MCTS: TraditionalMCTS,
}

if _ASYNC_AVAILABLE:
    _METHOD_CLASSES.update(
        {
            MethodType.ASYNC_ALPHA_ZERO_MCTS: AsyncAlphaZeroMCTS,
            MethodType.ASYNC_BEAM_SEARCH: AsyncBeamSearch,
            MethodType.ASYNC_BFTS: AsyncBFTS,
            MethodType.ASYNC_TRADITIONAL_MCTS: AsyncTraditionalMCTS,
        }
    )

METHODS: List[str] = [m.value for m in MethodType]
METHOD_TYPE = TypeVar("METHOD_TYPE", bound=BaseMethod)


def get_method(treethink_config, root_node, policy, evaluator):
    """Instantiate a method from config using :class:`MethodType`."""
    try:
        method_type = MethodType.from_str(treethink_config.method_name)
        return method_type.initialize(
            root_node=root_node,
            policy=policy,
            evaluator=evaluator,
            **treethink_config,
        )
    except (ValueError, KeyError):
        logger.error(
            f"Could not find method: {treethink_config.method_name}. "
            + f"Available methods: {[m.value for m in MethodType]}"
        )
        raise


__all__ = [
    "AlphaZeroMCTS",
    "AsyncAlphaZeroMCTS",
    "AsyncTraditionalMCTS",
    "BFTS",
    "BaseMethod",
    "BeamSearch",
    "MethodType",
    "Node",
    "TraditionalMCTS",
    "alpha_zero_mcts",
    "base_method",
    "beam",
    "bfts",
    "get_method",
    "get_total_child_num",
    "node",
    "traditional_mcts",
]
