import lazy_loader

__getattr__, __dir__, __all__ = lazy_loader.attach(
    __name__,
    submodules={
        "base_method",
        "beam",
        "bfts",
        "mcts",
        "method_factory",
        "node",
    },
    submod_attrs={
        "base_method": [
            "BaseMethod",
        ],
        "beam": [
            "AsyncBeamSearch",
            "BeamSearch",
        ],
        "bfts": [
            "AsyncBFTS",
            "BFTS",
        ],
        "mcts": [
            "AsyncMCTS",
            "MCTS",
        ],
        "method_factory": [
            "IMPLEMENTED_METHODS",
            "METHODS",
            "METHOD_TYPE",
            "get_method",
        ],
        "node": [
            "Node",
            "get_total_child_num",
        ],
    },
)

__all__ = [
    "AsyncBFTS",
    "AsyncBeamSearch",
    "AsyncMCTS",
    "BFTS",
    "BaseMethod",
    "BeamSearch",
    "IMPLEMENTED_METHODS",
    "MCTS",
    "METHODS",
    "METHOD_TYPE",
    "Node",
    "base_method",
    "beam",
    "bfts",
    "get_method",
    "get_total_child_num",
    "mcts",
    "method_factory",
    "node",
]
