import lazy_loader


__getattr__, __dir__, __all__ = lazy_loader.attach(
    __name__,
    submodules={
        "alpha_zero_mcts",
        "base_method",
        "beam",
        "bfts",
        "method_factory",
        "node",
        "traditional_mcts",
    },
    submod_attrs={
        "alpha_zero_mcts": [
            "AlphaZeroMCTS",
            "AsyncAlphaZeroMCTS",
        ],
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
        "method_factory": [
            "METHODS",
            "METHOD_TYPE",
            "MethodType",
            "get_method",
        ],
        "node": [
            "Node",
            "get_total_child_num",
        ],
        "traditional_mcts": [
            "AsyncTraditionalMCTS",
            "TraditionalMCTS",
        ],
    },
)

__all__ = [
    "AlphaZeroMCTS",
    "AsyncAlphaZeroMCTS",
    "AsyncBFTS",
    "AsyncBeamSearch",
    "AsyncTraditionalMCTS",
    "BFTS",
    "BaseMethod",
    "BeamSearch",
    "METHODS",
    "METHOD_TYPE",
    "MethodType",
    "Node",
    "TraditionalMCTS",
    "alpha_zero_mcts",
    "base_method",
    "beam",
    "bfts",
    "get_method",
    "get_total_child_num",
    "method_factory",
    "node",
    "traditional_mcts",
]
