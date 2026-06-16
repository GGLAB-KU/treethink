import lazy_loader


__getattr__, __dir__, __all__ = lazy_loader.attach(
    __name__,
    submodules={
        "config",
        "dry_run",
        "graph_stats",
        "iteration",
        "logging_setup",
    },
    submod_attrs={
        "config": [
            "convert_to_async",
            "parse_inference_arguments",
            "setup_model",
            "simple_messages_to_string",
        ],
        "graph_stats": [
            "build_graph_stats_payload",
            "infer_graph_dir",
        ],
        "iteration": [
            "get_time",
            "run_async_iterations",
            "run_inference_loop",
        ],
        "logging_setup": [
            "setup_logging",
        ],
    },
)

__all__ = [
    "build_graph_stats_payload",
    "config",
    "convert_to_async",
    "dry_run",
    "get_time",
    "graph_stats",
    "infer_graph_dir",
    "iteration",
    "logging_setup",
    "parse_inference_arguments",
    "run_async_iterations",
    "run_inference_loop",
    "setup_logging",
    "setup_model",
    "simple_messages_to_string",
]
