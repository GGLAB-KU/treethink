import lazy_loader


__getattr__, __dir__, __all__ = lazy_loader.attach(
    __name__,
    submodules={
        "graph",
        "help_cmd",
        "run",
    },
    submod_attrs={
        "graph": [
            "analyze",
            "extract",
            "info",
            "visualize",
        ],
        "help_cmd": [
            "show_help",
        ],
        "run": [
            "run",
        ],
    },
)

__all__ = [
    "analyze",
    "extract",
    "graph",
    "help_cmd",
    "info",
    "run",
    "show_help",
    "visualize",
]
