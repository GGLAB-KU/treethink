import lazy_loader


__getattr__, __dir__, __all__ = lazy_loader.attach(
    __name__,
    submodules={
        "rocq",
    },
    submod_attrs={
        "rocq": [
            "RocqClient",
            "RocqSnippetResult",
        ],
    },
)

__all__ = ["RocqClient", "RocqSnippetResult", "rocq"]
