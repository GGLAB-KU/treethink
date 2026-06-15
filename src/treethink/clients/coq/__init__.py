import lazy_loader


__getattr__, __dir__, __all__ = lazy_loader.attach(
    __name__,
    submodules={
        "rocq",
    },
    submod_attrs={
        "rocq": [
            "RocqBatchClient",
            "RocqSnippetResult",
        ],
    },
)

__all__ = ["RocqBatchClient", "RocqSnippetResult", "rocq"]
