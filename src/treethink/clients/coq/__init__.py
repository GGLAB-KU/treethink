import lazy_loader

__getattr__, __dir__, __all__ = lazy_loader.attach(
    __name__,
    submodules={
        "rocq",
    },
    submod_attrs={
        "rocq": [
            "AsyncRocqClient",
            "RocqClient",
            "RocqSnippetResult",
        ],
    },
)

__all__ = ["AsyncRocqClient", "RocqClient", "RocqSnippetResult", "rocq"]
