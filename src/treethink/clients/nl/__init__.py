import lazy_loader

__getattr__, __dir__, __all__ = lazy_loader.attach(
    __name__,
    submodules={
        "client",
    },
    submod_attrs={
        "client": [
            "AsyncNLClient",
            "NLClient",
            "default_compare",
            "default_extract_answer",
        ],
    },
)
