import lazy_loader


__getattr__, __dir__, __all__ = lazy_loader.attach(
    __name__,
    submodules={
        "client",
        "infotree",
    },
    submod_attrs={
        "client": [
            "Lean4Client",
            "batch_verify_proof",
            "process_batch",
            "process_batches",
        ],
        "infotree": [
            "WRAPPER_TACTICS",
            "adjust_intervals",
            "ends_with_by",
            "extract_data",
            "extract_nodes_and_edges",
            "get_intervals",
            "is_balanced",
            "is_by",
            "is_calc",
            "is_wrapper",
            "merge_intervals",
            "remove_lean_comments",
            "retrieve_tactics",
            "separate_trailing_comment",
            "separate_trailing_whitespace",
            "transfer_trailing_whitespaces_and_comments",
        ],
    },
)

__all__ = [
    "Lean4Client",
    "WRAPPER_TACTICS",
    "adjust_intervals",
    "batch_verify_proof",
    "client",
    "ends_with_by",
    "extract_data",
    "extract_nodes_and_edges",
    "get_intervals",
    "infotree",
    "is_balanced",
    "is_by",
    "is_calc",
    "is_wrapper",
    "merge_intervals",
    "process_batch",
    "process_batches",
    "remove_lean_comments",
    "retrieve_tactics",
    "separate_trailing_comment",
    "separate_trailing_whitespace",
    "transfer_trailing_whitespaces_and_comments",
]
