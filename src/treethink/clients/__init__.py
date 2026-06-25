from treethink.clients.base import (
    AsyncProofAssistantClient,
    CheckResponse,
    ProofAssistantClient,
    ProofStateInfo,
    SnippetResult,
)
from treethink.clients.cache import (
    AsyncCachedClient,
    CachedClient,
    CacheEntry,
    ProofCache,
)

__all__ = [
    "AsyncCachedClient",
    "AsyncProofAssistantClient",
    "CachedClient",
    "CacheEntry",
    "CheckResponse",
    "ProofAssistantClient",
    "ProofCache",
    "ProofStateInfo",
    "SnippetResult",
    # Language-specific symbols loaded lazily via __getattr__
]


def __getattr__(name):
    # Lazy-load language-specific submodules on first access
    if name in (
        "WRAPPER_TACTICS",
        "AsyncLeanClientAdapter",
        "Lean4Client",
        "LeanClientAdapter",
        "adapter",
        "adjust_intervals",
        "analyze",
        "analyze_sample",
        "batch_verify_proof",
        "client",
        "ends_with_by",
        "extract_data",
        "extract_nodes_and_edges",
        "get_error_msg",
        "get_intervals",
        "get_messages_for_lines",
        "has_error_response",
        "infotree",
        "is_balanced",
        "is_by",
        "is_calc",
        "is_wrapper",
        "lean",
        "merge_intervals",
        "parse_client_response",
        "parse_error_message",
        "parse_lean_response",
        "parse_messages",
        "process_batch",
        "process_batches",
        "proof_utils",
        "remove_lean_comments",
        "retrieve_tactics",
        "separate_trailing_comment",
        "separate_trailing_whitespace",
        "split_proof_header",
        "transfer_trailing_whitespaces_and_comments",
    ):
        import treethink.clients.lean as lean_module

        return getattr(lean_module, name)

    if name in ("RocqClient", "RocqSnippetResult", "rocq", "coq"):
        import treethink.clients.coq as coq_module

        return getattr(coq_module, name)

    if name in ("IsabelleClient", "client", "isabelle"):
        import treethink.clients.isabelle as isabelle_module

        return getattr(isabelle_module, name)

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
