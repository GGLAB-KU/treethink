"""Termination-check functions used during tree search.

These functions are language-agnostic: they receive any
:class:`~treethink.clients.base.ProofAssistantClient` and rely on its
``is_success_response`` method to decide whether a proof is valid.
"""

from typing import List, Optional

from loguru import logger

from .clients.base import ProofAssistantClient
from .methods import Node


def check_termination_encountered(
    method,
    node: Node,
    client: ProofAssistantClient,
    timeout: int = 400,
    num_proc: int = 4,
    batch_size: int = 8,
):
    """Verify a single termination node immediately when encountered.

    Called as ``termination_encountered_fn`` inside ``method.simulate()``.
    Traverses from the node to root, sends the parsed proof snippet to the
    REPL client, and returns the proof path if verified (``None`` otherwise).
    On success, sets ``best_answer_reason = CHECKED_AND_TRUE``.
    """

    proof_path = method.traverse_to_root(node, include_root=True)
    parsed_proof = method.parse_proof(proof_path)
    snips = [parsed_proof]

    logger.trace(f"Sending to REPL (encountered): {snips[0][:200]}...")

    try:
        response = client.check(
            snips=snips,
            timeout=timeout,
            show_progress=False,
            batch_size=batch_size,
            max_workers=num_proc,
        )
    except Exception as e:
        logger.error(f"REPL check failed (encountered): {e}")
        return None

    if not response or not hasattr(response, "results"):
        logger.warning("REPL returned no results (encountered).")
        return None

    result = response.results[0]
    if not result.response:
        logger.warning("REPL result has no response (encountered).")
        return None

    if client.is_success_response(result.response):
        logger.info("REPL found a solution (encountered)!")
        return proof_path

    return None


def check_terminated_paths(
    method,
    client: ProofAssistantClient,
    node: Optional[Node] = None,
    timeout: int = 400,
    num_proc: int = 4,
    batch_size: int = 8,
    max_repl: int = 16,
):
    """Batch-verify all terminated leaves after search completes.

    Collects leaves where ``is_termination_node == True``, limits to
    ``max_repl`` (prioritised by ``win_value``), and batch-verifies
    them via the REPL client.  Returns the first verified proof path
    (str) or ``None``.
    """
    node = node if node else method.root_node

    leaves: List[Node] = method.find_leaves(node)
    terminated_leaves = [leaf for leaf in leaves if leaf.is_termination_node]

    if not terminated_leaves:
        logger.debug("No terminated leaves found.")
        return None

    logger.debug(f"Found {len(terminated_leaves)} terminated leaves.")

    if len(terminated_leaves) >= max_repl:
        terminated_leaves = sorted(
            terminated_leaves, key=lambda x: x.win_value, reverse=True
        )[:max_repl]

    proof_paths = [method.traverse_to_root(leaf) for leaf in terminated_leaves]
    snips = [method.parse_proof(proof) for proof in proof_paths]

    logger.debug(f"Prepared {len(snips)} proofs for REPL verification.")
    logger.trace(f"First proof snippet: {snips[0][:200]}...")

    try:
        response = client.check(
            snips=snips,
            timeout=timeout,
            show_progress=False,
            batch_size=batch_size,
            max_workers=num_proc,
        )
    except Exception as e:
        logger.error(f"REPL batch check failed: {e}")
        return None

    if not response or not hasattr(response, "results"):
        logger.warning("REPL returned no results (terminated_paths).")
        return None

    for idx, result in enumerate(response.results):
        if client.is_success_response(result.response):
            logger.success(f"REPL found a solution at index {idx}!")
            return proof_paths[idx]

    logger.info("No valid solution found in any terminated path.")
    return None
