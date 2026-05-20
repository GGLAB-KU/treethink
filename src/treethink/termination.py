from typing import List, Optional

from kimina_client import AsyncKiminaClient, KiminaClient
from loguru import logger

from .grading import has_error_response
from .methods import Node


def repl_encountered_termination(
    method,
    node: Node = None,
    client: Optional[KiminaClient] = None,
    timeout: int = 400,
    num_proc: int = 4,
    batch_size: int = 8,
):
    """Run REPL when a termination node is encountered. Can be used as
    termination_encountered_fn for method.simulate() with the following call:
    ```python
    from functools import partial

    _termination_fn = partial(
        repl_encountered_termination,
        client=client,
        timeout=timeout,
        num_proc=num_proc,
        batch_size=batch_size,
    )
    method.simulate(..., termination_encountered_fn=_termination_fn)
    ```

    TODO(burak): can we take this outside of base_method? Where could it be?
    """

    # Get proof trajectory
    proof_path = method.traverse_to_root(node, include_root=True)

    # Parse it and curate it for REPL
    parsed_proof = method.parse_proof(proof_path)
    snips = [parsed_proof]

    # Send to REPL
    logger.trace(f"Sending to Sync REPL: {snips[0][:200]}...")

    try:
        response = client.check(
            snips=snips,
            timeout=timeout,
            show_progress=False,
            batch_size=batch_size,
            max_workers=num_proc,
        )
    except Exception as e:
        logger.error(f"Sync REPL failed: {e}")
        return None

    if not response or not hasattr(response, "results"):
        logger.warning("Sync REPL returned no results")
        return None

    logger.trace(f"Sync REPL response: {response}")

    result = response.results[0]
    if not result.response:
        logger.warning("Sync REPL result has no response.")
        return None

    if not has_error_response(result.response, accept_sorry=False):
        logger.info("Sync REPL found a solution!")
        return proof_path


def repl_terminated_paths(
    method,
    node: Node = None,
    client: Optional[KiminaClient] = None,
    timeout: int = 400,
    num_proc: int = 4,
    batch_size: int = 8,
    max_repl: int = 16,
):
    """Run REPL over terminated paths (i.e. ones that end with
    Node.termination_str) and return if a successful proof is found.
    """
    node = node if node else method.root_node

    # Collect leaves that are termination nodes
    leaves: List[Node] = method.find_leaves(node)
    terminated_leaves = list(filter(lambda x: x.is_termination_node, leaves))

    if not terminated_leaves:
        logger.debug("Could not found any terminated leaves.")
        return None
    else:
        logger.debug(f"Found {len(terminated_leaves)} many terminated leaves.")

    # Limit the number of termination nodes to process
    if len(terminated_leaves) >= max_repl:
        # Sort by win value of the nodes
        terminated_leaves = sorted(
            terminated_leaves, key=lambda x: x.win_value, reverse=True
        )[:max_repl]

    # Collect proof trajectories and curate it for REPL
    proof_paths = [method.traverse_to_root(leaf) for leaf in terminated_leaves]
    snips = [method.parse_proof(proof) for proof in proof_paths]

    logger.debug(f"Prepared {len(snips)} proofs for Sync REPL verification")

    # Send to REPL
    logger.trace(f"Sending to Sync REPL: {snips[0][:200]}...")
    try:
        response = client.check(
            snips=snips,
            timeout=timeout,
            show_progress=False,
            batch_size=batch_size,
            max_workers=num_proc,
        )
    except Exception as e:
        logger.error(f"Sync REPL batch check failed: {e}")
        return None

    if not response or not hasattr(response, "results"):
        logger.warning("Sync REPL returned no results")
        return None

    # Find first successful proof
    logger.trace(f"Sync REPL response: {response}")
    for idx, result in enumerate(response.results):
        if not has_error_response(result.response, accept_sorry=False):
            logger.success(f"Sync REPL found a solution at index {idx}!")
            return proof_paths[idx]

    logger.info("No valid solution found in any terminated path.")
    return None


async def async_repl_encountered_termination(
    method,
    node: Node = None,
    client: Optional[AsyncKiminaClient] = None,
    timeout: int = 400,
    batch_size: int = 8,
    num_proc: int = 4,
):
    """
    Async version of repl_encountered_termination.

    Uses AsyncKiminaClient for non-blocking REPL verification.
    """
    logger.debug("async_repl_encountered_termination started.")

    # Get proof trajectory
    proof_path = method.traverse_to_root(node, include_root=True)

    # Parse and prepare for REPL
    parsed_proof = method.parse_proof(proof_path)
    snips = [parsed_proof]

    # Async REPL check
    logger.trace(f"Sending to Async REPL: {snips[0][:200]}...")
    try:
        response = await client.check(
            snips=snips,
            timeout=timeout,
            show_progress=False,
            batch_size=batch_size,
            max_workers=num_proc,
        )
    except Exception as e:
        logger.error(f"Async REPL failed: {e}")
        return None

    if not response or not getattr(response, "results", None):
        logger.warning("Async REPL returned no results.")
        return None

    # Check if successful (no errors)
    logger.trace(f"Async REPL response: {response}")
    result = response.results[0]
    if not has_error_response(result.response, accept_sorry=False):
        logger.success("Async REPL found a solution!")
        return proof_path

    logger.debug("Encountered terminated path is not a valid solution.")
    return None


async def async_repl_terminated_paths(
    method,
    node: Node = None,
    client: Optional[AsyncKiminaClient] = None,
    timeout: int = 400,
    num_proc: int = 4,
    batch_size: int = 8,
    max_repl: int = 16,
):
    """
    Async version of repl_terminated_paths with concurrent REPL checks.

    This method checks multiple terminated paths concurrently using
    AsyncKiminaClient, providing significant speedup.
    """
    logger.debug("async_repl_terminated_paths started.")

    node = node if node else method.root_node

    # Collect terminated leaves
    leaves = method.find_leaves(node)
    terminated_leaves = [leaf for leaf in leaves if leaf.is_termination_node]

    if not terminated_leaves:
        logger.info("No terminated leaves found.")
        return None

    logger.debug(f"Found {len(terminated_leaves)} terminated leaves.")

    # Limit to max_repl
    if len(terminated_leaves) > max_repl:
        terminated_leaves = sorted(
            terminated_leaves, key=lambda x: x.win_value, reverse=True
        )[:max_repl]
        logger.debug(
            f"Limited to top {max_repl} terminated leaves by win_value."
        )

    # Prepare proofs
    proof_paths = [method.traverse_to_root(leaf) for leaf in terminated_leaves]
    snips = [method.parse_proof(proof) for proof in proof_paths]

    logger.debug(f"Prepared {len(snips)} proofs for Async REPL verification.")
    logger.trace(f"First proof snippet: {snips[0][:200]}...")
    # Async REPL check (concurrent!)
    try:
        response = await client.check(
            snips=snips,
            timeout=timeout,
            show_progress=False,
            batch_size=batch_size,
            max_workers=num_proc,
        )
    except Exception as e:
        logger.error(f"Async REPL batch check failed: {e}")
        return None

    if not response or not getattr(response, "results", None):
        logger.warning("Async REPL returned no results")
        return None

    logger.trace(f"First Async REPL result: {response.results[0]}")
    # Find first successful proof
    for idx, result in enumerate(response.results):
        if not has_error_response(result.response, accept_sorry=False):
            logger.info(f"Async REPL found a solution at index {idx}!")
            return proof_paths[idx]

    logger.debug("No valid solution found in any terminated path.")
    return None
