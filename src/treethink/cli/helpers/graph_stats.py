"""Helpers for building graph-statistics payloads during inference."""

from pathlib import Path
from typing import Any, Dict, List, Optional

from treethink.graph import analyze_graph_stats
from treethink.utils.parser import serialize_args


def infer_graph_dir(
    treethink_args, output_path: Path, iteration_index: int, num_iterations: int
) -> Path:
    """Determine the directory where tree graph files are written or expected.

    Args:
        treethink_args: The TreeThink configuration (may hold ``graph_path``).
        output_path: Base output directory.
        iteration_index: Current iteration index.
        num_iterations: Total number of iterations.

    Returns:
        Path to the graph directory.
    """
    base_graph_dir = output_path
    if treethink_args and treethink_args.graph_path:
        graph_path = Path(treethink_args.graph_path)
        if graph_path.suffix:
            base_graph_dir = graph_path.parent
        else:
            base_graph_dir = graph_path / "dev"

    if num_iterations > 1 and base_graph_dir == output_path:
        candidate = output_path / f"graphs_{iteration_index}"
        if candidate.exists() and any(candidate.glob("*.txt")):
            return candidate
    return base_graph_dir


def build_graph_stats_payload(
    model,
    results: List[Dict],
    output_path: Path,
    run_name: str,
    iteration_index: int,
    timestamp: str,
    num_iterations: int,
) -> Optional[Dict[str, Any]]:
    """Aggregate graph statistics from model results into a serialisable payload.

    Args:
        model: The sampler model (may hold ``treethink_args`` etc.).
        results: List of result datapoints.
        output_path: Base output directory.
        run_name: Run name identifier.
        iteration_index: Current iteration index.
        timestamp: Formatted timestamp string.
        num_iterations: Total number of iterations.

    Returns:
        A dict containing experiment metadata and aggregated graph stats,
        or ``None`` if no ``graph_stats`` were found.
    """
    if not results:
        return None

    graph_stats_list = [r["graph_stats"] for r in results if "graph_stats" in r]
    if not graph_stats_list:
        return None

    treethink_args = getattr(model, "treethink_args", None)
    policy_args = getattr(model, "policy_args", None)
    evaluator_args = getattr(model, "evaluator_args", None)
    method_name = (
        treethink_args.method_name if treethink_args is not None else None
    )

    graph_dir = infer_graph_dir(
        treethink_args, output_path, iteration_index, num_iterations
    )

    return {
        "run_name": run_name,
        "iteration": iteration_index,
        "timestamp": timestamp,
        "output_dir": str(output_path),
        "graph_dir": str(graph_dir),
        "graph_path": getattr(treethink_args, "graph_path", None)
        if treethink_args
        else None,
        "method_name": method_name,
        "treethink_args": serialize_args(treethink_args),
        "policy_args": serialize_args(policy_args),
        "evaluator_args": serialize_args(evaluator_args),
        "graph_stats_summary": analyze_graph_stats(graph_stats_list),
        "graph_stats_count": len(graph_stats_list),
    }
