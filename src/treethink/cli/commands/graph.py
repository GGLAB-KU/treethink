"""``treethink graph`` — interact with saved tree graphs."""

import json
from pathlib import Path
from typing import Optional

import typer
from loguru import logger

from treethink.graph import (
    analyze_graph_stats,
    convert_folder_of_txt_to_proofs,
    load_graphviz_state,
    run_graphviz_on_file,
)


def visualize(
    ctx: typer.Context,
    path: str = typer.Argument(
        ...,
        help="Path to a graphviz .txt file or a directory containing .txt files. "
        "If a directory, all .txt files will be rendered.",
    ),
    output_dir: Optional[str] = typer.Option(
        None,
        "-o",
        "--output-dir",
        help="Output directory for rendered images. "
        "Defaults to the same directory as the input file.",
    ),
):
    """Render a tree graph from a graphviz .txt file to PNG/SVG.

    Requires ``graphviz`` (the ``dot`` command) to be installed.

    Example::

        treethink graph visualize outputs/tree_20250101_120000.txt
    """
    _path = Path(path)

    if _path.is_dir():
        txt_files = list(_path.glob("*.txt"))
        if not txt_files:
            logger.error(f"No .txt files found in {_path}")
            raise typer.Exit(code=1)
        for f in txt_files:
            run_graphviz_on_file(f)
        logger.success(f"Rendered {len(txt_files)} trees from {_path}")
    else:
        if not _path.exists():
            logger.error(f"File not found: {_path}")
            raise typer.Exit(code=1)
        run_graphviz_on_file(_path)
        logger.success(f"Rendered {_path}")


def extract(
    ctx: typer.Context,
    folder: str = typer.Argument(
        ...,
        help="Folder containing graphviz .txt tree files.",
    ),
    output: Optional[str] = typer.Option(
        None,
        "-o",
        "--output",
        help="Output JSON file path. Defaults to ``extracted_proofs.json`` "
        "inside the input folder.",
    ),
    identifier: str = typer.Option(
        "color=red",
        "--identifier",
        help="Graphviz attribute used to identify solution-path nodes "
        "(default: ``color=red``).",
    ),
):
    """Extract solutions from graphviz .txt files.

    Walks through all ``.txt`` files in *folder*, extracts the text labels
    of solution-path nodes, and writes them to a JSON file.

    Example::

        treethink graph extract outputs/graphs_0/ -o solutions.json
    """
    _folder = Path(folder)
    if not _folder.is_dir():
        logger.error(f"{_folder} is not a directory")
        raise typer.Exit(code=1)

    _output = Path(output) if output else None
    convert_folder_of_txt_to_proofs(
        _folder,
        _output or (_folder / "extracted_proofs.json"),
        identifier=identifier,
    )


def analyze(
    ctx: typer.Context,
    folder: str = typer.Argument(
        ...,
        help="Folder containing experiment stats JSON files "
        "(``*_exp_*.json`` or ``*_exp_stats_*.json``).",
    ),
):
    """Compute aggregate statistics across multiple experiment runs.

    Scans *folder* for experiment JSON files (matching ``*_exp_*.json``
    and ``*_exp_stats_*.json``), loads their ``graph_stats_summary``,
    and prints combined averages, medians, and standard deviations.

    Example::

        treethink graph analyze outputs/
    """
    _folder = Path(folder)
    if not _folder.is_dir():
        logger.error(f"{_folder} is not a directory")
        raise typer.Exit(code=1)

    all_stats = []
    for pattern in ["*_exp_*.json", "*_exp_stats_*.json"]:
        for f in _folder.glob(pattern):
            try:
                with open(f, "r") as fh:
                    data = json.load(fh)
                summary = data.get("graph_stats_summary")
                if summary:
                    all_stats.append(summary)
            except Exception as e:
                logger.warning(f"Failed to read {f}: {e}")

    if not all_stats:
        logger.warning(f"No experiment stats found in {_folder}")
        return

    aggregated = analyze_graph_stats(all_stats)
    print(json.dumps(aggregated, indent=2))
    logger.success(
        f"Aggregated stats across {len(all_stats)} experiment files."
    )


def info(
    ctx: typer.Context,
    path: str = typer.Argument(
        ...,
        help="Path to a graphviz .txt tree file.",
    ),
    metadata: Optional[str] = typer.Option(
        None,
        "--metadata",
        help="Optional path to a JSON metadata file (e.g. experiment stats) "
        "used to reconstruct TreeThinkArgs for the tree.",
    ),
):
    """Inspect a saved tree: show structure, node counts, and metadata.

    Rebuilds the tree from the graphviz file and prints summary info.

    Example::

        treethink graph info outputs/tree_my_problem_20250101.txt \\
            --metadata outputs/0_exp_run_20250101.json
    """
    _path = Path(path)
    if not _path.exists():
        logger.error(f"File not found: {_path}")
        raise typer.Exit(code=1)

    meta = None
    if metadata:
        _meta_path = Path(metadata)
        if _meta_path.exists():
            with open(_meta_path, "r") as f:
                meta = json.load(f)

    state = load_graphviz_state(_path, metadata=meta)
    root = state["root_node"]

    print(f"Tree file: {_path}")
    print(
        f"  Root node text: {root.text[:100]}..."
        if root.text
        else "  Root node text: (empty)"
    )
    print(f"  Total nodes (incl. root): {_count_nodes(root)}")
    print(f"  Tree depth: {_max_depth(root)}")
    print(f"  Max children: {root.max_children}")
    print(f"  Method: {state.get('treethink_args')}")

    if metadata:
        print(f"  Metadata file: {_metadata_summary(meta)}")


def _count_nodes(node) -> int:
    """Count total nodes in the tree rooted at *node*."""
    count = 1
    for child in node.children:
        count += _count_nodes(child)
    return count


def _max_depth(node, depth=0) -> int:
    """Compute maximum depth of the tree rooted at *node*."""
    if not node.children:
        return depth
    return max(_max_depth(child, depth + 1) for child in node.children)


def _metadata_summary(meta: dict) -> str:
    """Return a short summary string for a metadata dict."""
    parts = []
    for key in ("run_name", "method_name", "iteration", "graph_stats_count"):
        if key in meta:
            parts.append(f"{key}={meta[key]}")
    return ", ".join(parts)
