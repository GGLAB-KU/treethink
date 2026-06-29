import codecs
import json
import re
import subprocess
from dataclasses import is_dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Tuple,
    Union,
    get_args,
    get_origin,
    get_type_hints,
)

import numpy as np
from loguru import logger
from tqdm import tqdm

from treethink.methods import Node

from .utils.enums import coerce_enum


def save_tree_to_txt(
    root_node: Node,
    output_path: Union[str, Path],
    selected_solution: Optional[str] = None,
    problem_id: Optional[str] = None,
):
    """Export a search tree to graphviz DOT format for rendering.

    Args:
        root_node: Root node of the tree.
        output_path: File or directory path.  If a directory, a timestamped
            filename is generated.
        selected_solution: If provided, nodes on the solution path are
            coloured red.
        problem_id: Included in the filename for tracking.

    The output file uses a simplified DOT format::

        graph
        {
            "node_0" [label="...", color="blue"];
            "node_0" -- "node_1";
        }
    """
    output_path = Path(output_path)
    if output_path.is_dir():
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # Add problem_id to filename if provided
        if problem_id:
            # Sanitize problem_id for filename (remove special chars)
            safe_problem_id = re.sub(r"[^\w\-_]", "_", str(problem_id))
            filename = f"tree_{safe_problem_id}_{timestamp}.txt"
        else:
            filename = f"tree_{timestamp}.txt"
        output_path = output_path / Path(filename)

    with open(str(output_path), "w", encoding="utf-8") as f:
        f.write("graph\n{\n")
        root_node.print_node(f, 0, "a", solution=selected_solution)
        f.write("}\n")

    logger.info(f"Output tree saved to {output_path}.")


def _unescape_graphviz_label(label_text: str) -> str:
    label_text = label_text.replace("&lt;", "<")
    label_text = label_text.replace("&gt;", ">")
    label_text = label_text.replace("&quot;", '"')
    label_text = label_text.replace("&amp;", "&")

    label_text = label_text.replace("\\n", "\n")

    if "\\u" in label_text or "\\\\u" in label_text:
        label_text = label_text.replace("\\\\u", "\\u")
        try:
            label_text = codecs.decode(label_text, "unicode_escape")
        except Exception:
            pass

    return label_text


def extract_solution_from_graphviz(
    graphviz_content: str, identifier: str = "color=red"
):
    """
    Extract text labels from nodes with identifier in each line in a
    graphviz file.

    Args:
        graphviz_content: String containing the graphviz file content
        identifier: String to be used as identifier. Defaults to color=red".

    Returns:
        List of text labels from nodes
    """
    labels = []

    # Split into lines
    lines = graphviz_content.strip().split("\n")

    for line in lines:
        # Check if this line defines a node with identifier
        if "[label=" in line and identifier in line:
            # Extract the label content between <TD ROWSPAN="3"> and </TD>
            match = re.search(r'<TD ROWSPAN="3">(.*?)</TD>', line)
            if match:
                labels.append(_unescape_graphviz_label(match.group(1)))

    return "".join(labels)


def convert_folder_of_txt_to_proofs(
    folder_path: Union[Path, str],
    output_path: Union[Path, str],
    identifier: str = "color=red",
):
    """Convert a folder of .txt files into a .json file containing extracted
    solutions using `extract_solution_from_graphviz`.

    Args:
        folder_path (Path): Folder containing graphviz .txt outputs.
        output_path (Path): Path to the output file.
        identifier (str, optional): String to be used as identifier. Defaults to
            "color=red".
    """
    output_path = Path(output_path)
    if output_path.is_dir():
        output_path = output_path / "extracted_proofs.json"

    folder_path = Path(folder_path)
    if not folder_path.is_dir():
        logger.error(f"{folder_path} is not a directory.")
        return
    logger.info(f"Extracting proofs from {folder_path}.")

    txt_files = list(folder_path.glob("*.txt"))
    logger.debug(f"Found {len(txt_files)} text files.")

    proofs = []
    for txt_file in tqdm(txt_files):
        try:
            with open(txt_file, "r", encoding="utf-8") as f:
                content = f.read()
            solution = extract_solution_from_graphviz(content, identifier)
            if solution:
                # extract_solution_from_graphviz already handles all the unescaping
                proofs.append({"id": txt_file.stem, "output": [solution]})
        except Exception as e:
            logger.error(f"Skipping {txt_file.stem} because of error: {e}")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(proofs, f, indent=2, ensure_ascii=False)
    logger.success(f"Proofs saved to {output_path}.")


def run_graphviz_on_file(txt_path: Union[str, Path]):
    """Run graphviz subprocess on the given txt and visualize the tree.
    Save the result to the same folder.
    """

    output_tree_path = Path(txt_path).with_suffix(".svg")
    try:
        subprocess.run(
            ["dot", "-Tpng", str(txt_path), "-o", output_tree_path],
            check=True,
            capture_output=True,
        )
        logger.info(f"PNG file saved to {output_tree_path}")
    except subprocess.CalledProcessError as e:
        logger.error(f"Error generating graph: {e}")
    except FileNotFoundError:
        logger.error(
            "Graphviz 'dot' command not found. Please install graphviz."
            + " See: https://graphviz.org/download/"
        )


def vis_proof_tree(path: Union[str, Path]):
    """Run both `run_graphviz` and `save_tree_to_txt` and save
    them to `path`.

    If `path` is a filename then both .txt and .png file will share the same
    name. Otherwise, default filename will be used for .txt and .png.
    """
    path = Path(path)
    if path.is_dir():
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = path / Path(f"tree_{timestamp}.txt")

    save_tree_to_txt(path)
    run_graphviz_on_file(path)


def analyze_graph_stats(stats: List[Dict[str, Any]]):
    """Calculate std, avg and median of graph statistics for a list of graph
    stats that is stored along the model output."""
    if not stats:
        return {}

    analysis = {}
    keys = stats[0].keys()

    for key in keys:
        values = [s[key] for s in stats]
        analysis[f"AVG_{key}"] = float(np.average(values))
        analysis[f"MEDIAN_{key}"] = float(np.median(values))
        analysis[f"STD_{key}"] = float(np.std(values))

    return analysis


def _is_optional_type(type_hint) -> bool:
    origin = get_origin(type_hint)
    if origin is None:
        return False
    args = get_args(type_hint)
    return type(None) in args


def _get_optional_inner_type(type_hint):
    if not _is_optional_type(type_hint):
        return type_hint
    return next(arg for arg in get_args(type_hint) if arg is not type(None))


def _is_enum_type(type_hint) -> bool:
    return isinstance(type_hint, type) and issubclass(type_hint, Enum)


def _coerce_dataclass_value(expected_type, value):
    if _is_optional_type(expected_type):
        inner_type = _get_optional_inner_type(expected_type)
        if value is None:
            return None
        return _coerce_dataclass_value(inner_type, value)

    if is_dataclass(expected_type) and isinstance(value, dict):
        return _dataclass_from_dict(expected_type, value)

    if _is_enum_type(expected_type):
        return coerce_enum(value, expected_type)

    return value


def _dataclass_from_dict(dataclass_type, data: Dict[str, Any]):
    if data is None:
        return None
    if not is_dataclass(dataclass_type):
        raise ValueError(f"{dataclass_type} is not a dataclass")

    type_hints = get_type_hints(dataclass_type)
    parsed = {}
    for key, value in data.items():
        if key not in type_hints:
            continue
        expected = type_hints[key]
        parsed[key] = _coerce_dataclass_value(expected, value)
    return dataclass_type(**parsed)


def _parse_label_value(label: str, key: str) -> Optional[str]:
    match = re.search(rf"{re.escape(key)}:([^<]+)", label)
    if not match:
        return None
    return match.group(1).strip()


def _parse_node_label(label: str) -> Dict[str, Any]:
    text_match = re.search(r'<TD ROWSPAN="3">(.*?)</TD>', label)
    text = _unescape_graphviz_label(text_match.group(1)) if text_match else ""

    win_value_raw = _parse_label_value(label, "W")
    visits_raw = _parse_label_value(label, "V")
    level_raw = _parse_label_value(label, "L")

    try:
        win_value = float(win_value_raw) if win_value_raw is not None else 0.0
    except ValueError:
        win_value = 0.0

    try:
        visits = int(visits_raw) if visits_raw is not None else 0
    except ValueError:
        visits = 0

    try:
        level = int(level_raw) if level_raw is not None else 0
    except ValueError:
        level = 0

    return {
        "text": text,
        "win_value": win_value,
        "visits": visits,
        "level": level,
    }


def _parse_graphviz_content(
    graphviz_content: str,
) -> Tuple[Dict[str, Dict[str, Any]], List[Tuple[str, str]]]:
    nodes: Dict[str, Dict[str, Any]] = {}
    edges: List[Tuple[str, str]] = []

    for line in graphviz_content.splitlines():
        stripped = line.strip()
        if not stripped:
            continue

        node_match = re.match(
            r"^(?P<node_id>[^\s]+)\s+\[label=(?P<label><<TABLE.*</TABLE>>)",
            stripped,
        )
        if node_match:
            node_id = node_match.group("node_id")
            label = node_match.group("label")
            nodes[node_id] = _parse_node_label(label)
            continue

        edge_match = re.match(
            r"^(?P<left>[^\s]+)\s+--\s+(?P<right>[^\s]+)",
            stripped,
        )
        if edge_match:
            edges.append((edge_match.group("left"), edge_match.group("right")))

    return nodes, edges


def _merge_node_text(parent_text: str, diff_text: str) -> str:
    if not diff_text:
        return parent_text or ""
    if not parent_text:
        return diff_text
    parent_lines = parent_text.split("\n")
    diff_lines = diff_text.split("\n")
    merged_lines = parent_lines[:]
    for line in diff_lines:
        if line not in parent_lines:
            merged_lines.append(line)
    return "\n".join(merged_lines)


def _infer_parent_map(
    nodes: Dict[str, Dict[str, Any]], edges: List[Tuple[str, str]]
) -> Dict[str, str]:
    parent_map: Dict[str, str] = {}

    for node_id in nodes.keys():
        if "_" in node_id:
            parent_id = node_id.rsplit("_", 1)[0]
            if parent_id in nodes:
                parent_map[node_id] = parent_id

    for left, right in edges:
        if left not in nodes or right not in nodes:
            continue
        if (
            right not in parent_map
            and nodes[left]["level"] + 1 == nodes[right]["level"]
        ):
            parent_map[right] = left
        elif (
            left not in parent_map
            and nodes[right]["level"] + 1 == nodes[left]["level"]
        ):
            parent_map[left] = right

    return parent_map


def _infer_children_order(
    nodes: Dict[str, Dict[str, Any]],
    edges: List[Tuple[str, str]],
    parent_map: Dict[str, str],
) -> Dict[str, List[str]]:
    children: Dict[str, List[str]] = {node_id: [] for node_id in nodes}
    seen = set()

    for left, right in edges:
        if left not in nodes or right not in nodes:
            continue
        if parent_map.get(right) == left:
            parent_id, child_id = left, right
        elif parent_map.get(left) == right:
            parent_id, child_id = right, left
        elif right.startswith(left + "_"):
            parent_id, child_id = left, right
        elif left.startswith(right + "_"):
            parent_id, child_id = right, left
        elif nodes[left]["level"] + 1 == nodes[right]["level"]:
            parent_id, child_id = left, right
        elif nodes[right]["level"] + 1 == nodes[left]["level"]:
            parent_id, child_id = right, left
        else:
            continue

        if (parent_id, child_id) in seen:
            continue
        children[parent_id].append(child_id)
        seen.add((parent_id, child_id))

    for child_id, parent_id in parent_map.items():
        if (parent_id, child_id) not in seen:
            children[parent_id].append(child_id)
            seen.add((parent_id, child_id))

    return children


def _get_root_id(nodes: Dict[str, Dict[str, Any]], parent_map: Dict[str, str]):
    root_candidates = [
        node_id for node_id in nodes if node_id not in parent_map
    ]
    if not root_candidates:
        raise ValueError("No root node found in graphviz content.")
    return min(
        root_candidates,
        key=lambda node_id: (nodes[node_id]["level"], len(node_id)),
    )


def _load_metadata(metadata):
    if metadata is None:
        return None
    if isinstance(metadata, (str, Path)):
        with open(metadata, "r", encoding="utf-8") as f:
            return json.load(f)
    if isinstance(metadata, dict):
        return metadata
    raise TypeError("metadata must be a dict or a path to a JSON file")


def _build_args_from_metadata(metadata):
    if not metadata:
        return None, None, None

    from treethink import EvaluatorArgs, PolicyArgs, TreeThinkArgs

    treethink_args = None
    policy_args = None
    evaluator_args = None

    if metadata.get("treethink_args"):
        treethink_args = _dataclass_from_dict(
            TreeThinkArgs, metadata["treethink_args"]
        )
    if metadata.get("policy_args"):
        policy_args = _dataclass_from_dict(PolicyArgs, metadata["policy_args"])
    if metadata.get("evaluator_args"):
        evaluator_args = _dataclass_from_dict(
            EvaluatorArgs, metadata["evaluator_args"]
        )

    return treethink_args, policy_args, evaluator_args


def build_tree_from_graphviz(
    graphviz_content: str,
    metadata: Optional[Dict[str, Any]] = None,
) -> Node:
    nodes, edges = _parse_graphviz_content(graphviz_content)
    if not nodes:
        raise ValueError("No nodes found in graphviz content.")

    metadata = _load_metadata(metadata)
    treethink_args, _, _ = _build_args_from_metadata(metadata)

    max_children = treethink_args.max_children if treethink_args else None
    termination_str = treethink_args.termination_str if treethink_args else None
    exploration_weight = (
        treethink_args.exploration_weight if treethink_args else None
    )

    parent_map = _infer_parent_map(nodes, edges)
    children_map = _infer_children_order(nodes, edges, parent_map)
    root_id = _get_root_id(nodes, parent_map)

    def _build_node(node_id: str, parent: Optional[Node] = None) -> Node:
        info = nodes[node_id]
        if parent is None:
            full_text = info["text"]
        else:
            full_text = _merge_node_text(parent.text or "", info["text"])

        node = Node(
            text=full_text,
            max_children=max_children,
            level=info["level"],
            parent=None,
            termination_str=termination_str,
            exploration_weight=exploration_weight,
        )
        node.win_value = info["win_value"]
        node.visits = info["visits"]
        node.graphviz_id = node_id

        if parent is not None:
            parent.add_child(node)

        for child_id in children_map.get(node_id, []):
            _build_node(child_id, node)

        return node

    return _build_node(root_id)


def load_graphviz_state(
    txt_path: Union[str, Path],
    metadata: Optional[Union[str, Path, Dict[str, Any]]] = None,
    policy=None,
    evaluator=None,
):
    metadata_dict = _load_metadata(metadata)
    txt_path = Path(txt_path)
    with open(txt_path, "r", encoding="utf-8") as f:
        content = f.read()

    root_node = build_tree_from_graphviz(content, metadata_dict)
    treethink_args, policy_args, evaluator_args = _build_args_from_metadata(
        metadata_dict
    )

    method = None
    if treethink_args:
        if policy is None and policy_args is not None:
            from treethink import get_policy_from_config

            policy = get_policy_from_config(policy_args)
        if evaluator is None and evaluator_args is not None:
            from treethink import get_evaluator_from_config

            evaluator = get_evaluator_from_config(evaluator_args)

        if policy is not None and evaluator is not None:
            from treethink import get_method

            method = get_method(
                treethink_config=treethink_args,
                root_node=root_node,
                policy=policy,
                evaluator=evaluator,
            )

    return {
        "root_node": root_node,
        "method": method,
        "treethink_args": treethink_args,
        "policy_args": policy_args,
        "evaluator_args": evaluator_args,
    }


if __name__ == "__main__":
    path = Path("../../examples/outputs/dev")
    if not path.is_dir():
        logger.error(f"{path} is not a directory.")
    else:
        output_path = path / "extracted_proofs.json"
        convert_folder_of_txt_to_proofs(path, output_path)
