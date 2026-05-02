import json
import re
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
from loguru import logger
from tqdm import tqdm

from treethink.methods import Node


def save_tree_to_txt(
    root_node: Node,
    output_path: Union[str, Path],
    selected_solution: Optional[str] = None,
    problem_id: Optional[str] = None,
):
    """Save the resulting tree with to be visualized by graphviz.

    Args:
        root_node (Node): root node of the tree.
        output_path (Union[str, Path]): output path to save the tree to.
        selected_solution (Optional[str], optional): If provided will color the
            solution path with red. Defaults to None.
        problem_id (Optional[str], optional): If provided will be added to the filename
            for easier tracking of which tree belongs to which problem. Defaults to None.
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
        output_path = output_path / "dev" / Path(filename)

    with open(str(output_path), "w", encoding="utf-8") as f:
        f.write("graph\n{\n")
        root_node.print_node(
            f, 0, root_node, "a", solution=selected_solution, prev_colored=True
        )
        f.write("}\n")

    logger.info(f"Output tree saved to {output_path}.")


import codecs


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
                label_text = match.group(1)

                # Reverse the HTML escaping done in print_node
                label_text = label_text.replace("&lt;", "<")
                label_text = label_text.replace("&gt;", ">")
                label_text = label_text.replace("&quot;", '"')
                label_text = label_text.replace(
                    "&amp;", "&"
                )  # This must be last!

                # Handle escaped newlines in the label
                label_text = label_text.replace("\\n", "\n")

                # Handle Unicode escapes (\\u2115 -> ℕ)
                if "\\u" in label_text or "\\\\u" in label_text:
                    # Replace double backslash with single
                    label_text = label_text.replace("\\\\u", "\\u")
                    # Decode unicode escapes using codecs
                    try:
                        label_text = codecs.decode(label_text, "unicode_escape")
                    except:
                        # If decoding fails, keep the original
                        pass

                labels.append(label_text)

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


if __name__ == "__main__":
    path = Path("../../examples/outputs/dev")
    if not path.is_dir():
        logger.error(f"{path} is not a directory.")
    else:
        output_path = path / "extracted_proofs.json"
        convert_folder_of_txt_to_proofs(path, output_path)
