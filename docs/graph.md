# Graph — Tree Visualization & Statistics

TreeThink can **serialise search trees** to graphviz-compatible `.txt` files
for visualisation and analysis.

---

## save_tree_to_txt()

**File:** `src/treethink/graph.py`

The main function for exporting a tree to graphviz format:

```python
save_tree_to_txt(
    root_node,              # Root Node of the search tree
    output_path,            # File or directory path
    selected_solution=None, # Optional: highlight solution path in red
    problem_id=None,        # Optional: included in filename for tracking
)
```

- If `output_path` is a directory, a timestamped filename is generated:
  `tree_{problem_id}_{timestamp}.txt`
- The output uses graphviz DOT format:

```
graph
{
    "node_0" [label="problem text...", color="blue"];
    "node_1" [label="child text...", color="black"];
    "node_0" -- "node_1";
}
```

- If `selected_solution` is provided, nodes on the solution path are
  coloured **red**.
- Win values are displayed per node.

---

## Rendering with Graphviz

The `.txt` files can be rendered with standard graphviz tools:

```bash
# Convert to PNG
dot -Tpng tree_20240607_120000.txt -o tree.png

# Convert to SVG
dot -Tsvg tree_20240607_120000.txt -o tree.svg
```

Or via the CLI:

```bash
treethink graph visualize input.txt --output output.png --format png
```

---

## CLI Graph Commands

See [cli.md](cli.md) for the full `treethink graph` subcommand reference.

| Command | Description |
|---------|-------------|
| `treethink graph visualize` | Render `.txt` → PNG/SVG |
| `treethink graph extract` | Extract solutions from `.txt` files |
| `treethink graph analyze` | Compute aggregate stats across runs |
| `treethink graph info` | Show tree structure and metadata |

---

## Graph Statistics

When `store_graph_stats` is enabled in the config, the method collects:

- **Tree depth** — maximum node level
- **Node count** — total nodes in the tree
- **Branching factor** — average/max children per node
- **Termination count** — number of leaves with `is_termination_node == True`
- **Solution path length** — depth of the selected solution

Statistics are accessible via:

```python
method.get_stat_dict()  # Returns dict of statistics
```

And stored in the `TreeThinkOutputs.graph_stats` field for post-hoc analysis.

---

## Graph Analysis

The `treethink graph analyze` command computes aggregate statistics across
multiple experiment runs:

```bash
treethink graph analyze /path/to/experiment/outputs/ --output summary.json
```

This collects all `.txt` graph files in the directory and produces a summary
JSON with mean, median, min, max for each metric.

---

## File Format

Tree `.txt` files use a simplified DOT format:

```
graph
{
    "node_0x1234" [label="...", color="blue"];
    "node_0x5678" [label="...", color="red"];
    "node_0x1234" -- "node_0x5678";
}
```

- Node IDs are the Python `id()` of the `Node` object.
- `color="blue"` marks the root.
- `color="red"` marks the selected solution path.

For the full API, refer to the source at `src/treethink/graph.py`.
