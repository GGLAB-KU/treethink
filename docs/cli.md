# CLI Reference — `treethink`

The CLI is built with **Typer**.  Entry point defined in `pyproject.toml`:

```toml
[project.scripts]
treethink = "treethink.cli.main:main"
```

---

## Global Usage

```bash
treethink [OPTIONS] COMMAND [ARGS]...
```

---

## Commands

### `treethink run`

Run tree-search inference (MCTS / BFTS / BeamSearch).

```bash
treethink run --config path/to/config.yaml [OPTIONS]
```

**Options:**

| Option | Description |
|--------|-------------|
| `--config`, `-c` | Path to YAML config file (required) |
| `--data-config-path` | Path to dataset TOML config file |
| `--data-config-name` | Dataset config name in the TOML file |
| `--async` | Enable async mode (auto-converts components) |
| `--overwrite`, `-o` | Overwrite existing output |
| `-n` | Limit number of problems to process |
| `--verbose`, `-v` | Verbose logging |

**Source:** `src/treethink/cli/commands/run.py`

---

### `treethink help`

Show comprehensive documentation about TreeThink components.

```bash
treethink help [OPTIONS] [TOPIC]
```

**Arguments:**

| Argument | Description |
|----------|-------------|
| `topic` | Topic to show. If omitted, shows the overview. |

**Options:**

| Option | Description |
|--------|-------------|
| `--list`, `-l` | List all available help topics |

**Available topics:**

| Topic | Description |
|-------|-------------|
| `overview` | High-level architecture |
| `methods` | Tree search strategies |
| `policies` | Child generation via LLM |
| `evaluators` | Node scoring |
| `termination` | Proof verification |
| `clients` | REPL proof assistant backends |
| `config` | YAML/TOML configuration reference |
| `datasets` | Dataset preparation and config |
| `async` | Async execution mode |
| `sampler` | Batch processing |
| `cli` | CLI commands reference |
| `graph` | Tree visualization & statistics |
| `extending` | Extending the library |

**Source:** `src/treethink/cli/commands/help_cmd.py`

---

### `treethink graph`

Interact with saved tree graphs.

```bash
treethink graph COMMAND [ARGS]...
```

**Subcommands:**

| Command | Description |
|---------|-------------|
| `visualize` | Render a tree graph from a graphviz `.txt` file to PNG/SVG |
| `extract` | Extract solutions from graphviz `.txt` files |
| `analyze` | Compute aggregate graph statistics across experiment runs |
| `info` | Inspect a saved tree — show structure and metadata |

**Source:** `src/treethink/cli/commands/graph.py`

#### `treethink graph visualize`

```bash
treethink graph visualize <input_file> [--output OUTPUT] [--format FORMAT]
```

#### `treethink graph extract`

```bash
treethink graph extract <input_file> [--output OUTPUT]
```

#### `treethink graph analyze`

```bash
treethink graph analyze <directory> [--output OUTPUT]
```

#### `treethink graph info`

```bash
treethink graph info <input_file>
```

---

## Entry Point

```python
# treethink.cli.main:main
def main():
    """Entry point for the ``treethink`` CLI."""
    app()
```

For the full command implementations, refer to `src/treethink/cli/commands/`.
