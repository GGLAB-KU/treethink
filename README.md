<div align="center">

# treethink

<h3>Formal Mathematical Reasoning of LLMs with Tree Search Methods</h3>

<p>
    <a href="https://img.shields.io/badge/python-3.12+-blue.svg"><img alt="Python" src="https://img.shields.io/badge/python-3.12+-blue.svg"></a>
    <a href="https://img.shields.io/badge/code%20style-ruff-000000.svg"><img alt="Ruff" src="https://img.shields.io/badge/code%20style-ruff-000000.svg"></a>
    <a href="https://img.shields.io/github/license/GGLAB-KU/treethink"><img alt="License" src="https://img.shields.io/github/license/GGLAB-KU/treethink.svg?color=red"></a>
</p>

**TreeThink** is a Python library for mathematical reasoning with LLMs
using tree-search methods. It enables LLMs to explore multiple proof paths in
parallel, verify candidates against formal proof assistants (Lean 4, Rocq,
Isabelle), and select the most promising solutions.

</div>

---

## Table of Contents

- [What is TreeThink?](#what-is-treethink)
- [Core Concepts](#core-concepts)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Documentation](#documentation)
- [Extending TreeThink](#extending-treethink)
- [Project Structure](#project-structure)
- [License](#license)

---

## What is TreeThink?

TreeThink wraps an LLM-powered generation loop inside a tree search. Starting
from a problem statement (root node), the system repeatedly:

1. **Expands** promising nodes by asking the LLM to generate candidate next
   proof steps (via **Policies**)
2. **Scores** each new node to estimate its quality (via **Evaluators**)
3. **Selects** the best node(s) for further expansion based on the search
   strategy (via **Methods**)
4. **Terminates** branches when a complete proof is found or a branch is
   exhausted (via **Termination** — REPL-based proof verification)

### Supported Proof Assistants

| Language | Client | Status |
|----------|--------|--------|
| **Lean 4** | Kimina server | Stable (sync + async) |
| **Rocq** (Coq 8.20) | `rocq-ml-server` | Stable (sync) |
| **Isabelle** | - | Not yet implemented |

---

## Core Concepts

### Methods — Search Strategies

Methods determine **how the search tree is explored**.  Each method inherits
from `BaseMethod` and implements a `simulate()` loop.

| Method | Description |
|--------|-------------|
| **MCTS** | Monte Carlo Tree Search with UCB1 selection (`exploration_weight`) |
| **BFTS** | Breadth-First Tree Search — level-by-level exploration |
| **BeamSearch** | Beam Search — maintains a fixed-size beam of top-k nodes |

All methods are available in sync and async variants (auto-converted via
`--async`).

[Detailed docs](docs/methods.md) · `treethink help methods`

### Policies — Child Generation

Policies wrap an LLM (via vLLM) to produce candidate next proof steps.

| Policy | Description |
|--------|-------------|
| `vllm_policy` | Standard local vLLM model — supports LoRA adapters |
| `dynamic_policy` | Like vllm_policy, with dynamically adjustable sampling params |
| `vllm_server_policy` | External vLLM server via OpenAI-compatible API |

[Detailed docs](docs/policies.md) · `treethink help policies`

### Evaluators — Node Scoring

Evaluators assign scores to guide the search toward promising branches.

| Evaluator | Description |
|-----------|-------------|
| `cumulative_logprob_evaluator` | Cumulative log-probability from the LLM |
| `repl_evaluator` | REPL verification via formal language servers (binary feedback) |
| `llm_as_judge_evaluator` | Secondary LLM judge scores proof quality |
| `pairwise_tournament_evaluator` | Single-elimination tournament between siblings |
| `normalized_lengths_evaluator` | BFS-Prover: logprob / L^alpha |
| `rocq_evaluator` | Rocq proof verification via `rocq-ml-server` |

[Detailed docs](docs/evaluators.md) · `treethink help evaluators`

### Termination — Proof Verification

Two-phase verification system:

1. **On encounter** — verifies a proof immediately when a termination node is
   found during search (fast path)
2. **On paths** — batch-verifies all termination leaves after search completes
   (fallback)

[Detailed docs](docs/termination.md) · `treethink help termination`

### Async Mode

Fully asynchronous tree search for maximum throughput.  Auto-converts methods,
policies, and evaluators to async equivalents when `--async` is passed.

➡️ [Detailed docs](docs/async.md) · `treethink help async`

---

## Installation

### Prerequisites

- Python 3.12+
- [uv](https://docs.astral.sh/uv/) (package manager)

### Install from PyPI (when published)

```bash
# Core only (vLLM + base utilities)
uv pip install treethink

# With Lean 4 support
uv pip install treethink[lean]

# With Rocq (Coq) support
uv pip install treethink[rocq]

# With Isabelle support
uv pip install treethink[isabelle]

# Everything (all languages + dev tools)
uv pip install treethink[full]
```

### Development Setup

```bash
# Clone the repository
git clone https://github.com/GGLAB-KU/treethink.git
cd treethink

# Create a virtual environment and install everything for development
uv sync --all-extras

# Verify the CLI works
uv run treethink help --list
```

To install only a subset of extras during development:

```bash
# Core + Lean only
uv sync --extra lean

# Core + Rocq only
uv sync --extra rocq
```

### What's in each extra?

| Extra | Includes | Purpose |
|-------|----------|---------|
| *(core)* | `vllm`, `loguru`, `datasets`, `typer`, … | Always installed — policies, CLI, logging |
| `lean` | `kimina-client` | Lean 4 proof verification (via Kimina server) |
| `rocq` | `rocq-ml-toolbox`, `pytanque` | Rocq (Coq 8.20) proof verification (via `rocq-ml-server`) |
| `isabelle` | *(empty — placeholder)* | Isabelle support (not yet implemented) |
| `dev` | `pytest` | Development & testing tools |
| `full` | all of the above | Everything |

### Runtime Requirements

- **Lean 4:** A running Kimina Lean server (typically `http://localhost:8000`).
- **Rocq:** A running `rocq-ml-server` on `localhost:5000`.
- **Isabelle:** Not yet available.

---

## Quick Start

### 1. Create a YAML configuration

```yaml
# config.yaml
treethink:
  method_name: "MCTS"
  max_children: 4
  expansion_count: 64
  timeout: 120
  termination_str: "```"
  language: "lean4"
  repl_args:
    lean_server_url: "http://localhost:8000"

policy:
  func_name: "vllm_policy"
  model:
    model: "internlm/internlm2-7b"
    tensor_parallel_size: 1
  sampling:
    max_tokens: 2048
    temperature: 1.0
    n: 4
    stop: ["\n"]

evaluator:
  func_name: "cumulative_logprob_evaluator"
```

### 2. Run tree-search inference

```bash
# Sync mode
uv run treethink run --config config.yaml

# Async mode (auto-converts components)
uv run treethink run --config config.yaml --async

# With a dataset configuration
uv run treethink run \
  --config config.yaml \
  --data-config-path configs/dataset_configs.toml \
  --data-config-name my_experiment
```

### 3. Visualise results

```bash
# Inspect a saved tree
uv run treethink graph info output/tree_20240607_120000.txt

# Render to PNG
uv run treethink graph visualize output/tree_20240607_120000.txt \
  --output tree.png --format png
```

---

## Documentation

All documentation is available via the CLI and as markdown files in `docs/`.

### CLI Help

```bash
# List all topics
treethink help --list

# Read a specific topic
treethink help methods
treethink help evaluators
treethink help termination
treethink help config
```

### docs/ Directory

| Document | Description |
|----------|-------------|
| [`docs/overview.md`](docs/overview.md) | High-level architecture and component wiring |
| [`docs/methods.md`](docs/methods.md) | Tree search strategies (MCTS, BFTS, BeamSearch) |
| [`docs/policies.md`](docs/policies.md) | Child generation via LLM |
| [`docs/evaluators.md`](docs/evaluators.md) | Node scoring |
| [`docs/termination.md`](docs/termination.md) | Proof verification (two-phase) |
| [`docs/clients.md`](docs/clients.md) | REPL proof assistant backends |
| [`docs/config.md`](docs/config.md) | YAML/TOML configuration reference |
| [`docs/datasets.md`](docs/datasets.md) | Dataset preparation and configuration |
| [`docs/async.md`](docs/async.md) | Async execution mode |
| [`docs/sampler.md`](docs/sampler.md) | Batch processing |
| [`docs/cli.md`](docs/cli.md) | CLI commands reference |
| [`docs/graph.md`](docs/graph.md) | Tree visualisation and statistics |
| [`docs/extending.md`](docs/extending.md) | Extending the library |

---

## Extending TreeThink

TreeThink is designed to be easily extended.  Here is where to look:

| Component | Base class | Register in | Source |
|-----------|------------|-------------|--------|
| **Method** | `BaseMethod` | `IMPLEMENTED_METHODS` dict | `src/treethink/methods/__init__.py` |
| **Policy** | `BasePolicy` | `PolicyType` enum | `src/treethink/policies.py` |
| **Evaluator** | `BaseEvaluator` | `EvaluatorType` enum | `src/treethink/evaluators.py` |
| **REPL Client** | `ProofAssistantClient` | `FormalLanguage` enum + `client_factory.py` | `src/treethink/clients/base.py` |

For detailed instructions, see [`docs/extending.md`](docs/extending.md) or run
`treethink help extending`.

---

## Project Structure

```
treethink/
├── src/treethink/
│   ├── cli/              # CLI commands (run, help, graph)
│   │   ├── commands/     # Command implementations
│   │   └── helpers/      # Config parsing, iteration logic
│   ├── clients/          # REPL proof assistant clients
│   │   ├── lean/         # Lean 4 (Kimina)
│   │   ├── coq/          # Rocq (rocq-ml-server)
│   │   └── isabelle/     # Isabelle (not yet implemented)
│   ├── methods/          # Tree search algorithms
│   │   ├── mcts.py       # Monte Carlo Tree Search
│   │   ├── bfts.py       # Breadth-First Tree Search
│   │   └── beam.py       # Beam Search
│   ├── evaluators.py     # Node scoring
│   ├── policies.py       # Child generation
│   ├── termination.py    # Proof verification
│   ├── treethink.py      # Main orchestrator
│   └── sampler.py        # Batch processing
├── docs/                 # Documentation markdown files
├── examples/             # Example scripts and configs
├── tests/                # pytest test suite
├── pyproject.toml        # Project metadata and dependencies
└── ruff.toml             # Linter/formatter configuration
```

---

## License

This project is licensed under the terms of the [LICENSE](LICENSE) file.


## Citation
If you use TreeThink in your research, please consider citing:
