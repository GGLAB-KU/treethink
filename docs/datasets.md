# Datasets — Configuration & Preparation

Dataset configurations define **how problems are loaded and formatted** before
being fed into the sampler.

---

## TOML Configuration Format

Datasets are configured in **TOML files** with the following structure:

```toml
[configs.my_experiment]
name = "my_experiment"
path_or_name = "path/to/data.jsonl"
dataset_split = "train"
prompt_format = "Solve: {}"
system_prompt = "You are a math assistant."
data_keys = ["problem"]
format_type = "jsonl"
renamed_data_keys = ["problem"]
```

### Field Reference

| Field | Required | Description |
|-------|----------|-------------|
| `name` | ✅ | Configuration name (must be unique) |
| `path_or_name` | ✅ | File path or HuggingFace dataset name |
| `dataset_split` | ✅ | Dataset split (e.g. `"train"`, `"test"`) |
| `prompt_format` | ✅ | Template with `{}` placeholders for `data_keys` values |
| `system_prompt` | ✅ | System prompt prepended to every request |
| `data_keys` | ✅ | Column names in the dataset |
| `format_type` | — | `"jsonl"`, `"huggingface"`, `"json"`, etc. (inferred from extension if omitted) |
| `download_dir` | — | Cache directory for downloaded datasets |
| `renamed_data_keys` | — | Optional mapping — must be same length as `data_keys` |
| `custom_params` | — | Additional params passed to the loader |

### Prompt Format

The `prompt_format` template is filled with values from `data_keys`.  If
`renamed_data_keys` is specified, use those names in the template instead:

```toml
data_keys = ["question", "answer"]
renamed_data_keys = ["problem", "solution"]
prompt_format = "Problem: {problem}\nSolution: {solution}"
```

---

## Supported Dataset Formats

| Format | `format_type` | Description |
|--------|---------------|-------------|
| JSONL | `"jsonl"` | One JSON object per line |
| JSON | `"json"` | Array of objects |
| HuggingFace | `"huggingface"` | HuggingFace `datasets` library |
| CSV | `"csv"` | Comma-separated values |
| TSV | `"tsv"` | Tab-separated values |
| Parquet | `"parquet"` | Apache Parquet |
| Pickle | `"pickle"` | Python pickle |
| Excel | `"excel"` | `.xlsx` / `.xls` |

If `format_type` is omitted, it is inferred from the file extension.

---

## Loading Datasets in Code

```python
from treethink.dataset_prep import PreparationConfig, ConfigRegistry, prepare_datapoints

# Load config from TOML
registry = ConfigRegistry("path/to/configs.toml")
config = registry.get("my_experiment")

# Prepare datapoints
datapoints = prepare_datapoints(config)
# Returns a list of prompts ready for the sampler
```

---

## CLI Usage

```bash
treethink run \
  --data-config-path configs/dataset_configs.toml \
  --data-config-name my_experiment \
  ...
```

---

## PreparationConfig

**Dataclass:** `PreparationConfig` in `src/treethink/dataset_prep.py`

Validates configuration on initialisation:
- `name` must not be empty.
- If `renamed_data_keys` is provided, it must be the same length as
  `data_keys`.

---

## ConfigRegistry

**Class:** `ConfigRegistry` in `src/treethink/dataset_prep.py`

Manages a collection of dataset configurations loaded from a TOML file:

```python
registry = ConfigRegistry(path)
registry.list_configs()    # List available config names
registry.get(name)         # Get a specific config
```

For the full API, refer to the source at `src/treethink/dataset_prep.py`.
