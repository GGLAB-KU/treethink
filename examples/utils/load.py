import json
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import datasets
import pandas as pd
from loguru import logger


def load_dataset(
    path: Union[str, Path],
    format_type: Optional[str] = None,
    hf_config: Optional[str] = None,
    hf_split: str = "train",
    csv_delimiter: str = ",",
    **kwargs,
) -> Any:
    """
    Load datasets from various formats.

    Args:
        path: Path to dataset file or HuggingFace dataset name
        format_type: Format type ('huggingface', 'json', 'jsonl', 'csv', 'parquet', 'pickle', 'excel')
                    If None, will try to infer from file extension
        hf_config: Configuration name for HuggingFace datasets (optional)
        hf_split: Split to load for HuggingFace datasets (default: 'train')
        csv_delimiter: Delimiter for CSV files (default: ',')
        **kwargs: Additional arguments passed to specific loaders

    Returns:
        Loaded dataset (format depends on the loader used)
    """

    path = Path(path) if isinstance(path, str) else path

    # Auto-detect format if not specified
    if format_type is None:
        format_type = _infer_format(path)
        logger.debug(f"Inferred dataset type: {format_type}")

    format_type = format_type.lower()

    try:
        if format_type == "huggingface" or format_type == "hf":
            return _load_huggingface(str(path), hf_config, hf_split, **kwargs)

        elif format_type == "huggingface_disk":
            return _load_huggingface_disk(
                str(path), hf_config, hf_split, **kwargs
            )

        elif format_type == "json":
            return _load_json(path, **kwargs)

        elif format_type == "jsonl" or format_type == "ndjson":
            return _load_jsonl(path, **kwargs)

        elif format_type == "csv":
            return _load_csv(path, csv_delimiter, **kwargs)

        elif format_type == "parquet":
            return _load_parquet(path, **kwargs)

        elif format_type == "pickle" or format_type == "pkl":
            return _load_pickle(path, **kwargs)

        elif format_type in ["excel", "xlsx", "xls"]:
            return _load_excel(path, **kwargs)

        elif format_type == "tsv":
            return _load_csv(path, "\t", **kwargs)

        else:
            raise ValueError(f"Unsupported format: {format_type}")

    except Exception as e:
        logger.error(f"Failed to load dataset from {path}: {str(e)}")
        raise


def _infer_format(path: Path) -> str:
    """Infer format from file extension."""
    if path.exists() and path.is_dir():
        # Assume it's a HuggingFace dataset name
        return "huggingface_disk"

    if str(path).find("/") != -1:
        return "huggingface"

    suffix = path.suffix.lower()

    format_map = {
        ".json": "json",
        ".jsonl": "jsonl",
        ".ndjson": "jsonl",
        ".csv": "csv",
        ".tsv": "tsv",
        ".parquet": "parquet",
        ".pkl": "pickle",
        ".pickle": "pickle",
        ".xlsx": "excel",
        ".xls": "excel",
    }

    return format_map.get(suffix, "json")  # Default to json


def _load_huggingface(
    dataset_name: str, config: Optional[str], split: str, **kwargs
) -> Any:
    """Load HuggingFace dataset."""
    logger.info(f"Loading HuggingFace dataset: {dataset_name}")

    load_kwargs = {"path": dataset_name, "split": split}
    if config:
        load_kwargs["name"] = config

    # Add any additional kwargs
    load_kwargs.update(kwargs)

    return datasets.load_dataset(**load_kwargs)


def _load_huggingface_disk(
    dataset_path: str, config: Optional[str], split: str, **kwargs
) -> Any:
    """Load HuggingFace dataset."""
    logger.info(f"Loading HuggingFace dataset from disk: {dataset_path}")

    load_kwargs = {"dataset_path": dataset_path}
    if config:
        load_kwargs["name"] = config

    # Add any additional kwargs
    load_kwargs.update(kwargs)

    return (
        datasets.load_from_disk(**load_kwargs)[split]
        if split
        else datasets.load_from_disk(**load_kwargs)
    )


def _load_json(path: Path, **kwargs) -> Union[Dict, List]:
    """Load JSON file."""
    logger.info(f"Loading JSON file: {path}")

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f, **kwargs)


def _load_jsonl(path: Path, **kwargs) -> List[Dict]:
    """Load JSONL file."""
    logger.info(f"Loading JSONL file: {path}")

    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line.strip(), **kwargs))

    return data


def _load_csv(path: Path, delimiter: str = ",", **kwargs) -> pd.DataFrame:
    """Load CSV file."""
    logger.info(f"Loading CSV file: {path}")

    return pd.read_csv(path, delimiter=delimiter, **kwargs)


def _load_parquet(path: Path, **kwargs) -> pd.DataFrame:
    """Load Parquet file."""
    logger.info(f"Loading Parquet file: {path}")

    return pd.read_parquet(path, **kwargs)


def _load_pickle(path: Path, **kwargs) -> Any:
    """Load Pickle file."""
    logger.info(f"Loading Pickle file: {path}")

    with open(path, "rb") as f:
        return pickle.load(f, **kwargs)


def _load_excel(path: Path, **kwargs) -> pd.DataFrame:
    """Load Excel file."""
    logger.info(f"Loading Excel file: {path}")

    return pd.read_excel(path, **kwargs)


# Convenience functions for specific formats
def load_hf_dataset(
    dataset_name: str,
    config: Optional[str] = None,
    split: str = "train",
    **kwargs,
):
    """Convenience function to load HuggingFace datasets."""
    return load_dataset(
        dataset_name, "huggingface", hf_config=config, hf_split=split, **kwargs
    )


def load_json_dataset(path: Union[str, Path], **kwargs):
    """Convenience function to load JSON datasets."""
    return load_dataset(path, "json", **kwargs)


def load_jsonl_dataset(path: Union[str, Path], **kwargs):
    """Convenience function to load JSONL datasets."""
    return load_dataset(path, "jsonl", **kwargs)
