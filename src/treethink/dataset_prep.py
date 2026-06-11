"""Dataset preparation utilities for TreeThink.

Provides :class:`PreparationConfig` and :class:`ConfigRegistry` for managing
dataset preprocessing configurations, and :func:`prepare_datapoints` for
loading and transforming datasets into the format expected by the sampler.
"""

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import toml
import yaml
from loguru import logger

from treethink.utils.load import load_dataset


@dataclass
class PreparationConfig:
    """Configuration for a specific preprocessing setup.

    Args:
        name: Name of the configuration.
        path_or_name: Path or HuggingFace name of the dataset.
            Also supported: jsonl, json.
        dataset_split: Dataset split for HuggingFace datasets.
        prompt_format: Prompt template to pass into batched_inference.
            Note: if *renamed_data_keys* is specified, then *prompt_format*
            should include keywords from *renamed_data_keys*, not *data_keys*.
        system_prompt: System prompt prepended to every request.
        data_keys: Data column names in the dataset.
        format_type: Format type hint (e.g. ``"jsonl"``, ``"huggingface"``).
            If ``None``, inferred from file extension.
        download_dir: Directory to cache downloaded datasets.
        renamed_data_keys: Optional mapping of *data_keys* to new column names
            in the output. Must be the same length as *data_keys*.
        custom_params: Additional parameters passed to the dataset loader.
    """

    name: str
    path_or_name: str
    dataset_split: str
    prompt_format: str
    system_prompt: str
    data_keys: List[str]
    format_type: str = None
    download_dir: Optional[str] = None
    renamed_data_keys: Optional[List[str]] = None
    custom_params: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate configuration after initialization."""
        if not self.name:
            raise ValueError("Configuration name cannot be empty")
        if self.renamed_data_keys:
            if len(self.renamed_data_keys) != len(self.data_keys):
                raise ValueError(
                    "renamed_data_keys and data_keys should be the same length."
                )

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary for serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PreparationConfig":
        """Create config from dictionary."""
        return cls(**data)

    def save_yaml(self, filepath: Union[str, Path]) -> None:
        """Save configuration to YAML file."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        with open(filepath, "w", encoding="utf-8") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, indent=2)

        logger.info(f"Saved config '{self.name}' to {filepath}")

    @classmethod
    def load_yaml(cls, filepath: Union[str, Path]) -> "PreparationConfig":
        """Load configuration from YAML file."""
        with open(filepath, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        config = cls.from_dict(data)
        logger.info(f"Loaded config '{config.name}' from {filepath}")
        return config

    def save_toml(self, filepath: Union[str, Path]) -> None:
        """Save configuration to TOML file."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        with open(filepath, "w", encoding="utf-8") as f:
            toml.dump(self.to_dict(), f)

        logger.info(f"Saved config '{self.name}' to {filepath}")

    @classmethod
    def load_toml(cls, filepath: Union[str, Path]) -> "PreparationConfig":
        """Load configuration from TOML file."""
        with open(filepath, "r", encoding="utf-8") as f:
            data = toml.load(f)

        config = cls.from_dict(data)
        logger.info(f"Loaded config '{config.name}' from {filepath}")
        return config


class ConfigRegistry:
    """Registry for managing preprocessing configurations."""

    def __init__(self):
        self._configs: Dict[str, PreparationConfig] = {}

    def register_config(self, config: PreparationConfig) -> None:
        """Register a new preprocessing configuration."""
        if config.name in self._configs:
            logger.warning(f"Overwriting existing config: {config.name}")
        self._configs[config.name] = config
        logger.info(f"Registered config: {config.name}")

    def get_config(self, name: str) -> PreparationConfig:
        """Get a configuration by name."""
        if name not in self._configs:
            raise KeyError(f"Configuration '{name}' not found")
        return self._configs[name]

    def list_configs(self) -> List[str]:
        """List all registered configuration names."""
        return list(self._configs.keys())

    def save_all_configs(
        self, filepath: Union[str, Path], format: str = "toml"
    ) -> None:
        """Save all configurations to a single file."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        all_configs = {
            "configs": {
                config_name: config.to_dict()
                for config_name, config in self._configs.items()
            }
        }

        if format.lower() == "toml":
            with open(filepath, "w", encoding="utf-8") as f:
                toml.dump(all_configs, f)
        elif format.lower() == "yaml":
            with open(filepath, "w", encoding="utf-8") as f:
                yaml.dump(all_configs, f, default_flow_style=False, indent=2)
        elif format.lower() == "json":
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(all_configs, f, indent=2, ensure_ascii=False)
        else:
            raise ValueError("Format must be 'toml', 'yaml', or 'json'")

        logger.info(f"Saved {len(self._configs)} configurations to {filepath}")

    def load_all_configs(
        self, filepath: Union[str, Path]
    ) -> List[PreparationConfig]:
        """Load all configurations from a single file."""
        filepath = Path(filepath)

        if filepath.suffix.lower() == ".toml":
            with open(filepath, "r", encoding="utf-8") as f:
                data = toml.load(f)
        elif filepath.suffix.lower() in [".yaml", ".yml"]:
            with open(filepath, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
        elif filepath.suffix.lower() == ".json":
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
        else:
            raise ValueError(f"Unsupported file format: {filepath.suffix}")

        loaded_configs = []
        configs_data = data.get("configs", {})

        for config_name, config_dict in configs_data.items():
            config = PreparationConfig.from_dict(config_dict)
            self.register_config(config)
            loaded_configs.append(config)

        logger.info(
            f"Loaded {len(loaded_configs)} configurations from {filepath}"
        )
        return loaded_configs

    def get_config_from_file(
        self, filepath: Union[str, Path], config_name: str
    ) -> PreparationConfig:
        """Load a specific configuration from a multi-config file without registering all."""
        filepath = Path(filepath)

        if filepath.suffix.lower() == ".toml":
            with open(filepath, "r", encoding="utf-8") as f:
                data = toml.load(f)
        elif filepath.suffix.lower() in [".yaml", ".yml"]:
            with open(filepath, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
        elif filepath.suffix.lower() == ".json":
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
        else:
            raise ValueError(f"Unsupported file format: {filepath.suffix}")

        configs_data = data.get("configs", {})
        if config_name not in configs_data:
            raise KeyError(
                f"Configuration '{config_name}' not found in {filepath}"
            )

        config = PreparationConfig.from_dict(configs_data[config_name])
        logger.info(f"Loaded config '{config_name}' from {filepath}")
        return config

    def register_config_from_file(
        self, filepath: Union[str, Path], config_name: Optional[str] = None
    ) -> PreparationConfig:
        """Load configuration from file.

        If *config_name* is provided, loads from a multi-config file.
        Otherwise, loads a single config file.
        """
        if config_name:
            config = self.get_config_from_file(filepath, config_name)
            self.register_config(config)
            return config
        else:
            filepath = Path(filepath)

            if filepath.suffix.lower() == ".toml":
                config = PreparationConfig.load_toml(filepath)
            elif filepath.suffix.lower() in [".yaml", ".yml"]:
                config = PreparationConfig.load_yaml(filepath)
            elif filepath.suffix.lower() == ".json":
                config = PreparationConfig.load_json(filepath)
            else:
                raise ValueError(f"Unsupported file format: {filepath.suffix}")

            self.register_config(config)
            return config

    def load_configs_from_directory(
        self, directory: Union[str, Path]
    ) -> List[PreparationConfig]:
        """Load all configuration files from a directory."""
        directory = Path(directory)
        if not directory.exists():
            raise FileNotFoundError(f"Directory not found: {directory}")

        loaded_configs = []

        for pattern in ["*.toml", "*.yaml", "*.yml", "*.json"]:
            for filepath in directory.glob(pattern):
                try:
                    config = self.register_config_from_file(filepath)
                    loaded_configs.append(config)
                except Exception as e:
                    logger.warning(f"Failed to load {filepath}: {e}")

        logger.info(
            f"Loaded {len(loaded_configs)} configurations from {directory}"
        )
        return loaded_configs


def prepare_datapoints(config: PreparationConfig):
    """Load and transform a dataset according to *config*.

    Returns a list of dicts where each dict's keys are *data_keys*
    (or *renamed_data_keys* if provided).  Handles multiple dataset
    formats (jsonl, huggingface, csv, etc.).
    """
    logger.info(f"Processing dataset with config: {config.name}")
    logger.info(f"Dataset split: {config.dataset_split}")
    logger.info(f"Prompt template: {config.prompt_format}")

    ds = load_dataset(
        path=config.path_or_name,
        hf_split=config.dataset_split,
        format_type=config.format_type,
        **config.custom_params,
    )

    # If ds is already in the shape we want, just return it
    if isinstance(ds, list) and isinstance(ds[0], dict):
        return ds

    datapoints = []

    for data in ds:
        single_datapoint = {}

        for i, key in enumerate(config.data_keys):
            if config.renamed_data_keys:
                single_datapoint[config.renamed_data_keys[i]] = data[key]
            else:
                single_datapoint[key] = data[key]

        datapoints.append(single_datapoint)

    return datapoints
