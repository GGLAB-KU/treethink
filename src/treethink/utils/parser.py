"""YAML config parsing utilities for TreeThink and normal inference configurations."""

from dataclasses import fields, is_dataclass
from enum import Enum
from functools import partial
from typing import (
    Any,
    Dict,
    Tuple,
    Type,
    TypeVar,
    get_args,
    get_origin,
    get_type_hints,
)

import yaml
from loguru import logger

from treethink.utils.args import (
    EvaluatorArgs,
    ModelArgs,
    PolicyArgs,
    SamplingArgs,
    TreeThinkArgs,
)
from treethink.utils.enums import coerce_enum

T = TypeVar("T")


def drop_none(value):
    """Recursively remove None values from dicts and lists."""
    if isinstance(value, dict):
        return {k: drop_none(v) for k, v in value.items() if v is not None}
    if isinstance(value, list):
        return [drop_none(v) for v in value if v is not None]
    return value


def serialize_args(obj):
    """Serialize a dataclass instance (or nested structure) into plain dicts."""
    if obj is None:
        return None
    if is_dataclass(obj):
        return drop_none(
            {
                dataclass_field.name: serialize_args(
                    getattr(obj, dataclass_field.name)
                )
                for dataclass_field in fields(obj)
            }
        )
    if isinstance(obj, dict):
        return drop_none({k: serialize_args(v) for k, v in obj.items()})
    if isinstance(obj, Enum):
        return obj.value
    if isinstance(obj, tuple):
        return [serialize_args(v) for v in obj]
    if isinstance(obj, list):
        return [serialize_args(v) for v in obj]
    if hasattr(obj, "__dict__"):
        return drop_none(
            {k: serialize_args(v) for k, v in obj.__dict__.items()}
        )
    return obj


def _parse_yaml_file(
    config_path: str, section_class: Dict[str, Any]
) -> Tuple[Any]:
    """Parse the yaml file and compare it with the default arguments.

    Args:
        config_path: path to the config file.
        section_class: a dictionary holding the section names in
            the yaml file as keys, and corresponding classes as values.

    Returns:
        Tuple of parsed dataclass instances.
    """
    logger.debug(f"YAML config path: {config_path}")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    logger.debug(f"Loaded YAML config: {config}")

    args = ()
    for section, class_name in section_class.items():
        args += (
            _parse_config_to_dataclass(
                config.get(section, {}),
                class_name,
            ),
        )

    return args


def _parse_config_to_dataclass(config_dict: dict, dataclass_type: Type[T]) -> T:
    """Parse a config dictionary into a dataclass, handling nested dataclasses.

    Args:
        config_dict: Dictionary with configuration values
        dataclass_type: The dataclass type to create

    Returns:
        Instance of the dataclass
    """
    logger.trace(f"Parsing config for {dataclass_type.__name__}:")
    logger.trace(f"Input config: {config_dict}")

    if not is_dataclass(dataclass_type):
        raise ValueError(f"{dataclass_type} is not a dataclass")

    type_hints = get_type_hints(dataclass_type)
    parsed_args = {}

    for field_name, field_value in config_dict.items():
        if field_name not in type_hints:
            logger.warning(
                f"Unknown field '{field_name}' in config for {dataclass_type.__name__}"
            )
            continue

        expected_type = type_hints[field_name]

        parsed_args[field_name] = _coerce_config_value(
            expected_type, field_value
        )

    result = dataclass_type(**parsed_args)
    logger.debug(f"Resulting dataclass: {result}")
    return result


def _is_optional_type(type_hint) -> bool:
    """Check if a type hint represents an Optional type (Union[X, None])."""
    origin = get_origin(type_hint)
    if origin is not None:
        from typing import Union

        if origin is Union:
            args = get_args(type_hint)
            return len(args) == 2 and type(None) in args
    return False


def _get_optional_inner_type(type_hint):
    """Get the non-None type from an Optional type hint."""
    if _is_optional_type(type_hint):
        args = get_args(type_hint)
        return args[0] if args[1] is type(None) else args[1]
    return type_hint


def _is_enum_type(type_hint) -> bool:
    return isinstance(type_hint, type) and issubclass(type_hint, Enum)


def _coerce_config_value(expected_type, field_value):
    if _is_optional_type(expected_type):
        inner_type = _get_optional_inner_type(expected_type)
        if field_value is None:
            return None
        return _coerce_config_value(inner_type, field_value)

    if is_dataclass(expected_type) and isinstance(field_value, dict):
        return _parse_config_to_dataclass(field_value, expected_type)

    if _is_enum_type(expected_type):
        return coerce_enum(field_value, expected_type)

    if expected_type is float and isinstance(field_value, (str, int)):
        return float(field_value)

    if expected_type is int and isinstance(field_value, str):
        return int(field_value)

    return field_value


# -- Inference Time config parsing --
__treethink = {
    "treethink": TreeThinkArgs,
    "policy": PolicyArgs,
    "evaluator": EvaluatorArgs,
}

parse_treethink_args = partial(_parse_yaml_file, section_class=__treethink)

# -- Normal Inference config parsing --
__normal_inference = {
    "model": ModelArgs,
    "sampling": SamplingArgs,
}

parse_normal_inference_args = partial(
    _parse_yaml_file, section_class=__normal_inference
)


__all__ = [
    "T",
    "drop_none",
    "serialize_args",
    "parse_treethink_args",
    "parse_normal_inference_args",
]
