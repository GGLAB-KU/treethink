from dataclasses import asdict, fields, is_dataclass
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

T = TypeVar("T")


def drop_none(value):
    if isinstance(value, dict):
        return {k: drop_none(v) for k, v in value.items() if v is not None}
    if isinstance(value, list):
        return [drop_none(v) for v in value if v is not None]
    return value


def serialize_args(obj):
    if obj is None:
        return None
    if is_dataclass(obj):
        return drop_none(asdict(obj))
    if isinstance(obj, dict):
        return drop_none(obj)
    if hasattr(obj, "__dict__"):
        return drop_none(dict(obj.__dict__))
    return obj


def _parse_yaml_file(
    config_path: str, section_class: Dict[str, Any]
) -> Tuple[Any]:
    """Parse the yaml file and compare it with the default arguments.

    Args:
        config_path (str): path to the config file.
        section_class (Dict[str, Any]): a dictionary holding the section names in
        the yaml file as keys, and corresponding classes as values.

    Returns:
        Tuple[Any]:
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

        # Handle nested dataclasses
        if is_dataclass(expected_type) and isinstance(field_value, dict):
            parsed_args[field_name] = _parse_config_to_dataclass(
                field_value, expected_type
            )
        # Handle basic type conversions
        elif expected_type is float and isinstance(field_value, (str, int)):
            parsed_args[field_name] = float(field_value)
        elif expected_type is int and isinstance(field_value, str):
            parsed_args[field_name] = int(field_value)
        # Handle Optional types (Union[X, None])
        elif _is_optional_type(expected_type):
            inner_type = _get_optional_inner_type(expected_type)
            if is_dataclass(inner_type) and isinstance(field_value, dict):
                parsed_args[field_name] = _parse_config_to_dataclass(
                    field_value, inner_type
                )
            elif inner_type is float and isinstance(field_value, (str, int)):
                parsed_args[field_name] = float(field_value)
            elif inner_type is int and isinstance(field_value, str):
                parsed_args[field_name] = int(field_value)
            else:
                parsed_args[field_name] = field_value
        else:
            parsed_args[field_name] = field_value

    # Handle fields that weren't provided in config but have defaults
    dataclass_fields = {f.name: f for f in fields(dataclass_type)}
    for field_name, field_obj in dataclass_fields.items():
        if field_name not in parsed_args:
            # Only add if field has a default value or default_factory
            if field_obj.default is not field_obj.default_factory:
                # Field has a default value, let dataclass handle it
                pass
            elif (
                field_obj.default_factory
                is not field_obj.default_factory.__class__()
            ):
                # Field has a default_factory, let dataclass handle it
                pass
            # If no default and field is optional, we can skip it

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


# Inference Time
from treethink import (  # noqa: E402
    EvaluatorArgs,
    PolicyArgs,
    TreeThinkArgs,
)

__treethink = {
    "treethink": TreeThinkArgs,
    "policy": PolicyArgs,
    "evaluator": EvaluatorArgs,
}

parse_treethink_args = partial(_parse_yaml_file, section_class=__treethink)

# Normal Inference
from treethink import ModelArgs, SamplingArgs  # noqa: E402

__normal_inference = {
    "model": ModelArgs,
    "sampling": SamplingArgs,
}
parse_normal_inference_args = partial(
    _parse_yaml_file, section_class=__normal_inference
)
