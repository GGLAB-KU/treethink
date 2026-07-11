"""Configuration parsing and model setup helpers for the CLI."""

from typing import Tuple, Union

import yaml
from loguru import logger

from treethink import (
    EvaluatorArgs,
    PolicyArgs,
    TreeThinkArgs,
)
from treethink.sampler import AsyncTreeThinkSampler, TreeThinkSampler
from treethink.utils.parser import (
    parse_normal_inference_args,
    parse_treethink_args,
)


def _async_method_name(sync_name: str) -> str:
    """Convert a sync method name to its async variant.

    Convention: prepend ``\"Async\"``, e.g. ``RFMCTS`` → ``AsyncRFMCTS``.
    """
    return f"Async{sync_name}"


def _async_func_name(sync_name: str) -> str:
    """Convert a sync policy/evaluator func_name to its async variant.

    Convention: prepend ``\"async_\"``, e.g. ``vllm_policy`` → ``async_vllm_policy``.
    """
    return f"async_{sync_name}"


def convert_to_async(
    treethink_args: TreeThinkArgs,
    policy_args: PolicyArgs,
    evaluator_args: EvaluatorArgs,
) -> Tuple[TreeThinkArgs, PolicyArgs, EvaluatorArgs]:
    """Mutate args in-place to async variants when ``--async`` is set.

    Uses a simple prefix convention:
    * ``method_name`` → ``\"Async\" + method_name``
    * ``policy.func_name`` → ``\"async_\" + func_name``
    * ``evaluator.func_name`` → ``\"async_\" + func_name``

    Args:
        treethink_args: TreeThink configuration (mutated in place).
        policy_args: Policy configuration (mutated in place).
        evaluator_args: Evaluator configuration (mutated in place).

    Returns:
        The same three objects, now with async-compatible names.
    """
    async_method = _async_method_name(treethink_args.method_name)
    logger.info(
        f"Converting method: {treethink_args.method_name} → {async_method}"
    )
    treethink_args.method_name = async_method

    async_policy = _async_func_name(policy_args.func_name)
    logger.info(f"Converting policy: {policy_args.func_name} → {async_policy}")
    policy_args.func_name = async_policy

    async_evaluator = _async_func_name(evaluator_args.func_name)
    logger.info(
        f"Converting evaluator: {evaluator_args.func_name} → {async_evaluator}"
    )
    evaluator_args.func_name = async_evaluator

    return treethink_args, policy_args, evaluator_args


# -- Config parsing -------------------------------------------------------


def parse_inference_arguments(
    gen_config_path: str,
) -> Tuple[Union[Tuple, None], str]:
    """Parse YAML generation config and detect inference type.

    Args:
        gen_config_path: Path to the YAML config file.

    Returns:
        A tuple ``(args, inference_type)`` where:
        * ``args`` is the result of ``parse_treethink_args`` or
          ``parse_normal_inference_args``.
        * ``inference_type`` is ``"treethink"`` or ``"normal"``.
    """
    inference_type = None
    with open(gen_config_path, "r") as f:
        gen_config = yaml.safe_load(f)
        if "treethink" in gen_config:
            inference_type = "treethink"
        else:
            inference_type = "normal"

    _args = None
    if inference_type == "treethink":
        _args = parse_treethink_args(gen_config_path)
    else:
        _args = parse_normal_inference_args(gen_config_path)

    return _args, inference_type


# -- Prompter helpers -----------------------------------------------------


def simple_messages_to_string(messages):
    """Convert a message list to a single concatenated string."""
    result = ""
    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        if content:
            result += content
    return result


# -- Model setup ----------------------------------------------------------


def setup_model(
    gen_config_path: str,
    run_name: str,
    use_async: bool = False,
    max_concurrent: int = 4,
) -> Union[TreeThinkSampler, AsyncTreeThinkSampler]:
    """Initialize the model with given parameters.

    Args:
        gen_config_path: Path to the YAML generation config.
        run_name: Name identifier for the run.
        use_async: Whether to use the async stack.
        max_concurrent: Max concurrent datapoints for async mode.

    Returns:
        A :class:`TreeThinkSampler` (sync) or :class:`AsyncTreeThinkSampler` (async).

    Raises:
        RuntimeError: If inference type is unknown.
    """
    _args, _inference_type = parse_inference_arguments(gen_config_path)

    if _inference_type == "treethink":
        treethink_args, policy_args, evaluator_args = _args

        if use_async:
            # Auto-convert method/policy/evaluator names to async variants
            treethink_args, policy_args, evaluator_args = convert_to_async(
                treethink_args, policy_args, evaluator_args
            )

            logger.info("Using pure async stack (AsyncTreeThinkSampler)")
            model = AsyncTreeThinkSampler(
                policy_args=policy_args,
                evaluator_args=evaluator_args,
                treethink_args=treethink_args,
                prompter=simple_messages_to_string,
                max_concurrent_datapoints=max_concurrent,
                task_name=run_name,
            )
        else:
            logger.info("Using sequential TreeThink sampler (TreeThinkSampler)")
            model = TreeThinkSampler(
                policy_args=policy_args,
                evaluator_args=evaluator_args,
                treethink_args=treethink_args,
                sample_params=None,
                prompter=simple_messages_to_string,
                task_name=run_name,
            )
    else:
        raise RuntimeError(
            "Normal inference config detected. "
            "Use 'treethink run' with a TreeThink config "
            "or use the standalone VLLMSampler script (examples/run_vllm_sampler.py)."
        )

    logger.success("Model initialization completed!")
    return model
