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

# -- Async auto-conversion maps -------------------------------------------

ASYNC_METHOD_MAP = {
    "MCTS": "AsyncMCTS",
    "BFTS": "AsyncBFTS",
    "BeamSearch": "AsyncBeamSearch",
}

ASYNC_POLICY_MAP = {
    "vllm_policy": "async_vllm_policy",
    "vllm_server_policy": "async_vllm_server_policy",
}

ASYNC_EVALUATOR_MAP = {
    "cumulative_logprob_evaluator": "async_cumulative_logprob_evaluator",
    "lean_repl_evaluator": "async_lean_repl_evaluator",
    "llm_as_judge_evaluator": "async_judge_evaluator",
    "norm_len_evaluator": "async_norm_len_evaluator",
    "rocq_evaluator": "async_rocq_evaluator",
}


def convert_to_async(
    treethink_args: TreeThinkArgs,
    policy_args: PolicyArgs,
    evaluator_args: EvaluatorArgs,
) -> Tuple[TreeThinkArgs, PolicyArgs, EvaluatorArgs]:
    """Mutate args in-place to async variants when ``--async`` is set.

    Converts:
    * ``method_name`` → ``AsyncMCTS`` / ``AsyncBFTS`` / ``AsyncBeamSearch``
    * ``policy.func_name`` → async policy equivalent
    * ``evaluator.func_name`` → async evaluator equivalent

    Args:
        treethink_args: TreeThink configuration (mutated in place).
        policy_args: Policy configuration (mutated in place).
        evaluator_args: Evaluator configuration (mutated in place).

    Returns:
        The same three objects, now with async-compatible names.
    """
    if treethink_args.method_name in ASYNC_METHOD_MAP:
        new_name = ASYNC_METHOD_MAP[treethink_args.method_name]
        logger.info(
            f"Converting method: {treethink_args.method_name} → {new_name}"
        )
        treethink_args.method_name = new_name

    if policy_args.func_name in ASYNC_POLICY_MAP:
        new_name = ASYNC_POLICY_MAP[policy_args.func_name]
        logger.info(f"Converting policy: {policy_args.func_name} → {new_name}")
        policy_args.func_name = new_name

    if evaluator_args.func_name in ASYNC_EVALUATOR_MAP:
        new_name = ASYNC_EVALUATOR_MAP[evaluator_args.func_name]
        logger.info(
            f"Converting evaluator: {evaluator_args.func_name} → {new_name}"
        )
        evaluator_args.func_name = new_name

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
