"""Different datasets and tasks require different loading and processing
pipelines. This file aims to separate different dataset processing pipelines
into various functions that can be used with `prompters.py` and `sampler.py`
"""

import argparse
import asyncio
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import yaml
from async_sampler import AsyncSampler
from dataset_prep import (
    ConfigRegistry,
    prepare_datapoints,
)
from loguru import logger
from parallel_sampler import AsyncDatapointSampler
from sampler import TreeThinkSampler, VLLMSampler
from utils.parser import (
    parse_inftime_conf,
    parse_normal_inference_conf,
    serialize_args,
)

from treethink.graph import analyze_graph_stats


def get_time():
    """Get current timestamp in formatted string."""
    _t = time.localtime()
    _d = datetime.now().date()
    return _d.strftime("%d-%m-%Y") + time.strftime("-%H:%M:%S", _t)


def parse_inference_arguments(gen_config_path: str):
    """Try to parse both types of the inference and select the one without problems."""

    inference_type = None
    with open(gen_config_path, "r") as f:
        gen_config = yaml.safe_load(f)
        if "inference_time" in gen_config.keys():
            inference_type = "inftime"
        else:
            inference_type = "normal"

    _args = None

    if inference_type == "inftime":
        _args = parse_inftime_conf(gen_config_path)
    else:
        _args = parse_normal_inference_conf(gen_config_path)

    return _args, inference_type


def simple_messages_to_string(messages):
    result = ""
    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        if content:
            result += content

        # if role == "system":
        #     result += f"<|start_header_id|>system<|end_header_id|>\n\n{content}<|eot_id|>"
        # elif role == "user":
        #     result += f"<|start_header_id|>user<|end_header_id|>\n\n{content}<|eot_id|>"
        # elif role == "assistant":
        #     result += f"<|start_header_id|>assistant<|end_header_id|>\n\n{content}<|eot_id|>"

    return result


def _infer_graph_dir(
    inference_time_args, output_path: Path, iteration_index: int, num_iterations
):
    base_graph_dir = output_path
    if inference_time_args and inference_time_args.graph_path:
        graph_path = Path(inference_time_args.graph_path)
        if graph_path.suffix:
            base_graph_dir = graph_path.parent
        else:
            base_graph_dir = graph_path / "dev"

    if num_iterations > 1 and base_graph_dir == output_path:
        candidate = output_path / f"graphs_{iteration_index}"
        if candidate.exists() and any(candidate.glob("*.txt")):
            return candidate
    return base_graph_dir


def _build_graph_stats_payload(
    model,
    results,
    output_path: Path,
    run_name: str,
    iteration_index: int,
    timestamp: str,
    num_iterations: int,
):
    if not results:
        return None

    graph_stats_list = [r["graph_stats"] for r in results if "graph_stats" in r]
    if not graph_stats_list:
        return None

    inference_time_args = getattr(model, "inference_time_args", None)
    expander_args = getattr(model, "expander_args", None)
    evaluator_args = getattr(model, "evaluator_args", None)
    method_name = (
        inference_time_args.method_name
        if inference_time_args is not None
        else None
    )

    graph_dir = _infer_graph_dir(
        inference_time_args, output_path, iteration_index, num_iterations
    )

    return {
        "run_name": run_name,
        "iteration": iteration_index,
        "timestamp": timestamp,
        "output_dir": str(output_path),
        "graph_dir": str(graph_dir),
        "graph_path": getattr(inference_time_args, "graph_path", None)
        if inference_time_args
        else None,
        "method_name": method_name,
        "inference_time_args": serialize_args(inference_time_args),
        "expander_args": serialize_args(expander_args),
        "evaluator_args": serialize_args(evaluator_args),
        "graph_stats_summary": analyze_graph_stats(graph_stats_list),
        "graph_stats_count": len(graph_stats_list),
    }


def setup_model(
    gen_config_path,
    run_name,
    use_parallel=False,
    use_async=False,
    max_concurrent=4,
):
    """Initialize the model with given parameters."""
    _args, _inference_type = parse_inference_arguments(gen_config_path)
    if _inference_type == "inftime":
        inference_time_args, expander_args, evaluator_args = _args

        if use_async:
            # Pure async stack: AsyncMCTS + AsyncChildExpander + AsyncNodeEvaluator
            logger.info("Using pure async stack (AsyncSampler)")
            model = AsyncSampler(
                expander_args=expander_args,
                evaluator_args=evaluator_args,
                inference_time_args=inference_time_args,
                prompter=simple_messages_to_string,
                task_name=run_name,
                max_concurrent_datapoints=max_concurrent,
            )
        elif use_parallel:
            # Hybrid stack: Sync MCTS with async REPL parallelization
            logger.info(
                "Using parallel datapoint sampler (AsyncDatapointSampler)"
            )
            model = AsyncDatapointSampler(
                expander_args=expander_args,
                evaluator_args=evaluator_args,
                inference_time_args=inference_time_args,
                prompter=simple_messages_to_string,
                task_name=run_name,
                max_concurrent_datapoints=max_concurrent,
            )
        else:
            # Sequential processing
            logger.info("Using sequential TreeThink sampler (TreeThinkSampler)")
            model = TreeThinkSampler(
                expander_args=expander_args,
                evaluator_args=evaluator_args,
                inference_time_args=inference_time_args,
                sample_params=None,
                prompter=simple_messages_to_string,
                task_name=run_name,
            )
    elif _inference_type == "normal":
        model_args, sample_args = _args

        logger.info("Using standard vLLM sampler (VLLMSampler)")
        model = VLLMSampler(
            model_args=model_args,
            sample_params=sample_args,
            prompter=None,
            task_name=run_name,
        )
    else:
        raise RuntimeError("Failed to initialize model.")

    logger.success("Model initialization completed!")
    return model


async def run_async_iterations(
    model,
    datapoints,
    output_dir,
    num_iterations,
    run_name,
    system_prompt,
    data_key,
    prompt_format,
    skip_existing,
    save_graph_stats,
):
    """Async helper to run iterations within a single event loop."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    logger.info(f"Starting inference loop for {num_iterations} iterations.")
    logger.info(
        f"Using async processing with {model.max_concurrent_datapoints} concurrent datapoints."
    )

    for i in range(num_iterations):
        logger.info(f"Running iteration {i + 1}/{num_iterations}")

        if isinstance(model, AsyncSampler):
            # Pure async stack: AsyncMCTS + AsyncChildExpander + AsyncNodeEvaluator
            logger.info("Running with AsyncSampler (pure async stack)")
            results = await model.async_inference(
                data=datapoints,
                system_prompt=system_prompt,
                data_key=data_key,
                prompt_format=prompt_format,
                skip_if_exists=skip_existing,
                show_progress=True,
            )
        elif isinstance(model, AsyncDatapointSampler):
            # Hybrid stack: Sync MCTS with async REPL parallelization
            logger.info("Running with AsyncDatapointSampler (parallel REPL)")
            results = await model.async_inference(
                data=datapoints,
                system_prompt=system_prompt,
                data_key=data_key,
                prompt_format=prompt_format,
                show_progress=True,
                skip_if_exists=skip_existing,
            )

        _time = get_time()
        _path = output_path / f"{i}_answers_{run_name}_{_time}.json"

        with open(_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)

        # No need to move graphs if only one iteration
        if num_iterations > 1:
            _graph_path = output_path / f"graphs_{i}/"
            _graph_path.mkdir(parents=True, exist_ok=True)
            for _graph in Path(output_path).glob("*.txt"):
                _graph.rename(_graph_path / _graph.name)
            logger.info(f"Moving graphs to {_graph_path}")

        graph_stats_payload = None
        if save_graph_stats:
            graph_stats_payload = _build_graph_stats_payload(
                model=model,
                results=results,
                output_path=output_path,
                run_name=run_name,
                iteration_index=i,
                timestamp=_time,
                num_iterations=num_iterations,
            )
            if graph_stats_payload:
                _graph_analysis_pretty = json.dumps(
                    graph_stats_payload["graph_stats_summary"], indent=2
                )
                logger.success(
                    f"Graph stats analysis for iteration {i + 1}/{num_iterations}: "
                    + f"\n{_graph_analysis_pretty}"
                )
            else:
                logger.warning(
                    "Failed to find `graph_stats` key in outputs, "
                    "did you set `store_graph_stats=True` in InferenceTimeArgs?"
                )

        if graph_stats_payload:
            stats_path = (
                output_path / f"{i}_graph_stats_{run_name}_{_time}.json"
            )
            with open(stats_path, "w", encoding="utf-8") as f:
                json.dump(graph_stats_payload, f, indent=2, ensure_ascii=False)
            logger.success(f"Graph stats saved to {stats_path}.")
        logger.success(f"Answers saved to {_path}.")


def run_inference_loop(
    model,
    datapoints,
    output_dir,
    batch_size,
    num_iterations,
    run_name,
    system_prompt,
    data_key,
    prompt_format,
    lora_path,
    save_graph_stats,
    use_parallel=False,
    use_async=False,
    skip_existing=True,
):
    """Run the main inference loop and save results."""
    if use_async or use_parallel:
        # Use a single asyncio.run for async models to avoid loop issues
        asyncio.run(
            run_async_iterations(
                model=model,
                datapoints=datapoints,
                output_dir=output_dir,
                num_iterations=num_iterations,
                run_name=run_name,
                system_prompt=system_prompt,
                data_key=data_key,
                prompt_format=prompt_format,
                skip_existing=skip_existing,
                save_graph_stats=save_graph_stats,
            )
        )
    else:
        # Original sync logic for non-async models
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        logger.info(f"Starting inference loop for {num_iterations} iterations")
        logger.info(f"Batch size: {batch_size}, Output directory: {output_dir}")

        for i in range(num_iterations):
            logger.info(f"Running iteration {i + 1}/{num_iterations}")

            # Run sequential batched inference
            logger.info("Running with Sampler (sequential)")
            results = model.batched_inference(
                data=datapoints,
                batch_size=batch_size,
                system_prompt=system_prompt,
                data_key=data_key,
                prompt_format=prompt_format,
                lora_path=lora_path,
            )

            _time = get_time()
            _path = output_path / f"{i}_answers_{run_name}_{_time}.json"

            with open(_path, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2)

            # No need to move graphs if only one iteration
            if num_iterations > 1:
                _graph_path = output_path / f"graphs_{i}/"
                _graph_path.mkdir(parents=True, exist_ok=True)
                for _graph in Path(output_path).glob("*.txt"):
                    _graph.rename(_graph_path / _graph.name)
                logger.info(f"Moving graphs to {_graph_path}")

            graph_stats_payload = None
            if save_graph_stats:
                graph_stats_payload = _build_graph_stats_payload(
                    model=model,
                    results=results,
                    output_path=output_path,
                    run_name=run_name,
                    iteration_index=i,
                    timestamp=_time,
                    num_iterations=num_iterations,
                )
                if graph_stats_payload:
                    _graph_analysis_pretty = json.dumps(
                        graph_stats_payload["graph_stats_summary"],
                        indent=2,
                    )
                    logger.success(
                        f"Graph stats analysis for iteration {i + 1}/{num_iterations}: "
                        + f"\n{_graph_analysis_pretty}"
                    )
                else:
                    logger.warning(
                        "Failed to find `graph_stats` key in outputs, "
                        "did you set `store_graph_stats=True` in InferenceTimeArgs?"
                    )

            if graph_stats_payload:
                stats_path = (
                    output_path / f"{i}_graph_stats_{run_name}_{_time}.json"
                )
                with open(stats_path, "w", encoding="utf-8") as f:
                    json.dump(
                        graph_stats_payload, f, indent=2, ensure_ascii=False
                    )
                logger.success(f"Graph stats saved to {stats_path}.")
            logger.success(f"Answers saved to {_path}")


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run normal inference or with inference time scaling techniques."
    )

    parser.add_argument(
        "--data-config-path",
        type=str,
        required=True,
        help="Path to the preparation config.",
    )
    parser.add_argument(
        "--data-config-name",
        type=str,
        required=True,
        help="Name of the configuration to use.",
    )
    parser.add_argument(
        "--gen-config-path",
        type=str,
        required=False,
        help="Generation config. It can either be a normal model + sampling "
        + "configuration or one for inference time scaling methods used.",
    )

    # Inference arguments
    parser.add_argument(
        "--batch-size", type=int, default=4096, help="Batch size for inference"
    )
    parser.add_argument(
        "--num-iterations",
        type=int,
        default=1,
        help="Number of inference iterations. Set 1 to get single outputs",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=str,
        default="outputs",
        help="Directory to save output files",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default="run",
        help="Name identifier for this run",
    )
    parser.add_argument(
        "--lora-path",
        type=str,
        help="Path to LoRARequest",
    )
    parser.add_argument(
        "--no-save-graph-stats",
        action="store_true",
        help="Do NOT calculate avg&mean of graph related statistics for each solution."
        "In order to see these, set `store_graph_stats=True` in InferenceTimeArgs.",
    )
    parser.add_argument(
        "--continue-from-prev",
        action="store_true",
        help="If set will look at the number of .txt files (assuming they are "
        "graph outputs) in the given folder and skip that amount in the dataset.",
    )
    parser.add_argument(
        "--skip-data-num",
        default=0,
        type=int,
        help="Skip n data in the dataset, possibly in order to continue a "
        "failed inference attempt. Also see `--continue-from-prev`.",
    )

    # Logging arguments
    parser.add_argument(
        "-v",
        "--verbosity",
        type=str,
        default="info",
        help="Logging level",
    )

    parser.add_argument(
        "-d",
        "--debug",
        action="store_true",
        help="Whether to take only 4 many examples as debugging purpose.",
    )

    # Parallel processing arguments
    parser.add_argument(
        "--parallel",
        action="store_true",
        help="Enable concurrent datapoint processing (hybrid mode). "
        "vLLM is shared (serialized), REPL is parallelized. "
        "Provides speedup for REPL-heavy workloads.",
    )
    parser.add_argument(
        "--async",
        dest="use_async",
        action="store_true",
        help="Enable pure async stack (AsyncMCTS + AsyncChildExpander + AsyncNodeEvaluator). "
        "Fully asynchronous tree search with concurrent child generation and evaluation. "
        "Recommended for maximum throughput.",
    )
    parser.add_argument(
        "--max-concurrent",
        type=int,
        default=4,
        help="Maximum number of concurrent datapoints (default: 4). "
        "Higher values speed up REPL verification but use more memory.",
    )
    parser.add_argument(
        "--no-skip-existing",
        action="store_true",
        help="Process all problems even if tree file already exists. "
        "By default, already processed problems are skipped (resume capability).",
    )

    args = parser.parse_args()

    if args.skip_data_num != 0 and args.continue_from_prev:
        logger.warning(
            f"Both skip_data_num={args.skip_data_num} and continue_from_prev="
            + "True is set. Taking continue_from_prev=True."
        )
        args.skip_data_num = 0

    return args


def main():
    """Main function to orchestrate the inference process."""
    args = parse_arguments()

    # Setup logging
    logger.remove(0)
    logger.add(sys.stderr, level=args.verbosity.upper())

    # Add file logging
    log_dir = Path(args.output_dir) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"run_{args.run_name}_{timestamp}.log"

    logger.add(
        str(log_file),
        level=args.verbosity.upper(),
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
        rotation="500 MB",  # Rotate when file reaches 500MB
        retention="10 days",  # Keep logs for 10 days
        compression="zip",  # Compress rotated logs
    )

    logger.info(f"Logging to file: {log_file}")
    logger.info(f"Run name: {args.run_name}")
    logger.info(f"Verbosity level: {args.verbosity.upper()}")

    # Load dataset
    data_registry = ConfigRegistry()
    data_config = data_registry.get_config_from_file(
        args.data_config_path, args.data_config_name
    )
    datapoints = prepare_datapoints(data_config)

    if args.continue_from_prev != 0:
        args.skip_data_num = len(list(Path(args.output_dir).glob("*.txt")))

    if args.skip_data_num != 0:
        logger.info(f"Taking datapoints[{args.skip_data_num}:]")
        datapoints = datapoints[args.skip_data_num :]

    if args.debug:
        logger.critical("Taking the first 4 datapoints for debugging...")
        datapoints = datapoints[:4]

    # Setup model
    model = setup_model(
        gen_config_path=args.gen_config_path,
        run_name=args.run_name,
        use_parallel=args.parallel,
        use_async=args.use_async,
        max_concurrent=args.max_concurrent,
    )

    # Run inference
    run_inference_loop(
        model=model,
        datapoints=datapoints,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        num_iterations=args.num_iterations,
        run_name=args.run_name,
        system_prompt=data_config.system_prompt,
        data_key=data_config.renamed_data_keys
        if data_config.renamed_data_keys
        else data_config.data_keys,
        prompt_format=data_config.prompt_format,
        lora_path=args.lora_path,
        use_parallel=args.parallel,
        use_async=args.use_async,
        skip_existing=not args.no_skip_existing,  # Skip by default, unless --no-skip-existing
        save_graph_stats=not args.no_save_graph_stats,  # Save graph stats by default, unless --no-save-graph-stats
    )
    logger.success("Inference completed successfully!")


if __name__ == "__main__":
    main()
