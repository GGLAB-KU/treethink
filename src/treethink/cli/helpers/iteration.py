"""Inference loop helpers for both sync and async modes."""

import asyncio
import json
import time
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from loguru import logger

from treethink import AsyncTreeThinkSampler, TreeThinkSampler

from .graph_stats import build_graph_stats_payload


def get_time() -> str:
    """Get current timestamp in formatted string."""
    _t = time.localtime()
    _d = datetime.now().date()
    return _d.strftime("%d-%m-%Y") + time.strftime("-%H:%M:%S", _t)


async def run_async_iterations(
    model: AsyncTreeThinkSampler,
    datapoints: List[dict],
    output_dir: str,
    num_iterations: int,
    run_name: str,
    system_prompt: str,
    data_key: list,
    prompt_format: Optional[str],
    skip_existing: bool,
    save_graph_stats: bool,
):
    """Async helper to run iterations within a single event loop.

    Args:
        model: The async sampler.
        datapoints: List of datapoints to process.
        output_dir: Directory for outputs.
        num_iterations: How many times to repeat the full inference.
        run_name: Name identifier.
        system_prompt: System prompt passed to every request.
        data_key: Keys to extract from each datapoint.
        prompt_format: Optional prompt format string.
        skip_existing: Whether to skip already processed problems.
        save_graph_stats: Whether to compute and save graph stats.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    logger.info(
        f"Starting async inference loop for {num_iterations} iterations."
    )
    logger.info(
        f"Using async processing with {model.max_concurrent_datapoints} "
        "concurrent datapoints."
    )

    for i in range(num_iterations):
        logger.info(f"Running iteration {i + 1}/{num_iterations}")

        results = await model.async_inference(
            data=datapoints,
            system_prompt=system_prompt,
            data_key=data_key,
            prompt_format=prompt_format,
            skip_if_exists=skip_existing,
            show_progress=True,
        )

        _time = get_time()
        _path = output_path / f"{i}_answers_{run_name}_{_time}.json"

        with open(_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)

        # Move graph files to iteration-specific folder
        if num_iterations > 1:
            _graph_path = output_path / f"graphs_{i}/"
            _graph_path.mkdir(parents=True, exist_ok=True)
            for _graph in Path(output_path).glob("*.txt"):
                _graph.rename(_graph_path / _graph.name)
            logger.info(f"Moving graphs to {_graph_path}")

        graph_stats_payload = None
        if save_graph_stats:
            graph_stats_payload = build_graph_stats_payload(
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
                    "did you set `store_graph_stats=True` in TreeThinkArgs?"
                )

        if graph_stats_payload:
            stats_path = output_path / f"{i}_exp_{run_name}_{_time}.json"
            with open(stats_path, "w", encoding="utf-8") as f:
                json.dump(graph_stats_payload, f, indent=2, ensure_ascii=False)
            logger.success(
                f"Experiment parameters and graph stats saved to {stats_path}."
            )
        logger.success(f"Answers saved to {_path}.")


def run_inference_loop(
    model: TreeThinkSampler,
    datapoints: List[dict],
    output_dir: str,
    batch_size: int,
    num_iterations: int,
    run_name: str,
    system_prompt: str,
    data_key: list,
    prompt_format: Optional[str],
    lora_path: Optional[str],
    save_graph_stats: bool,
    use_async: bool = False,
    skip_existing: bool = True,
):
    """Run the main (sync) inference loop and save results.

    Args:
        model: The (sync) sampler.
        datapoints: List of datapoints.
        output_dir: Directory for outputs.
        batch_size: Number of datapoints per batch.
        num_iterations: How many times to repeat.
        run_name: Name identifier.
        system_prompt: System prompt.
        data_key: Keys to extract.
        prompt_format: Optional format string.
        lora_path: Optional LoRA adapter path.
        save_graph_stats: Whether to compute and save graph stats.
        use_async: If ``True``, delegates to :func:`run_async_iterations`.
        skip_existing: Whether to skip already processed problems.
    """
    if use_async:
        # Delegate to the async version
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
        return

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    logger.info(f"Starting inference loop for {num_iterations} iterations")
    logger.info(f"Batch size: {batch_size}, Output directory: {output_dir}")

    for i in range(num_iterations):
        logger.info(f"Running iteration {i + 1}/{num_iterations}")

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

        # Move graph files to iteration-specific folder
        if num_iterations > 1:
            _graph_path = output_path / f"graphs_{i}/"
            _graph_path.mkdir(parents=True, exist_ok=True)
            for _graph in Path(output_path).glob("*.txt"):
                _graph.rename(_graph_path / _graph.name)
            logger.info(f"Moving graphs to {_graph_path}")

        graph_stats_payload = None
        if save_graph_stats:
            graph_stats_payload = build_graph_stats_payload(
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
                    "did you set `store_graph_stats=True` in TreeThinkArgs?"
                )

        if graph_stats_payload:
            stats_path = output_path / f"{i}_exp_stats_{run_name}_{_time}.json"
            with open(stats_path, "w", encoding="utf-8") as f:
                json.dump(graph_stats_payload, f, indent=2, ensure_ascii=False)
            logger.success(
                f"Experiment parameters and graph stats saved to {stats_path}."
            )
        logger.success(f"Answers saved to {_path}")
