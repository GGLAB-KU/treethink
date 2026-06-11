"""``treethink run`` — run tree-search inference."""

from pathlib import Path
from typing import Optional

import typer
from loguru import logger

from treethink.cli.helpers.config import setup_model
from treethink.cli.helpers.iteration import run_inference_loop
from treethink.cli.helpers.logging_setup import setup_logging
from treethink.dataset_prep import ConfigRegistry, prepare_datapoints


def run(
    ctx: typer.Context,
    data_config_path: str = typer.Option(
        ...,
        "--data-config-path",
        help="Path to the dataset preparation config (TOML).",
    ),
    data_config_name: str = typer.Option(
        ...,
        "--data-config-name",
        help="Name of the dataset configuration to use.",
    ),
    gen_config_path: str = typer.Option(
        ...,
        "--gen-config-path",
        help="Path to generation config YAML (treethink + policy + evaluator sections).",
    ),
    batch_size: int = typer.Option(
        4096,
        "--batch-size",
        help="Number of datapoints per batch for sync mode. "
        "In async mode this is ignored (concurrency is controlled by --max-concurrent).",
    ),
    num_iterations: int = typer.Option(
        1,
        "--num-iterations",
        help="Number of inference iterations. Each iteration runs the full "
        "dataset through the search method. Set >1 to run repeated experiments.",
    ),
    output_dir: str = typer.Option(
        "outputs",
        "-o",
        "--output-dir",
        help="Directory to save output files (answers JSON, graphs, logs).",
    ),
    run_name: str = typer.Option(
        "run",
        "--run-name",
        help="Name identifier for this run. Used in output filenames.",
    ),
    lora_path: Optional[str] = typer.Option(
        None,
        "--lora-path",
        help="Path to a LoRA adapter (optional).",
    ),
    no_save_graph_stats: bool = typer.Option(
        False,
        "--no-save-graph-stats",
        help="Disable calculation of graph statistics (avg, median, std) "
        "across all trees. Requires ``store_graph_stats=True`` in the "
        "TreeThink config to have any effect.",
    ),
    continue_from_prev: bool = typer.Option(
        False,
        "--continue-from-prev",
        help="If set, skips datapoints whose graph files already exist in the "
        "output directory. Useful for resuming interrupted runs.",
    ),
    skip_data_num: int = typer.Option(
        0,
        "--skip-data-num",
        help="Skip N datapoints at the start of the dataset. "
        "Useful to continue a failed inference attempt. "
        "Mutually exclusive with ``--continue-from-prev``.",
    ),
    debug: bool = typer.Option(
        False,
        "-d",
        "--debug",
        help="Only process the first 4 datapoints for debugging purposes.",
    ),
    use_async: bool = typer.Option(
        False,
        "--async",
        help="Enable fully asynchronous tree search. "
        "Automatically converts method/policy/evaluator to their async variants "
        "(e.g. MCTS → AsyncMCTS, vllm_policy → async_vllm_policy). "
        "Requires compatible async components.",
    ),
    max_concurrent: int = typer.Option(
        4,
        "--max-concurrent",
        help="Maximum number of datapoints to process concurrently in async mode. "
        "Higher values increase throughput but consume more GPU memory.",
    ),
    no_skip_existing: bool = typer.Option(
        False,
        "--no-skip-existing",
        help="Process all problems even if a tree file already exists. "
        "By default, already-processed problems are skipped (resume capability).",
    ),
    verbosity: str = typer.Option(
        "info",
        "-v",
        "--verbosity",
        help="Logging level (debug, info, warning, error, critical).",
    ),
):
    """Run tree-search inference (MCTS / BFTS / BeamSearch).

    Example::

        treethink run \\
            --data-config-path configs/dataset_configs.toml \\
            --data-config-name leanworkbook_solve \\
            --gen-config-path configs/treethink.yaml \\
            -o outputs/my_run \\
            --run-name experiment_1 \\
            --async --max-concurrent 8
    """
    # ── Validate arguments ──────────────────────────────────────────────
    if skip_data_num != 0 and continue_from_prev:
        logger.warning(
            f"Both skip_data_num={skip_data_num} and continue_from_prev=True "
            "are set. Using continue_from_prev=True."
        )
        skip_data_num = 0

    # ── Setup logging ───────────────────────────────────────────────────
    setup_logging(
        output_dir=output_dir,
        run_name=run_name,
        verbosity=verbosity,
    )

    # ── Load dataset ────────────────────────────────────────────────────
    data_registry = ConfigRegistry()
    data_config = data_registry.get_config_from_file(
        data_config_path, data_config_name
    )
    datapoints = prepare_datapoints(data_config)

    if continue_from_prev:
        skip_data_num = len(list(Path(output_dir).glob("*.txt")))

    if skip_data_num > 0:
        logger.info(f"Skipping first {skip_data_num} datapoints")
        datapoints = datapoints[skip_data_num:]

    if debug:
        logger.critical("Debug mode: processing first 4 datapoints only.")
        datapoints = datapoints[:4]

    # ── Setup model ─────────────────────────────────────────────────────
    model = setup_model(
        gen_config_path=gen_config_path,
        run_name=run_name,
        use_async=use_async,
        max_concurrent=max_concurrent,
    )

    # ── Run inference ───────────────────────────────────────────────────
    run_inference_loop(
        model=model,
        datapoints=datapoints,
        output_dir=output_dir,
        batch_size=batch_size,
        num_iterations=num_iterations,
        run_name=run_name,
        system_prompt=data_config.system_prompt,
        data_key=data_config.renamed_data_keys
        if data_config.renamed_data_keys
        else data_config.data_keys,
        prompt_format=data_config.prompt_format,
        lora_path=lora_path,
        use_async=use_async,
        skip_existing=not no_skip_existing,
        save_graph_stats=not no_save_graph_stats,
    )

    logger.success("Inference completed successfully!")
