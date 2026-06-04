"""
Standalone script for running normal (non-tree-search) vLLM inference.

This is a convenience script kept for users who only need batched
vLLM generation.  For tree-search methods (MCTS, BFTS, BeamSearch),
use the ``treethink run`` CLI instead.
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import yaml
from loguru import logger

from examples.sampler import VLLMSampler
from treethink.dataset_prep import (
    ConfigRegistry,
    prepare_datapoints,
)
from treethink.utils.parser import (
    parse_normal_inference_args,
)


def get_time():
    _t = time.localtime()
    _d = datetime.now().date()
    return _d.strftime("%d-%m-%Y") + time.strftime("-%H:%M:%S", _t)


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Run batched vLLM inference (without tree search)."
    )

    parser.add_argument(
        "--data-config-path",
        type=str,
        required=True,
        help="Path to the dataset preparation config (TOML).",
    )
    parser.add_argument(
        "--data-config-name",
        type=str,
        required=True,
        help="Name of the dataset configuration to use.",
    )
    parser.add_argument(
        "--gen-config-path",
        type=str,
        required=True,
        help="Path to generation config YAML (model + sampling).",
    )

    parser.add_argument(
        "--batch-size", type=int, default=4096, help="Batch size for inference."
    )
    parser.add_argument(
        "--num-iterations",
        type=int,
        default=1,
        help="Number of inference iterations.",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=str,
        default="outputs",
        help="Directory to save output files.",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default="run",
        help="Name identifier for this run.",
    )
    parser.add_argument(
        "--lora-path",
        type=str,
        help="Path to LoRA adapter (optional).",
    )
    parser.add_argument(
        "--no-save-graph-stats",
        action="store_true",
        help="Deprecated for normal inference (no trees to analyze).",
    )
    parser.add_argument(
        "--skip-data-num",
        default=0,
        type=int,
        help="Skip N datapoints at the start of the dataset.",
    )
    parser.add_argument(
        "-v",
        "--verbosity",
        type=str,
        default="info",
        help="Logging level.",
    )
    parser.add_argument(
        "-d",
        "--debug",
        action="store_true",
        help="Only process the first 4 datapoints (for debugging).",
    )

    return parser.parse_args()


def main():
    args = parse_arguments()

    # Setup logging
    logger.remove(0)
    logger.add(sys.stderr, level=args.verbosity.upper())

    log_dir = Path(args.output_dir) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"run_{args.run_name}_{timestamp}.log"
    logger.add(
        str(log_file),
        level=args.verbosity.upper(),
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
        rotation="500 MB",
        retention="10 days",
        compression="zip",
    )
    logger.info(f"Run name: {args.run_name}")

    # Load dataset
    data_registry = ConfigRegistry()
    data_config = data_registry.get_config_from_file(
        args.data_config_path, args.data_config_name
    )
    datapoints = prepare_datapoints(data_config)

    if args.skip_data_num > 0:
        logger.info(f"Skipping first {args.skip_data_num} datapoints")
        datapoints = datapoints[args.skip_data_num :]

    if args.debug:
        logger.critical("Debug mode: processing first 4 datapoints only.")
        datapoints = datapoints[:4]

    # Parse generation config
    with open(args.gen_config_path, "r") as f:
        gen_config = yaml.safe_load(f)

    if "treethink" in gen_config:
        logger.warning(
            "Config contains 'treethink' section but this script only supports "
            "normal inference. Use 'treethink run' for tree-search methods."
        )
        # Still try to extract model+sampling if present
        model_args, sample_args = parse_normal_inference_args(
            args.gen_config_path
        )
    else:
        model_args, sample_args = parse_normal_inference_args(
            args.gen_config_path
        )

    logger.info("Using standard vLLM sampler (VLLMSampler)")
    model = VLLMSampler(
        model_args=model_args,
        sample_params=sample_args,
        prompter=None,
        task_name=args.run_name,
    )
    logger.success("Model initialization completed!")

    # Run inference loop
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    logger.info(
        f"Starting inference loop for {args.num_iterations} iterations."
    )

    results = model.batched_inference(
        data=datapoints,
        batch_size=args.batch_size,
        system_prompt=data_config.system_prompt,
        data_key=data_config.renamed_data_keys
        if data_config.renamed_data_keys
        else data_config.data_keys,
        prompt_format=data_config.prompt_format,
        lora_path=args.lora_path,
    )

    _time = get_time()
    _path = output_path / f"answers_{args.run_name}_{_time}.json"
    with open(_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    logger.success(f"Inference completed. Results saved to {_path}.")


if __name__ == "__main__":
    main()
