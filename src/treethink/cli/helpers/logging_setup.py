"""Logging configuration for the CLI."""

import sys
from datetime import datetime
from pathlib import Path

from loguru import logger


def setup_logging(
    output_dir: str,
    run_name: str,
    verbosity: str = "INFO",
):
    """Configure loguru with both stderr and file sinks.

    Args:
        output_dir: Directory for log files.
        run_name: Name identifier for the run.
        verbosity: Logging level (e.g. ``"INFO"``, ``"DEBUG"``).
    """
    logger.remove(0)
    logger.add(sys.stderr, level=verbosity.upper())

    log_dir = Path(output_dir) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"run_{run_name}_{timestamp}.log"

    logger.add(
        str(log_file),
        level=verbosity.upper(),
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
        rotation="500 MB",
        retention="10 days",
        compression="zip",
    )

    logger.info(f"Logging to file: {log_file}")
    logger.info(f"Run name: {run_name}")
    logger.info(f"Verbosity level: {verbosity.upper()}")
