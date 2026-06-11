"""``--dry-run`` helper for ``treethink run``.

Parses configs, loads the dataset to count/preview datapoints,
prints a compact summary to stderr, and exits with code 85 —
without loading any model or running tree search.
"""

import sys
from pathlib import Path
from typing import List, Optional

from treethink.cli.helpers.config import (
    ASYNC_EVALUATOR_MAP,
    ASYNC_METHOD_MAP,
    ASYNC_POLICY_MAP,
    parse_inference_arguments,
    simple_messages_to_string,
)
from treethink.dataset_prep import ConfigRegistry, prepare_datapoints


def _perform_dry_run(
    data_config_path: str,
    data_config_name: str,
    gen_config_path: str,
    output_dir: str,
    run_name: str,
    skip_data_num: int = 0,
    continue_from_prev: bool = False,
    debug: bool = False,
    use_async: bool = False,
    num_iterations: int = 1,
    no_skip_existing: bool = False,
) -> None:
    """Parse configs, load/preview dataset, and print a compact dry-run summary.

    Exits with code 85.
    """
    # ── 1. Parse dataset config ────────────────────────────────────────
    data_registry = ConfigRegistry()
    data_config = data_registry.get_config_from_file(
        data_config_path, data_config_name
    )

    # ── 2. Load dataset ────────────────────────────────────────────────
    datapoints = prepare_datapoints(data_config)
    total_count = len(datapoints)

    # Compute filtered count
    filtered_count = total_count
    filter_notes: List[str] = []

    if continue_from_prev:
        existing = len(list(Path(output_dir).glob("*.txt")))
        filtered_count = max(0, filtered_count - existing)
        filter_notes.append(f"continue-from-prev (-{existing})")

    if skip_data_num > 0:
        filtered_count = max(0, filtered_count - skip_data_num)
        filter_notes.append(f"skip-data-num (-{skip_data_num})")

    if debug:
        filtered_count = min(filtered_count, 4)
        filter_notes.append("debug (max 4)")

    # Format the first datapoint as a sample prompt
    data_key = (
        data_config.renamed_data_keys
        if data_config.renamed_data_keys
        else data_config.data_keys
    )
    sample_prompt = _format_sample_prompt(
        datapoints[0] if datapoints else {},
        data_config.system_prompt,
        data_key,
        data_config.prompt_format,
    )

    # ── 3. Parse generation config ─────────────────────────────────────
    _args, inference_type = parse_inference_arguments(gen_config_path)

    if inference_type != "treethink":
        print(
            "Error: --dry-run only supports TreeThink configs "
            "(with a 'treethink:' section in the YAML).",
            file=sys.stderr,
        )
        raise SystemExit(1)

    treethink_args, policy_args, evaluator_args = _args

    # ── 4. Compute async mappings ──────────────────────────────────────
    method_name = treethink_args.method_name
    policy_name = policy_args.func_name
    evaluator_name = evaluator_args.func_name

    async_method = ASYNC_METHOD_MAP.get(method_name)
    async_policy = ASYNC_POLICY_MAP.get(policy_name)
    async_evaluator = ASYNC_EVALUATOR_MAP.get(evaluator_name)

    if use_async:
        running_method = async_method or method_name
        running_policy = async_policy or policy_name
        running_evaluator = async_evaluator or evaluator_name
        method_display = f"{method_name} \u2192 {running_method}"
        policy_display = f"{policy_name} \u2192 {running_policy}"
        evaluator_display = f"{evaluator_name} \u2192 {running_evaluator}"
    else:
        running_method = method_name
        running_policy = policy_name
        running_evaluator = evaluator_name
        if async_method:
            method_display = (
                f"{method_name}  (\u2192 {async_method} with --async)"
            )
        else:
            method_display = method_name
        if async_policy:
            policy_display = (
                f"{policy_name}  (\u2192 {async_policy} with --async)"
            )
        else:
            policy_display = policy_name
        if async_evaluator:
            evaluator_display = (
                f"{evaluator_name}  (\u2192 {async_evaluator} with --async)"
            )
        else:
            evaluator_display = evaluator_name

    # ── 5. Check file paths ─────────────────────────────────────────────
    output_path = Path(output_dir)
    graph_path_str = getattr(treethink_args, "graph_path", None)

    path_warnings: List[str] = []
    if output_path.exists():
        path_warnings.append(
            "\u26a0  Output directory already exists \u2014 files may be overwritten"
        )
    if graph_path_str:
        graph_path = Path(graph_path_str)
        if graph_path.exists() and any(graph_path.iterdir()):
            path_warnings.append(
                "\u26a0  Graph path already contains files \u2014 may be overwritten"
            )

    # ── 6. Print compact summary ────────────────────────────────────────
    model_name: Optional[str] = None
    if policy_args.model:
        model_name = policy_args.model.model

    _print_dry_run_summary(
        method_display=method_display,
        policy_display=policy_display,
        evaluator_display=evaluator_display,
        dataset_name=data_config.name,
        total_count=total_count,
        filtered_count=filtered_count,
        filter_notes=filter_notes,
        output_dir=output_dir,
        num_iterations=num_iterations,
        path_warnings=path_warnings,
        sample_prompt=sample_prompt,
        model_name=model_name,
    )

    raise SystemExit(85)


def _format_sample_prompt(
    datapoint: dict,
    system_prompt: str,
    data_key: List[str],
    prompt_format: Optional[str],
) -> str:
    """Format a single datapoint as a prompt string."""
    if not datapoint:
        return "(no datapoints)"

    try:
        values = [datapoint[k] for k in data_key]
        if prompt_format:
            content = prompt_format.format(*values)
        else:
            content = " ".join(str(v) for v in values)

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": content},
        ]
        return simple_messages_to_string(messages)
    except (KeyError, IndexError) as e:
        return f"(error formatting prompt: {e})"


def _print_dry_run_summary(
    method_display: str,
    policy_display: str,
    evaluator_display: str,
    dataset_name: str,
    total_count: int,
    filtered_count: int,
    filter_notes: List[str],
    output_dir: str,
    num_iterations: int,
    path_warnings: List[str],
    sample_prompt: str,
    model_name: Optional[str] = None,
) -> None:
    """Print the compact dry-run summary to stderr."""
    lines: List[str] = []
    lines.append("")

    # ── Header ──────────────────────────────────────────────────────────
    lines.append(
        "\u250c\u2500\u2500 Dry Run \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2510"
    )

    # ── Method ──────────────────────────────────────────────────────────
    lines.append(f"\u2502  Method:     {method_display}")

    # ── Policy ──────────────────────────────────────────────────────────
    if model_name:
        lines.append(f"\u2502  Policy:     {policy_display}")
        lines.append(f"\u2502              Model: {model_name}")
    else:
        lines.append(f"\u2502  Policy:     {policy_display}")

    # ── Evaluator ───────────────────────────────────────────────────────
    lines.append(f"\u2502  Evaluator:  {evaluator_display}")

    # ── Dataset ─────────────────────────────────────────────────────────
    if filter_notes:
        notes_str = ", ".join(filter_notes)
        lines.append(
            f"\u2502  Dataset:    {dataset_name} \u2014 "
            f"{total_count:,} points ({notes_str} \u2192 {filtered_count:,})"
        )
    else:
        lines.append(
            f"\u2502  Dataset:    {dataset_name} \u2014 {total_count:,} points"
        )

    # ── Output ──────────────────────────────────────────────────────────
    lines.append(f"\u2502  Output:     {output_dir}")
    lines.append(f"\u2502  Iterations: {num_iterations}")

    # ── Warnings ────────────────────────────────────────────────────────
    for warning in path_warnings:
        lines.append(f"\u2502  {warning}")

    # ── Sample prompt ───────────────────────────────────────────────────
    lines.append(
        "\u2502  \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500"
        "\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500"
        "\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500"
    )
    if sample_prompt:
        truncated = sample_prompt[:500]
        if len(sample_prompt) > 500:
            truncated += "..."
        display_prompt = truncated.replace("\n", "\\n")
        lines.append("\u2502  Sample prompt (truncated to 500 chars):")
        lines.append(f'\u2502  "{display_prompt}"')
    else:
        lines.append("\u2502  Sample prompt: (empty)")

    # ── Footer ──────────────────────────────────────────────────────────
    lines.append(
        "\u2514\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500"
        "\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500"
        "\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2518"
    )
    lines.append("")

    print("\n".join(lines), file=sys.stderr)
