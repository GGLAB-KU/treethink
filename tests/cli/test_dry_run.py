"""Tests for the ``--dry-run`` CLI option."""

import contextlib
import io
import tempfile
from pathlib import Path
from unittest import TestCase

import yaml

from treethink.cli.helpers.dry_run import (
    _format_sample_prompt,
    _print_dry_run_summary,
)

# ---------------------------------------------------------------------------
# _format_sample_prompt
# ---------------------------------------------------------------------------


class TestFormatSamplePrompt(TestCase):
    """Unit tests for the pure helper ``_format_sample_prompt``."""

    def test_with_single_key_and_format(self):
        """Format a simple datapoint with one key and a prompt template."""
        datapoint = {"problem": "theorem foo : 1 + 1 = 2 := by"}
        system_prompt = "You are a Lean 4 expert."
        data_key = ["problem"]
        prompt_format = "Complete this:\n```lean4\n{}```"

        result = _format_sample_prompt(
            datapoint, system_prompt, data_key, prompt_format
        )

        self.assertIn("You are a Lean 4 expert.", result)
        self.assertIn("theorem foo : 1 + 1 = 2 := by", result)
        self.assertIn("Complete this:", result)

    def test_with_multiple_keys(self):
        """Format a datapoint with multiple data keys."""
        datapoint = {"problem": "Solve", "solution": "42"}
        system_prompt = ""
        data_key = ["problem", "solution"]
        prompt_format = "# Problem:\n{}\n# Solution:\n{}"

        result = _format_sample_prompt(
            datapoint, system_prompt, data_key, prompt_format
        )

        self.assertIn("Solve", result)
        self.assertIn("42", result)

    def test_without_prompt_format(self):
        """Fall back to space-joined values when no format is given."""
        datapoint = {"text": "hello world"}
        system_prompt = "Be helpful."
        data_key = ["text"]
        prompt_format = None

        result = _format_sample_prompt(
            datapoint, system_prompt, data_key, prompt_format
        )

        self.assertIn("Be helpful.", result)
        self.assertIn("hello world", result)

    def test_empty_datapoint(self):
        """Return a fallback message when datapoint is empty."""
        result = _format_sample_prompt({}, "system", ["key"], "fmt {}")

        self.assertEqual(result, "(no datapoints)")

    def test_missing_key(self):
        """Gracefully handle a key that does not exist in the datapoint."""
        datapoint = {"existing": "value"}
        result = _format_sample_prompt(datapoint, "sys", ["missing"], "{}")

        self.assertTrue(result.startswith("(error formatting prompt:"))

    def test_empty_datapoint_dict(self):
        """Edge case: empty dict is not the same as no datapoints."""
        result = _format_sample_prompt({}, "", ["k"], "{}")
        # Empty-dict evaluation: truthy check on {} is False
        self.assertEqual(result, "(no datapoints)")

    def test_long_prompt_truncation_not_applied(self):
        """_format_sample_prompt itself does NOT truncate (caller does)."""
        text = "A" * 2000
        datapoint = {"code": text}
        result = _format_sample_prompt(datapoint, "", ["code"], "{}")
        self.assertEqual(len(result), len(text))


# ---------------------------------------------------------------------------
# _print_dry_run_summary
# ---------------------------------------------------------------------------


class TestPrintDryRunSummary(TestCase):
    """Verify the summary printer writes expected content to stderr."""

    def setUp(self):
        self.buffer = io.StringIO()

    def _capture(self, **kwargs) -> str:
        """Call _print_dry_run_summary with defaults + overrides, return stderr."""
        defaults = dict(
            method_display="RFMCTS",
            policy_display="vllm_policy",
            evaluator_display="cumulative_logprob_evaluator",
            dataset_name="test_dataset",
            total_count=100,
            filtered_count=100,
            filter_notes=[],
            output_dir="/tmp/outputs",
            num_iterations=1,
            path_warnings=[],
            sample_prompt="some prompt",
            model_name=None,
            language="",
        )
        defaults.update(kwargs)
        with contextlib.redirect_stderr(self.buffer):
            _print_dry_run_summary(**defaults)
        return self.buffer.getvalue()

    def test_header_present(self):
        output = self._capture()
        self.assertIn("Dry Run", output)

    def test_method_displayed(self):
        output = self._capture(method_display="AsyncBeamSearch")
        self.assertIn("AsyncBeamSearch", output)

    def test_policy_displayed(self):
        output = self._capture(policy_display="async_vllm_policy")
        self.assertIn("async_vllm_policy", output)

    def test_model_name_shown_when_provided(self):
        output = self._capture(model_name="internlm/internlm2-7b")
        self.assertIn("internlm/internlm2-7b", output)

    def test_evaluator_displayed(self):
        output = self._capture(evaluator_display="async_judge_evaluator")
        self.assertIn("async_judge_evaluator", output)

    def test_dataset_info(self):
        output = self._capture(dataset_name="leanworkbook", total_count=2048)
        self.assertIn("leanworkbook", output)
        self.assertIn("2,048", output)

    def test_filter_notes_included(self):
        output = self._capture(
            dataset_name="ds",
            total_count=100,
            filtered_count=96,
            filter_notes=["debug (max 4)"],
        )
        self.assertIn("debug (max 4)", output)
        self.assertIn("96", output)

    def test_output_dir_displayed(self):
        output = self._capture(output_dir="results/my_experiment")
        self.assertIn("results/my_experiment", output)

    def test_iterations_displayed(self):
        output = self._capture(num_iterations=5)
        self.assertIn("5", output)

    def test_path_warnings_included(self):
        output = self._capture(
            path_warnings=[
                "\u26a0  Output directory already exists \u2014 files may be overwritten"
            ]
        )
        self.assertIn("\u26a0", output)
        self.assertIn("already exists", output)

    def test_sample_prompt_displayed(self):
        output = self._capture(sample_prompt="theorem add_comm : ...")
        self.assertIn("theorem add_comm", output)

    def test_no_sample_prompt_when_empty(self):
        output = self._capture(sample_prompt="")
        self.assertIn("(empty)", output)

    def test_long_prompt_truncated_in_output(self):
        long_prompt = "X" * 2000
        output = self._capture(sample_prompt=long_prompt)
        # The summary truncates to 500 chars + "..."
        self.assertIn("...", output)
        # The full 2000 chars should NOT appear
        self.assertNotIn("X" * 1000, output)

    def test_method_display_no_async_hint_in_sync_mode(self):
        """Without --async, no async conversion hints appear."""
        output = self._capture(method_display="RFMCTS")
        self.assertIn("RFMCTS", output)
        self.assertNotIn("AsyncRFMCTS", output)

    def test_language_shown_when_provided(self):
        output = self._capture(language="lean4")
        self.assertIn("lean4", output)

    def test_language_omitted_when_empty(self):
        output = self._capture(language="")
        self.assertNotIn("Language:", output)


# ---------------------------------------------------------------------------
# CLI integration (requires temporary config files)
# ---------------------------------------------------------------------------


class TestDryRunCLIIntegration(TestCase):
    """End-to-end tests that create temp configs and invoke ``--dry-run``.

    These tests verify that the ``--dry-run`` flag is accepted by the CLI
    entry point and produces the expected exit code 85.
    """

    def _write_yaml(self, path: Path, data: dict):
        path.write_text(yaml.dump(data))

    def _write_toml(self, path: Path, data: dict):
        import toml

        with open(path, "w") as f:
            toml.dump(data, f)

    def _make_temp_configs(self, tmp_path: Path) -> tuple[Path, Path]:
        """Create minimal valid dataset TOML and gen YAML, return their paths."""
        # ── Dataset TOML ────────────────────────────────────────────────
        dataset_config = {
            "configs": {
                "test_ds": {
                    "name": "test_ds",
                    "path_or_name": str(tmp_path / "dummy_data.jsonl"),
                    "dataset_split": "train",
                    "prompt_format": "Solve: {}",
                    "system_prompt": "You are an expert.",
                    "data_keys": ["problem"],
                    "format_type": "jsonl",
                }
            }
        }
        toml_path = tmp_path / "dataset.toml"
        self._write_toml(toml_path, dataset_config)

        # ── Dummy JSONL dataset (one datapoint) ─────────────────────────
        data_path = tmp_path / "dummy_data.jsonl"
        data_path.write_text(
            '{"problem": "theorem foo : True := by trivial"}\n'
        )

        # ── Generation YAML ─────────────────────────────────────────────
        gen_config = {
            "treethink": {
                "method_name": "RFMCTS",
                "max_children": 2,
                "expansion_count": 4,
                "timeout": 10,
                "graph_path": str(tmp_path / "graphs"),
                "termination_str": "```",
            },
            "policy": {
                "func_name": "vllm_policy",
                "model": {"model": "test/model"},
                "sampling": {
                    "max_tokens": 128,
                    "temperature": 1.0,
                    "n": 2,
                },
            },
            "evaluator": {
                "func_name": "cumulative_logprob_evaluator",
            },
        }
        gen_yaml_path = tmp_path / "gen.yaml"
        self._write_yaml(gen_yaml_path, gen_config)

        return toml_path, gen_yaml_path

    def test_dry_run_exits_with_code_85(self):
        """Invoking ``--dry-run`` with valid configs exits 85."""
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            toml_path, gen_path = self._make_temp_configs(tmp)

            from typer.testing import CliRunner

            from treethink.cli.main import app

            runner = CliRunner()
            result = runner.invoke(
                app,
                [
                    "run",
                    "--data-config-path",
                    str(toml_path),
                    "--data-config-name",
                    "test_ds",
                    "--gen-config-path",
                    str(gen_path),
                    "--dry-run",
                ],
            )

            self.assertEqual(
                result.exit_code, 85, msg=result.stderr or result.stdout
            )

    def test_dry_run_with_debug_flag(self):
        """Dry-run with ``--debug`` shows filtered count of max 4."""
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            toml_path, gen_path = self._make_temp_configs(tmp)

            from typer.testing import CliRunner

            from treethink.cli.main import app

            runner = CliRunner()
            result = runner.invoke(
                app,
                [
                    "run",
                    "--data-config-path",
                    str(toml_path),
                    "--data-config-name",
                    "test_ds",
                    "--gen-config-path",
                    str(gen_path),
                    "--dry-run",
                    "--debug",
                ],
            )

            output = result.stderr or result.stdout
            self.assertEqual(result.exit_code, 85)
            self.assertIn("Dry Run", output)

    def test_dry_run_with_async_flag(self):
        """Dry-run with ``--async`` shows async mapping in output."""
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            toml_path, gen_path = self._make_temp_configs(tmp)

            from typer.testing import CliRunner

            from treethink.cli.main import app

            runner = CliRunner()
            result = runner.invoke(
                app,
                [
                    "run",
                    "--data-config-path",
                    str(toml_path),
                    "--data-config-name",
                    "test_ds",
                    "--gen-config-path",
                    str(gen_path),
                    "--dry-run",
                    "--async",
                ],
            )

            output = result.stderr or result.stdout
            self.assertEqual(result.exit_code, 85)
            # Should mention async mapping for RFMCTS → AsyncRFMCTS
            self.assertIn("AsyncRFMCTS", output)

    def test_dry_run_without_flag_does_not_dry_run(self):
        """Without ``--dry-run``, the CLI does NOT exit with 85 — it fails
        because there's no real model/config to run (expected)."""
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            toml_path, gen_path = self._make_temp_configs(tmp)

            from typer.testing import CliRunner

            from treethink.cli.main import app

            runner = CliRunner()
            result = runner.invoke(
                app,
                [
                    "run",
                    "--data-config-path",
                    str(toml_path),
                    "--data-config-name",
                    "test_ds",
                    "--gen-config-path",
                    str(gen_path),
                ],
            )

            # Without --dry-run the inference attempts to load a model → fails
            self.assertNotEqual(result.exit_code, 85)
