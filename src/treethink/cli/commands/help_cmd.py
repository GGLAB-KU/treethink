"""``treethink help`` — documentation loaded from ``docs/*.md``.

All help content lives in ``docs/`` markdown files at the project root.
Topic names map directly to filenames (e.g. ``treethink help config`` →
``docs/config.md``).
"""

from pathlib import Path
from typing import Optional

import typer

# ── Resolve docs/ directory relative to this file ──────────────────────
#   src/treethink/cli/commands/help_cmd.py  →  repo root
_REPO_ROOT = Path(__file__).resolve().parents[4]
_DOCS_DIR = _REPO_ROOT / "docs"

# ── Helpers ────────────────────────────────────────────────────────────


def _discover_topics() -> dict[str, Path]:
    """Return ``{topic_name: path}`` for every ``.md`` file in ``docs/``."""
    if not _DOCS_DIR.is_dir():
        return {}
    return {p.stem: p for p in sorted(_DOCS_DIR.glob("*.md"))}


def _load_topic(topic: str) -> str | None:
    """Read ``docs/{topic}.md`` and return its content, or ``None``."""
    path = _DOCS_DIR / f"{topic}.md"
    if not path.is_file():
        return None
    return path.read_text(encoding="utf-8")


def _topic_title(content: str) -> str:
    """Extract the first top-level heading from markdown content."""
    for line in content.strip().splitlines():
        line = line.strip()
        if line.startswith("# ") and not line.startswith("##"):
            return line.lstrip("# ").strip()
    return ""


# ── CLI command ────────────────────────────────────────────────────────


def show_help(
    ctx: typer.Context,
    topic: Optional[str] = typer.Argument(
        None,
        help="Topic to show help for. Use --list to see available topics. "
        "Defaults to 'overview'.",
    ),
    list_topics: bool = typer.Option(
        False,
        "--list",
        "-l",
        help="List all available help topics.",
    ),
):
    """Show comprehensive documentation about TreeThink components.

    Content is loaded from markdown files in the ``docs/`` directory.

    Use ``treethink help <topic>`` to get detailed information about
    methods, policies, evaluators, termination, configuration, datasets,
    or async mode::

        treethink help methods
        treethink help evaluators
        treethink help termination
        treethink help --list
    """
    topics = _discover_topics()

    # ── --list mode ─────────────────────────────────────────────────
    if list_topics:
        print()
        print("  " + "\u2550" * 72)
        print("    Available help topics")
        print("  " + "\u2550" * 72)
        print()
        print("    Core Components:")
        for name in ("overview", "methods", "policies", "evaluators"):
            if name in topics:
                print(f"      \u2022 {name}")
        print()
        print("    Proof Verification:")
        for name in ("termination", "clients"):
            if name in topics:
                print(f"      \u2022 {name}")
        print()
        print("    Configuration & Data:")
        for name in ("config", "datasets"):
            if name in topics:
                print(f"      \u2022 {name}")
        print()
        print("    Execution:")
        for name in ("async", "sampler", "cli", "graph"):
            if name in topics:
                print(f"      \u2022 {name}")
        print()
        print("    Extending:")
        for name in ("extending",):
            if name in topics:
                print(f"      \u2022 {name}")
        print()
        print("  Use:  treethink help <topic>")
        print()
        return

    # ── Resolve topic ───────────────────────────────────────────────
    if topic is None:
        topic = "overview"

    # Try exact match first
    content = _load_topic(topic)

    # Fallback: case-insensitive match
    if content is None:
        matches = [k for k in topics if k.lower() == topic.lower()]
        if matches:
            content = _load_topic(matches[0])

    # If still nothing found, show error
    if content is None:
        available = ", ".join(sorted(topics.keys()))
        print(f"Unknown topic: '{topic}'")
        print(f"Available topics: {available}")
        print("Use 'treethink help --list' for a grouped listing.")
        raise typer.Exit(code=1)

    # ── Render ──────────────────────────────────────────────────────
    title = _topic_title(content) or topic.capitalize()
    separator = "\u2550" * 72

    print()
    print(separator)
    print(f"  {title}")
    print(separator)
    print()
    print(content.strip())
    print()
