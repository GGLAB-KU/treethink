"""
TreeThink CLI — tree-search inference for formal mathematical reasoning.

Usage:

  treethink run ...        Run tree-search inference
  treethink graph ...      Interact with saved tree graphs
  treethink help [topic]   Show comprehensive documentation
"""

import typer

from treethink.cli.commands.graph import analyze, extract, info, visualize
from treethink.cli.commands.help_cmd import show_help
from treethink.cli.commands.run import run

app = typer.Typer(
    name="treethink",
    help="Tree-search inference for formal mathematical reasoning.",
    no_args_is_help=True,
    add_completion=False,
    rich_markup_mode="rich",
)

# ── Register subcommands ────────────────────────────────────────────────

app.command(
    name="run",
    help="Run tree-search inference (AlphaZeroMCTS / TraditionalMCTS / BFTS / BeamSearch).",
    rich_help_panel="Commands",
)(run)

graph_app = typer.Typer(
    name="graph",
    help="Interact with saved tree graphs (visualize, extract, analyze, info).",
    no_args_is_help=True,
    rich_markup_mode="rich",
)

graph_app.command(
    name="visualize",
    help="Render a tree graph from a graphviz .txt file to PNG/SVG.",
    rich_help_panel="Graph Commands",
)(visualize)

graph_app.command(
    name="extract",
    help="Extract solutions from graphviz .txt files.",
    rich_help_panel="Graph Commands",
)(extract)

graph_app.command(
    name="analyze",
    help="Compute aggregate graph statistics across experiment runs.",
    rich_help_panel="Graph Commands",
)(analyze)

graph_app.command(
    name="info",
    help="Inspect a saved tree: show structure and metadata.",
    rich_help_panel="Graph Commands",
)(info)

app.add_typer(
    graph_app,
    name="graph",
    help="Interact with saved tree graphs.",
)

app.command(
    name="help",
    help="Show comprehensive documentation about components.",
    rich_help_panel="Commands",
)(show_help)


def main():
    """Entry point for the ``treethink`` CLI."""
    app()


if __name__ == "__main__":
    main()
