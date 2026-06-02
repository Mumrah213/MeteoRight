"""Progress reporting for the dataset preparation pipeline.

Uses rich.console for styled progress output with:
- Section headers (rule blocks)
- Step status indicators (green ✓, yellow ⚠, red ✗)
- Row counts and summaries
- Completion banners

Usage
-----
>>> from datasets.progress import ProgressReporter

>>> reporter = ProgressReporter()
>>> reporter.section("Downloading data")
>>> reporter.step("forecasts", f"Downloaded 12345 rows")
>>> reporter.step("observations", f"Downloaded 17520 rows")
>>> reporter.complete("Data downloaded")
"""

from __future__ import annotations

import logging
from pathlib import Path

from rich.console import Console

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Progress reporter
# ---------------------------------------------------------------------------

class ProgressReporter:
    """Console-based progress reporter for the dataset pipeline.

    Provides structured, styled output for pipeline progress tracking.
    All methods return self for chaining.
    """

    def __init__(self, verbose: bool = False) -> None:
        """Initialize the progress reporter.

        Args:
            verbose: If True, also prints to logger.debug.
        """
        self.console = Console()
        self.verbose = verbose

    def section(self, title: str) -> "ProgressReporter":
        """Print a section header.

        Args:
            title: Section title.

        Returns:
            self for chaining.
        """
        self.console.rule(f"[blue]{title}[/blue]")
        logger.info("=== %s ===", title)
        return self

    def step(self, name: str, detail: str = "", icon: str = "•") -> "ProgressReporter":
        """Print a step status line.

        Args:
            name: Step identifier (e.g., 'forecasts', 'observations').
            detail: Status detail (e.g., 'Downloaded 12345 rows').
            icon: Unicode icon prefix (default: '•').

        Returns:
            self for chaining.
        """
        line = f"  {icon} {name}"
        if detail:
            line += f": {detail}"
        console_msg = f"[dim]{line}[/dim]"
        self.console.print(console_msg)
        logger.info(line)
        return self

    def success(self, name: str, detail: str = "") -> "ProgressReporter":
        """Print a success step with green checkmark.

        Args:
            name: Step name.
            detail: Status detail.

        Returns:
            self for chaining.
        """
        line = f"  ✓ {name}"
        if detail:
            line += f": {detail}"
        self.console.print(f"[green]{line}[/green]")
        logger.info(line)
        return self

    def warning(self, name: str, detail: str = "") -> "ProgressReporter":
        """Print a warning step with yellow indicator.

        Args:
            name: Step name.
            detail: Status detail.

        Returns:
            self for chaining.
        """
        line = f"  ⚠ {name}"
        if detail:
            line += f": {detail}"
        self.console.print(f"[yellow]{line}[/yellow]")
        logger.warning(line)
        return self

    def error(self, name: str, detail: str = "") -> "ProgressReporter":
        """Print an error step with red indicator.

        Args:
            name: Step name.
            detail: Error detail.

        Returns:
            self for chaining.
        """
        line = f"  ✗ {name}"
        if detail:
            line += f": {detail}"
        self.console.print(f"[red]{line}[/red]")
        logger.error(line)
        return self

    def complete(self, title: str = "Complete") -> "ProgressReporter":
        """Print a completion banner.

        Args:
            title: Completion message.

        Returns:
            self for chaining.
        """
        self.console.print(f"[bold green]✓ {title}[/bold green]")
        return self

    def summary(self, title: str, **kwargs: Any) -> "ProgressReporter":
        """Print a summary table with key-value pairs.

        Args:
            title: Summary title.
            **kwargs: Key-value pairs to display.

        Returns:
            self for chaining.
        """
        self.console.print()
        self.console.print(f"[bold]{title}:[/bold]")
        for key, value in kwargs.items():
            if isinstance(value, (int, float)):
                display = f"{value:,}"
            else:
                display = str(value)
            self.console.print(f"  {key:.<20s} {display}")
        return self

    def info(self, message: str) -> "ProgressReporter":
        """Print an informational message.

        Args:
            message: Message text.

        Returns:
            self for chaining.
        """
        self.console.print(f"[dim]i {message}[/dim]")
        logger.info(message)
        return self


# Type hint for kwargs in summary
from typing import Any
