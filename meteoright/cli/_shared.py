"""Console and logger shared by every command module."""

from __future__ import annotations

import logging

from rich.console import Console

console = Console()
logger = logging.getLogger("weather_analyzer")
