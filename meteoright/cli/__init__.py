"""Command-line interface for MeteoRight.

Each subcommand lives in its own module; `main` parses arguments and
dispatches to it.
"""

from .main import main

__all__ = ["main"]
