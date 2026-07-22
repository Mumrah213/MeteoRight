#!/usr/bin/env python3
"""Compatibility shim for the MeteoRight CLI.

The implementation lives in ``meteoright.cli``; this module keeps the
historical ``cli:main`` entry point and ``python cli.py`` invocation working.
"""

from __future__ import annotations

import sys

from meteoright.cli import main

__all__ = ["main"]


if __name__ == "__main__":
    sys.exit(main())
