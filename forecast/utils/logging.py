"""Structured JSON logging for the weather data platform.

Every request is logged as structured JSON including:
- provider, endpoint, parameters
- response size, timing
- validation outcome, null classifications

Usage:
    >>> from ..utils.logging import setup_logging
    >>> setup_logging(level="DEBUG", log_file="logs/platform.log")
"""


import json
import logging
import sys
import time
import traceback
from collections.abc import Generator
from contextlib import contextmanager
from datetime import UTC, datetime
from typing import Any


class StructuredFormatter(logging.Formatter):
    """JSON structured log formatter."""

    def format(self, record: logging.LogRecord) -> str:
        log_entry: dict[str, Any] = {
            "timestamp": datetime.now(UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        # Add exception info if present
        if record.exc_info and record.exc_info[0] is not None:
            log_entry["exception"] = {
                "type": record.exc_info[0].__name__,
                "message": str(record.exc_info[1]),
                "traceback": traceback.format_exception(*record.exc_info),
            }

        # Add any extra fields that were attached
        for key, value in record.__dict__.items():
            if key in (
                "exc_info",
                "stack_info",
                "args",
                "lineno",
                "filename",
                "funcName",
                "created",
                "module",
                "pathname",
                "msg",
            ):
                continue
            log_entry[key] = value

        return json.dumps(log_entry, default=str)


def setup_logging(
    level: str = "INFO",
    log_file: str | None = None,
    console: bool = True,
) -> None:
    """Configure structured logging for the platform.

    Args:
        level: Logging level string (DEBUG, INFO, WARNING, ERROR).
        log_file: Optional file path for file output.
        console: Whether to also log to console.
    """
    root = logging.getLogger()
    root.setLevel(getattr(logging, level.upper(), logging.INFO))

    formatter = StructuredFormatter()

    # File handler
    if log_file:
        import os

        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        root.addHandler(file_handler)

    # Console handler
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        root.addHandler(console_handler)


@contextmanager
def timed_request(
    provider: str,
    endpoint: str,
    params: dict[str, Any],
) -> Generator[None, None, None]:
    """Context manager that times and logs a request.

    Usage:
        with timed_request("openmeteo", "/v1/forecast", params):
            response = client.get(path, params)
            logger.info("validation=passed")

    Any exception in the block is logged with full context.
    """
    start = time.monotonic()
    context = {
        "provider": provider,
        "endpoint": endpoint,
        "params": params,
    }

    try:
        yield
        elapsed_ms = (time.monotonic() - start) * 1000
        logging.getLogger("meteo").info(
            f"request completed {elapsed_ms:.0f}ms",
            extra={**context, "elapsed_ms": elapsed_ms},
        )
    except Exception as exc:
        elapsed_ms = (time.monotonic() - start) * 1000
        logging.getLogger("meteo").error(
            f"request failed after {elapsed_ms:.0f}ms: {exc}",
            extra={**context, "elapsed_ms": elapsed_ms, "error": str(exc)},
        )
        raise
