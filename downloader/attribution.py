"""Data source attribution for Open-Meteo API responses.

Open-Meteo data is licensed under CC BY 4.0. Attribution must be included
when downloading, storing, or displaying the data.

https://open-meteo.com/en/license
"""

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# Attribution strings — used in plots and metadata
ATTRIBUTION_TEXT = "Data: Open-Meteo.com (CC BY 4.0)"
ATTRIBUTION_FULL = (
    "Weather data provided by Open-Meteo.com "
    "under CC BY 4.0 license "
    "(https://open-meteo.com/en/license)"
)


def write_attribution_metadata(
    output_dir: Path,
    source: str,
    latitude: float | None = None,
    longitude: float | None = None,
    model: str | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
) -> Path:
    """Write a JSON attribution metadata file to the output directory.

    This file is read back by downstream processing to propagate
    attribution information.

    Args:
        output_dir: Root output directory.
        source: Data source name (e.g. "open-meteo-archive", "open-meteo-single-runs").
        latitude: Requested latitude.
        longitude: Requested longitude.
        model: Model name (for forecasts).
        start_date: Start date string.
        end_date: End date string.

    Returns:
        Path to the written metadata file.
    """
    from downloader.storage import _ensure_dirs

    _ensure_dirs(output_dir)
    path = output_dir / "metadata" / "attribution.json"

    record: dict[str, Any] = {
        "license": "CC BY 4.0",
        "source": source,
        "url": "https://open-meteo.com/",
        "attribution": ATTRIBUTION_FULL,
        "required": True,
    }

    if latitude is not None:
        record["latitude"] = latitude
    if longitude is not None:
        record["longitude"] = longitude
    if model is not None:
        record["model"] = model
    if start_date is not None:
        record["start_date"] = start_date
    if end_date is not None:
        record["end_date"] = end_date

    with open(path, "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2)

    logger.info("Wrote attribution metadata: %s", path)
    return path


def read_attribution_metadata(output_dir: Path) -> dict[str, Any] | None:
    """Read the attribution metadata file, or return None if missing.

    Args:
        output_dir: Root output directory.

    Returns:
        Attribution dict, or None if not found.
    """
    path = output_dir / "metadata" / "attribution.json"
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)
