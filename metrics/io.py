"""I/O layer for verification and metrics data.

Handles:
- Reading canonical verification parquet files (glob-aware)
- Writing aggregated metrics to parquet
- Optional CSV export

All verification data is read-only. Metrics are written as new files.
"""

import logging
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Reading verification data
# ---------------------------------------------------------------------------


def read_verification(paths: str | Path | list[str | Path]) -> pd.DataFrame:
    """Read canonical verification parquet files.

    Supports:
    - Single file path (str or Path)
    - Glob pattern (e.g. "data/verification/verification_*.parquet")
    - List of paths

    Files are concatenated and sorted by (forecast_issue_time, forecast_target_time, model).

    Args:
        paths: Single path, glob pattern, or list of paths.

    Returns:
        Concatenated DataFrame with all verification rows.

    Raises:
        FileNotFoundError: If no files match the glob pattern.
        ValueError: If paths is empty.

    Example:
        >>> df = read_verification("data/verification/verification_*.parquet")
        >>> print(len(df))
        123456
    """
    paths = _resolve_paths(paths)

    if not paths:
        raise FileNotFoundError("No verification parquet files found")

    logger.info("Reading %d verification file(s)", len(paths))

    frames: list[pd.DataFrame] = []
    for path in paths:
        logger.debug("  Reading: %s", path)
        df = pd.read_parquet(path)
        frames.append(df)
        logger.debug("    %d rows", len(df))

    result = pd.concat(frames, ignore_index=True)
    logger.info("Total: %d rows across %d files", len(result), len(paths))

    # Sort by provenance columns for consistency
    sort_cols = [
        c for c in ["forecast_issue_time", "forecast_target_time", "model"] if c in result.columns
    ]
    if sort_cols:
        result = result.sort_values(sort_cols).reset_index(drop=True)

    return result


# ---------------------------------------------------------------------------
# Writing metrics data
# ---------------------------------------------------------------------------


def write_metrics(
    df: pd.DataFrame,
    output_path: str | Path,
    metadata: dict | None = None,
) -> Path:
    """Write aggregated metrics to a parquet file.

    Includes metadata about the computation:
    - source files
    - groupby dimensions
    - metrics computed
    - variables
    - computation timestamp

    Args:
        df: Long-format metrics DataFrame.
        output_path: Output file path (.parquet).
        metadata: Optional dict of metadata to include in the parquet file.

    Returns:
        Path to the written file.

    Example:
        >>> write_metrics(
        ...     result_df,
        ...     "data/metrics/by_lead_time/temperature_2m.parquet",
        ...     metadata={"groupby": ["lead_hours"], "metrics": ["mae", "rmse"]},
        ... )
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Build metadata
    meta = metadata or {}
    meta["computation_timestamp"] = datetime.now(UTC).isoformat()
    meta["row_count"] = len(df)

    # Convert to Arrow table with metadata
    table = pa.Table.from_pandas(df, preserve_index=False)

    # Add custom metadata
    import json

    existing_meta = dict(table.schema.metadata or {})
    existing_meta[b"weather_metrics"] = json.dumps(meta).encode("utf-8")
    table = table.replace_schema_metadata(existing_meta)

    pq.write_table(table, str(output_path), row_group_size=10000)

    logger.info("Wrote %d metric rows → %s", len(df), output_path)
    return output_path


# ---------------------------------------------------------------------------
# CSV export
# ---------------------------------------------------------------------------


def export_csv(
    df: pd.DataFrame,
    output_path: str | Path,
) -> Path:
    """Export metrics DataFrame to CSV.

    Useful for quick inspection in spreadsheets.

    Args:
        df: Metrics DataFrame.
        output_path: Output file path (.csv).

    Returns:
        Path to the written file.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(output_path, index=False)
    logger.info("Exported %d rows → %s", len(df), output_path)
    return output_path


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _resolve_paths(paths: str | Path | list[str | Path]) -> list[Path]:
    """Resolve paths, supporting glob patterns.

    Args:
        paths: Single path, glob pattern, or list of paths.

    Returns:
        List of resolved Path objects.
    """
    import glob

    if isinstance(paths, (str, Path)):
        paths = [paths]

    result: list[Path] = []
    for p in paths:
        p_str = str(p)
        # Check if it's a glob pattern
        if any(c in p_str for c in "*?["):
            matched = glob.glob(p_str)
            if not matched:
                logger.warning("Glob pattern matched no files: %s", p_str)
            else:
                result.extend(Path(m) for m in sorted(matched))
        else:
            result.append(Path(p))

    return result
