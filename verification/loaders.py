"""Loaders for forecast and observation parquet datasets.

Partition-aware reading with date-range filtering. Eager (in-memory) loading
returning pandas DataFrames with timezone-aware UTC timestamps.
"""

import logging
from pathlib import Path

import pandas as pd
import pyarrow.dataset as ds

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Forecast loader
# ---------------------------------------------------------------------------


def load_forecasts(
    data_dir: Path,
    model: str,
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    """Load forecast parquet files for a model and date range.

    Reads from: data_dir/forecasts/model={model}/year={y}/month={m}/data.parquet

    Filters:
      - model partition
      - forecast_target_time within [start_date, end_date] (inclusive)

    Args:
        data_dir: Root data directory containing the partitions.
        model: Model name (e.g. "ecmwf_ifs_hres").
        start_date: Start date string (YYYY-MM-DD).
        end_date: End date string (YYYY-MM-DD).

    Returns:
        DataFrame with columns:
          forecast_issue_time, forecast_target_time, lead_hours, model,
          latitude, longitude, elevation, [variable columns...]

        All timestamps are timezone-aware UTC. lead_hours is int32.

    Raises:
        FileNotFoundError: If no forecast partitions exist for the model.
    """
    base_path = data_dir / "forecasts" / f"model={model}"

    if not base_path.exists():
        raise FileNotFoundError(
            f"No forecast partitions found for model '{model}' at {base_path}. "
            f"Verify the model name and that data has been downloaded."
        )

    # Build date filter on forecast_target_time
    start_ts = pd.Timestamp(start_date, tz="UTC")
    end_ts = pd.Timestamp(end_date, tz="UTC")
    # End date is inclusive: include the entire end day
    end_ts = end_ts.replace(hour=23, minute=59, second=59)

    dataset = ds.dataset(str(base_path), format="parquet")
    filter_expr = (ds.field("forecast_target_time") >= start_ts) & (
        ds.field("forecast_target_time") <= end_ts
    )

    logger.info(
        "Loading forecasts: model=%s, %s to %s",
        model,
        start_date,
        end_date,
    )

    table = dataset.to_table(filter=filter_expr)
    df = table.to_pandas()

    # Ensure timezone-aware timestamps
    for col in ["forecast_issue_time", "forecast_target_time"]:
        if df[col].dt.tz is None:
            df[col] = df[col].dt.tz_localize("UTC")

    # Ensure lead_hours is int32
    if "lead_hours" in df.columns:
        df["lead_hours"] = df["lead_hours"].astype("int32")

    logger.info("  Found %d forecast rows", len(df))

    return df


# ---------------------------------------------------------------------------
# Observation loader
# ---------------------------------------------------------------------------


def load_observations(
    data_dir: Path,
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    """Load observation parquet files for a date range.

    Reads from: data_dir/observations/year={y}/month={m}/data.parquet

    Filters:
      - observation_time within [start_date, end_date] (inclusive)

    Args:
        data_dir: Root data directory containing the partitions.
        start_date: Start date string (YYYY-MM-DD).
        end_date: End date string (YYYY-MM-DD).

    Returns:
        DataFrame with columns:
          observation_time, latitude, longitude, elevation, [variable columns...]

        All timestamps are timezone-aware UTC.

    Raises:
        FileNotFoundError: If no observation partitions exist.
    """
    base_path = data_dir / "observations"

    if not base_path.exists():
        raise FileNotFoundError(
            f"No observation partitions found at {base_path}. "
            f"Verify that observation data has been downloaded."
        )

    # Build date filter on observation_time
    start_ts = pd.Timestamp(start_date, tz="UTC")
    end_ts = pd.Timestamp(end_date, tz="UTC")
    end_ts = end_ts.replace(hour=23, minute=59, second=59)

    dataset = ds.dataset(str(base_path), format="parquet")
    filter_expr = (ds.field("observation_time") >= start_ts) & (
        ds.field("observation_time") <= end_ts
    )

    logger.info(
        "Loading observations: %s to %s",
        start_date,
        end_date,
    )

    table = dataset.to_table(filter=filter_expr)
    df = table.to_pandas()

    # Ensure timezone-aware timestamps
    if "observation_time" in df.columns and df["observation_time"].dt.tz is None:
        df["observation_time"] = df["observation_time"].dt.tz_localize("UTC")

    logger.info("  Found %d observation rows", len(df))

    return df
