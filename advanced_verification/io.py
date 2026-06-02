"""I/O utilities for skill analysis.

Reads verification parquet files and writes event/confusion/skill outputs.

Usage
-----
>>> from advanced_verification.io import load_verification, write_skill_scores
>>> df = load_verification("data/verification/verification.parquet")
>>> write_skill_scores(skill_df, "data/advanced_verification/skill.parquet")
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


def load_verification(
    path: str | Path,
) -> pd.DataFrame:
    """Load verification parquet file(s).

    Reads one or more verification parquet files and combines them into
    a single DataFrame.

    Args:
        path: File path, directory, or glob pattern.
            - If a file: reads that file
            - If a directory: reads all *.parquet files
            - If a glob: uses glob expansion

    Returns:
        Combined DataFrame with all verification rows.

    Raises:
        FileNotFoundError: If no parquet files are found.
    """
    path = Path(path)

    if path.is_file():
        files = [path]
    elif path.is_dir():
        files = sorted(path.glob("*.parquet"))
    else:
        files = sorted(path.parent.glob(path.name))

    if not files:
        raise FileNotFoundError(f"No parquet files found at: {path}")

    logger.info("Loading %d verification file(s) from %s", len(files), path)
    dfs = [pd.read_parquet(f) for f in files]
    df = pd.concat(dfs, ignore_index=True)

    logger.info("Loaded %d verification rows", len(df))
    return df


def write_skill_scores(
    skill_df: pd.DataFrame,
    output_path: str | Path,
    event_name: str | None = None,
) -> Path:
    """Write skill scores DataFrame to parquet.

    Args:
        skill_df: Skill scores DataFrame from compute_skill_scores().
        output_path: Output file path (.parquet).
        event_name: Event name for filename generation (optional).

    Returns:
        Path to the written file.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    filename = output_path.name
    if not filename.endswith(".parquet"):
        if event_name:
            filename = f"{event_name}_skill_scores.parquet"
        else:
            filename = "skill_scores.parquet"
        output_path = output_path / filename

    skill_df.to_parquet(output_path, index=False)
    logger.info("Wrote %d skill score rows to %s", len(skill_df), output_path)
    return output_path


def write_confusion(
    confusion_df: pd.DataFrame,
    output_path: str | Path,
    event_name: str | None = None,
) -> Path:
    """Write confusion matrix DataFrame to parquet.

    Args:
        confusion_df: Confusion statistics DataFrame from compute_confusion_matrix().
        output_path: Output file path (.parquet).
        event_name: Event name for filename generation (optional).

    Returns:
        Path to the written file.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    filename = output_path.name
    if not filename.endswith(".parquet"):
        if event_name:
            filename = f"{event_name}_confusion.parquet"
        else:
            filename = "confusion_matrix.parquet"
        output_path = output_path / filename

    confusion_df.to_parquet(output_path, index=False)
    logger.info("Wrote %d confusion rows to %s", len(confusion_df), output_path)
    return output_path


def write_events(
    events_df: pd.DataFrame,
    output_path: str | Path,
    event_name: str | None = None,
) -> Path:
    """Write event-enriched DataFrame to parquet.

    Args:
        events_df: DataFrame with event columns from generate_events().
        output_path: Output file path (.parquet).
        event_name: Event name for filename generation (optional).

    Returns:
        Path to the written file.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    filename = output_path.name
    if not filename.endswith(".parquet"):
        if event_name:
            filename = f"{event_name}_events.parquet"
        else:
            filename = "events.parquet"
        output_path = output_path / filename

    events_df.to_parquet(output_path, index=False)
    logger.info("Wrote %d event rows to %s", len(events_df), output_path)
    return output_path
