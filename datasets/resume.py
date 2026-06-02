"""Checkpoint and resumability logic for the dataset pipeline.

Tracks which pipeline steps have been completed by checking for existing
output files. This enables:

- Interrupted pipelines to resume from where they left off
- Partial rebuilds (skip steps that already have valid output)
- Idempotent pipeline runs (safe to re-run multiple times)

Step detection rules:
- download: raw/forecasts/ and raw/observations/ directories exist
- verify: verification/verification.parquet exists
- metrics: metrics/by_lead_hours/ directory exists with files
- events: events/ directory exists with parquet files
- aggregations: aggregates/ directory exists with files
- report: reports/report.md exists
- manifest: manifests/manifest.json exists

Usage
-----
>>> from datasets.resume import CheckpointTracker, StepStatus

>>> tracker = CheckpointTracker("./datasets/copenhagen_2y")
>>> tracker.check_step("download")   # returns True if completed
>>> tracker.check_step("verify")     # returns False (not yet)
>>> tracker.mark_complete("download")
>>> tracker.get_completed()          # returns ["download"]
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Pipeline step definitions
# ---------------------------------------------------------------------------

STEP_DOWNLOAD = "download"
STEP_VERIFY = "verify"
STEP_METRICS = "metrics"
STEP_EVENTS = "events"
STEP_AGGREGATIONS = "aggregations"
STEP_REPORT = "report"
STEP_MANIFEST = "manifest"

ALL_STEPS = (
    STEP_DOWNLOAD,
    STEP_VERIFY,
    STEP_METRICS,
    STEP_EVENTS,
    STEP_AGGREGATIONS,
    STEP_REPORT,
    STEP_MANIFEST,
)


class CheckpointTracker:
    """Tracks pipeline step completion via filesystem checks.

    A step is considered complete when its expected output files exist.
    This allows resumable pipeline execution.
    """

    def __init__(self, dataset_dir: str | Path) -> None:
        """Initialize checkpoint tracker.

        Args:
            dataset_dir: Root dataset directory path.
        """
        self.dataset_dir = Path(dataset_dir)
        self._completed: set[str] = set()

    # ------------------------------------------------------------------
    # Step completion checks
    # ------------------------------------------------------------------

    def check_step(self, step: str) -> bool:
        """Check if a pipeline step is complete.

        Args:
            step: Step name (e.g., "download", "verify", "metrics").

        Returns:
            True if the step's expected output exists, False otherwise.
        """
        if step in self._completed:
            return True

        result = self._check_step_filesystem(step)
        if result:
            self._completed.add(step)
        return result

    def _check_step_filesystem(self, step: str) -> bool:
        """Check if step output files exist on disk.

        Args:
            step: Step name.

        Returns:
            True if step is considered complete.
        """
        try:
            if step == STEP_DOWNLOAD:
                forecasts_exist = (
                    self.dataset_dir / "raw" / "forecasts"
                ).is_dir()
                observations_exist = (
                    self.dataset_dir / "raw" / "observations"
                ).is_dir()
                return forecasts_exist and observations_exist
            elif step == STEP_VERIFY:
                return (self.dataset_dir / "verification" / "verification.parquet").exists()
            elif step == STEP_METRICS:
                metrics_dir = self.dataset_dir / "metrics"
                return metrics_dir.is_dir() and any(metrics_dir.rglob("*.parquet"))
            elif step == STEP_EVENTS:
                events_dir = self.dataset_dir / "events"
                return events_dir.is_dir() and any(events_dir.rglob("*.parquet"))
            elif step == STEP_AGGREGATIONS:
                agg_dir = self.dataset_dir / "aggregates"
                return agg_dir.is_dir() and any(agg_dir.rglob("*.parquet"))
            elif step == STEP_REPORT:
                return (self.dataset_dir / "reports" / "report.md").exists()
            elif step == STEP_MANIFEST:
                return (self.dataset_dir / "manifests" / "manifest.json").exists()
            else:
                logger.warning("Unknown step: %s", step)
                return False
        except Exception as e:
            logger.debug("Error checking step '%s': %s", step, e)
            return False

    def mark_complete(self, step: str) -> None:
        """Mark a step as complete (in-memory only).

        Args:
            step: Step name.
        """
        self._completed.add(step)

    def get_completed(self) -> list[str]:
        """Get list of completed steps.

        Returns:
            List of step names that are complete.
        """
        return sorted(self._completed)

    def get_pending(self) -> list[str]:
        """Get list of pending (incomplete) steps.

        Returns:
            List of step names that need to be executed.
        """
        completed = self.get_completed()
        return [s for s in ALL_STEPS if s not in completed]

    def skip_step(self, step: str) -> bool:
        """Check if a step should be skipped (already complete).

        Args:
            step: Step name.

        Returns:
            True if step is already complete and should be skipped.
        """
        return self.check_step(step)

    def reset(self) -> None:
        """Reset all completed steps (in-memory only).

        Does not delete files, only clears the in-memory tracker.
        """
        self._completed.clear()
        logger.info("Checkpoint tracker reset")
