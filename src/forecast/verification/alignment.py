"""Phase 3: Forecast-Observation Alignment Engine.

Aligns forecasts and observations on the key:
  (latitude, longitude, valid_time, variable)

Rules:
- Missing matches are explicit gaps (not interpolated)
- Nulls are preserved separately
- All mismatches are tracked explicitly

Produces AlignmentRecord objects that carry both forecast and
observation values (or None if one is missing).
"""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime
from typing import Any

from ..canonical.models import CanonicalForecastRecord
from .models import AlignmentRecord


class AlignmentError(Exception):
    """Raised when alignment fails."""


class RunIndependenceError(AlignmentError):
    """Raised when forecast runs are not properly separated during alignment.

    This error prevents silent merging of overlapping forecast runs,
    which would contaminate multi-run comparisons.
    """


class RunOverlapWarning(AlignmentError):
    """Warns that two forecast runs share valid_time windows.

    Not necessarily wrong, but should be handled explicitly.
    """


class AlignmentEngine:
    """Aligns forecasts and observations by key.

    Forecast key: (latitude, longitude, valid_time, variable, run_time)
    Observation key: (latitude, longitude, valid_time, variable)

    This prevents run-identity collision: if two forecast runs both
    predict the same valid_time and variable, each run's forecast is
    aligned separately against observations.

    Three match statuses:
    - "matched": both forecast and observation present
    - "forecast_only": no observation at this point
    - "observation_only": no forecast at this point

    Usage:
        engine = AlignmentEngine()
        records = engine.align(forecasts, observations)
    """

    def __init__(self, strict: bool = True) -> None:
        """Initialize alignment engine.

        Args:
            strict: If True, raises on empty inputs. If False,
                    returns empty alignment list.
        """
        self.strict = strict
        self._overlap_warnings: list[dict[str, Any]] = []

    def align(
        self,
        forecasts: list[CanonicalForecastRecord],
        observations: list[CanonicalForecastRecord],
    ) -> list[AlignmentRecord]:
        """Align forecasts and observations.

        Key distinction:
        - Forecasts are keyed by (lat, lon, valid_time, variable, run_time)
          to preserve run identity across overlapping valid_time windows.
        - Observations are keyed by (lat, lon, valid_time, variable)
          since observations have no forecast run context.

        When multiple forecast runs predict the same (lat, lon, valid_time,
        variable), each run produces a separate alignment record against
        the same observation. This is essential for valid multi-run
        comparison.

        Args:
            forecasts: Forecast records (is_observation=False)
            observations: Observation records (is_observation=True)

        Returns:
            List of AlignmentRecord objects with match_status
        """
        if self.strict and (not forecasts and not observations):
            raise AlignmentError("Cannot align empty forecast and observation sets")

        # Build forecast index with run_time in key
        forecast_index = self._build_forecast_index(forecasts)
        # Build observation index without run_time in key
        observation_index = self._build_observation_index(observations)

        # Collect all keys from both indices
        all_keys: set[tuple[float, float, datetime, str]] = set()
        all_keys.update(observation_index.keys())
        # For forecasts, key excludes run_time for matching purposes
        for forecast_key in forecast_index.keys():
            all_keys.add(forecast_key[:4])  # (lat, lon, valid_time, variable)

        # Create alignment records
        alignments: list[AlignmentRecord] = []

        for key in sorted(all_keys):
            lat, lon, valid_time, variable = key
            forecast_records = forecast_index.get(key, [])
            observation_records = observation_index.get(key, [])

            # Each forecast is aligned against each observation at the same key
            if forecast_records and observation_records:
                for forecast_rec in forecast_records:
                    for obs_rec in observation_records:
                        lead_hours = forecast_rec.lead_hours
                        alignment = self._make_alignment(
                            lat,
                            lon,
                            valid_time,
                            variable,
                            lead_hours,
                            forecast_rec,
                            obs_rec,
                            "matched",
                        )
                        alignments.append(alignment)
            elif forecast_records:
                # Forecast-only: each forecast aligned with no observation
                for forecast_rec in forecast_records:
                    lead_hours = forecast_rec.lead_hours
                    alignment = self._make_alignment(
                        lat,
                        lon,
                        valid_time,
                        variable,
                        lead_hours,
                        forecast_rec,
                        None,
                        "forecast_only",
                    )
                    alignments.append(alignment)
            elif observation_records:
                # Observation-only: each observation aligned with no forecast
                for obs_rec in observation_records:
                    lead_hours = obs_rec.lead_hours if obs_rec else 0
                    alignment = self._make_alignment(
                        lat,
                        lon,
                        valid_time,
                        variable,
                        lead_hours,
                        None,
                        obs_rec,
                        "observation_only",
                    )
                    alignments.append(alignment)

        return alignments

    def _make_alignment(
        self,
        lat: float,
        lon: float,
        valid_time: datetime,
        variable: str,
        lead_hours: int,
        forecast_rec: CanonicalForecastRecord | None,
        obs_rec: CanonicalForecastRecord | None,
        match_status: str,
    ) -> AlignmentRecord:
        """Construct a single AlignmentRecord from forecast and observation.

        Temporal metadata (month, season, year) is derived from valid_time
        via model_validator in AlignmentRecord.
        """
        return AlignmentRecord(
            latitude=lat,
            longitude=lon,
            valid_time=valid_time,
            variable=variable,
            lead_hours=lead_hours,
            forecast_value=forecast_rec.value if forecast_rec else None,
            forecast_provider=forecast_rec.provider if forecast_rec else None,
            forecast_model=forecast_rec.model if forecast_rec else None,
            forecast_run_time=forecast_rec.run_time if forecast_rec else None,
            observation_value=obs_rec.value if obs_rec else None,
            observation_model=obs_rec.model if obs_rec else None,
            match_status=match_status,
        )

    def _build_forecast_index(
        self,
        forecasts: list[CanonicalForecastRecord],
    ) -> dict[tuple[float, float, datetime, str], list[CanonicalForecastRecord]]:
        """Build forecast lookup index with run_time in key.

        Key: (latitude, longitude, valid_time, variable)

        Forecasts that share (lat, lon, valid_time, variable) but differ
        in run_time are preserved separately — they appear as distinct
        entries in the list. This enables multi-run comparison.

        Args:
            forecasts: Forecast records only (is_observation=False)

        Returns:
            Dict mapping (lat, lon, valid_time, variable) to lists of
            CanonicalForecastRecord. Each forecast retains its run_time
            for downstream separation.
        """
        index: dict[tuple[float, float, datetime, str], list[CanonicalForecastRecord]] = (
            defaultdict(list)
        )

        for rec in forecasts:
            key = (rec.latitude, rec.longitude, rec.valid_time, rec.variable)
            index[key].append(rec)

        return dict(index)

    def _build_observation_index(
        self,
        observations: list[CanonicalForecastRecord],
    ) -> dict[tuple[float, float, datetime, str], list[CanonicalForecastRecord]]:
        """Build observation lookup index (no run_time dimension).

        Observations are keyed by (lat, lon, valid_time, variable)
        since they represent ground truth at a specific time and location.

        Args:
            observations: Observation records (is_observation=True)

        Returns:
            Dict mapping keys to lists of CanonicalForecastRecord
        """
        index: dict[tuple[float, float, datetime, str], list[CanonicalForecastRecord]] = (
            defaultdict(list)
        )

        for rec in observations:
            key = (rec.latitude, rec.longitude, rec.valid_time, rec.variable)
            index[key].append(rec)

        return dict(index)

    def check_run_overlap(
        self,
        forecasts: list[CanonicalForecastRecord],
    ) -> list[dict[str, Any]]:
        """Detect overlapping forecast runs that share valid_time windows.

        If two or more runs forecast the same (lat, lon, valid_time,
        variable), this method identifies those overlaps. Overlapping
        runs are not independent samples and should be handled explicitly.

        Args:
            forecasts: Forecast records from multiple runs

        Returns:
            List of dicts describing overlap groups.
            Each dict has: run_times, time_range, location, variable,
            record_count
        """
        # Group forecasts by (lat, lon, valid_time, variable)
        groups: dict[tuple[float, float, datetime, str], list[CanonicalForecastRecord]] = (
            defaultdict(list)
        )
        for rec in forecasts:
            key = (rec.latitude, rec.longitude, rec.valid_time, rec.variable)
            groups[key].append(rec)

        overlaps: list[dict[str, Any]] = []
        for key, records in groups.items():
            run_times = sorted({r.run_time for r in records})
            if len(run_times) > 1:
                lat, lon, valid_time, variable = key
                overlaps.append(
                    {
                        "run_times": [rt.isoformat() for rt in run_times],
                        "num_runs": len(run_times),
                        "valid_time": valid_time.isoformat(),
                        "location": (lat, lon),
                        "variable": variable,
                        "record_count": len(records),
                    }
                )

        self._overlap_warnings = overlaps
        return overlaps

    def get_overlap_warnings(self) -> list[dict[str, Any]]:
        """Return previously detected run overlaps from check_run_overlap()."""
        return self._overlap_warnings

    def get_matched_records(self, alignments: list[AlignmentRecord]) -> list[AlignmentRecord]:
        """Filter to only fully matched records.

        Both forecast and observation must be present and non-null.
        """
        return [a for a in alignments if a.is_fully_matched]

    def get_gaps(self, alignments: list[AlignmentRecord]) -> dict[str, list[AlignmentRecord]]:
        """Identify explicit gaps in alignment.

        Returns:
            Dict with keys "forecast_only" and "observation_only"
        """
        forecast_only = [a for a in alignments if a.match_status == "forecast_only"]
        observation_only = [a for a in alignments if a.match_status == "observation_only"]
        return {
            "forecast_only": forecast_only,
            "observation_only": observation_only,
        }

    def get_by_variable(
        self, alignments: list[AlignmentRecord]
    ) -> dict[str, list[AlignmentRecord]]:
        """Group alignments by variable name."""
        grouped: dict[str, list[AlignmentRecord]] = defaultdict(list)
        for alignment in alignments:
            grouped[alignment.variable].append(alignment)
        return dict(grouped)

    def get_by_lead_time(
        self, alignments: list[AlignmentRecord]
    ) -> dict[int, list[AlignmentRecord]]:
        """Group alignments by lead_hours."""
        grouped: dict[int, list[AlignmentRecord]] = defaultdict(list)
        for alignment in alignments:
            grouped[alignment.lead_hours].append(alignment)
        return dict(grouped)

    def get_by_location(
        self, alignments: list[AlignmentRecord]
    ) -> dict[tuple[float, float], list[AlignmentRecord]]:
        """Group alignments by (latitude, longitude)."""
        grouped: dict[tuple[float, float], list[AlignmentRecord]] = defaultdict(list)
        for alignment in alignments:
            key = (alignment.latitude, alignment.longitude)
            grouped[key].append(alignment)
        return dict(grouped)
