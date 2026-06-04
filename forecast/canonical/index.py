"""Phase 2: Forecast index for fast querying.

Provides O(1) lookups by valid_time, lead_time, and run_time.
This enables future analysis (lead-time error curves, model
comparison, temporal slicing) without scanning the full record set.

Index is built once from canonical records, then queried.
"""


from collections import defaultdict
from datetime import datetime

from .models import CanonicalForecastRecord


class ForecastIndexError(Exception):
    """Raised when index building or querying encounters an error."""


class ForecastIndex:
    """In-memory query index over canonical forecast records.

    Three access patterns:
      - by_valid_time[datetime] → list[records at that time]
      - by_lead_time[int]       → list[records at that lead hour]
      - by_run_time[(run_time, lat, lon)] → list[records from one run

    Run keys are composite (datetime, latitude, longitude) to
    distinguish multi-location forecasts with the same issue time.

    All three views are built simultaneously during construction.
    """

    def __init__(
        self,
        by_valid_time: dict[datetime, list[CanonicalForecastRecord]] | None = None,
        by_lead_time: dict[int, list[CanonicalForecastRecord]] | None = None,
        by_run_time: dict[tuple[datetime, float, float], list[CanonicalForecastRecord]]
        | None = None,
    ) -> None:
        self.by_valid_time: dict[datetime, list[CanonicalForecastRecord]] = by_valid_time or {}
        self.by_lead_time: dict[int, list[CanonicalForecastRecord]] = by_lead_time or {}
        self.by_run_time: dict[tuple[datetime, float, float], list[CanonicalForecastRecord]] = (
            by_run_time or {}
        )

    @classmethod
    def from_records(cls, records: list[CanonicalForecastRecord]) -> ForecastIndex:
        """Build index from a list of canonical records.

        All records are distributed into the three index buckets
        simultaneously for efficiency. Run keys are composite
        (run_time, latitude, longitude) to distinguish multi-location
        forecasts.

        Raises ValueError if:
        - Duplicate keys exist (global uniqueness violation)
        - Record count doesn't match expected (duplication explosion)
        """
        if not records:
            raise ForecastIndexError("No records to index")

        # ── Global uniqueness check ──────────────────────────────────
        seen_keys: set[tuple] = set()
        duplicate_keys: set[tuple] = set()
        for rec in records:
            key = rec.key  # (provider, model, run_time, valid_time, variable, lat, lon)
            if key in seen_keys:
                duplicate_keys.add(key)
            seen_keys.add(key)

        if duplicate_keys:
            sample = next(iter(duplicate_keys))
            raise ForecastIndexError(
                f"Duplicate canonical records found: {sample!r}. "
                f"Total duplicates: {len(duplicate_keys)}. "
                f"Global uniqueness requires each key to appear exactly once."
            )

        # ── Duplication explosion check ──────────────────────────────
        # Group records by (provider, model, run_time, lat, lon)
        # to detect cartesian product bugs in flattening.
        run_buckets: dict[
            tuple[str, str | None, datetime, float, float], list[CanonicalForecastRecord]
        ] = defaultdict(list)
        for rec in records:
            run_key = (rec.provider, rec.model, rec.run_time, rec.latitude, rec.longitude)
            run_buckets[run_key].append(rec)

        # For each run group, verify no (valid_time, variable) pair
        # appears more than once. Key uniqueness already guarantees this,
        # but we additionally check that actual record count does not
        # exceed unique_times × unique_vars (catches cartesian product bugs).
        for run_key, bucket in run_buckets.items():
            unique_vars = {r.variable for r in bucket}
            unique_times = {r.valid_time for r in bucket}
            max_possible = len(unique_times) * len(unique_vars)
            actual = len(bucket)
            if actual > max_possible:
                raise ForecastIndexError(
                    f"Duplication explosion detected for run {run_key}: "
                    f"got {actual} records but max possible is "
                    f"{max_possible} ({len(unique_times)} times × "
                    f"{len(unique_vars)} vars). "
                    f"This indicates a flattening loop or cartesian product bug."
                )

        # ── Build index buckets ──────────────────────────────────────
        by_valid_time: dict[datetime, list[CanonicalForecastRecord]] = defaultdict(list)
        by_lead_time: dict[int, list[CanonicalForecastRecord]] = defaultdict(list)
        by_run_time: dict[tuple[datetime, float, float], list[CanonicalForecastRecord]] = (
            defaultdict(list)
        )

        for record in records:
            by_valid_time[record.valid_time].append(record)
            by_lead_time[record.lead_hours].append(record)
            run_key = (record.run_time, record.latitude, record.longitude)
            by_run_time[run_key].append(record)

        return cls(
            by_valid_time=dict(by_valid_time),
            by_lead_time=dict(by_lead_time),
            by_run_time=dict(by_run_time),
        )

    # ── Query methods ──────────────────────────────────────────────────

    def at_valid_time(self, valid_time: datetime) -> list[CanonicalForecastRecord]:
        """Get all records at a specific valid_time."""
        return self.by_valid_time.get(valid_time, [])

    def at_lead_time(self, lead_hours: int) -> list[CanonicalForecastRecord]:
        """Get all records at a specific lead hour."""
        return self.by_lead_time.get(lead_hours, [])

    def at_run_time(
        self,
        run_time: datetime,
        latitude: float | None = None,
        longitude: float | None = None,
    ) -> list[CanonicalForecastRecord]:
        """Get all records from a specific forecast run.

        If latitude and longitude are provided, returns only the run
        at that location. Otherwise returns ALL runs issued at that time.
        """
        if latitude is not None and longitude is not None:
            return self.by_run_time.get((run_time, latitude, longitude), [])
        # No location filter — return all runs at this run_time
        result: list[CanonicalForecastRecord] = []
        for key in self.by_run_time:
            if key[0] == run_time:
                result.extend(self.by_run_time[key])
        return result

    def get_variable_records(
        self,
        variable: str,
        run_time: datetime | None = None,
        lead_hours: int | None = None,
        valid_time: datetime | None = None,
    ) -> list[CanonicalForecastRecord]:
        """Get records for a specific variable, optionally filtered.

        If no filter is given, returns ALL records for the variable.
        Only one filter can be active at a time (run_time, lead_hours,
        or valid_time).
        """
        all_var_records: list[CanonicalForecastRecord] = []

        # Collect candidate records based on filter
        if run_time is not None:
            candidates = self.at_run_time(run_time)
        elif lead_hours is not None:
            candidates = self.at_lead_time(lead_hours)
        elif valid_time is not None:
            candidates = self.at_valid_time(valid_time)
        else:
            # No filter — scan all index buckets
            for records in self.by_valid_time.values():
                all_var_records.extend(records)

        if run_time is not None or lead_hours is not None or valid_time is not None:
            candidates = [r for r in candidates if r.variable == variable]
        else:
            all_var_records = [r for r in all_var_records if r.variable == variable]

        return candidates

    # ── Summary methods ────────────────────────────────────────────────

    def run_count(self) -> int:
        """Number of unique forecast runs."""
        return len(self.by_run_time)

    def valid_time_count(self) -> int:
        """Number of unique valid_time points."""
        return len(self.by_valid_time)

    def lead_time_range(self) -> tuple[int, int] | None:
        """(min_lead, max_lead) in hours, or None if empty."""
        if not self.by_lead_time:
            return None
        return (min(self.by_lead_time.keys()), max(self.by_lead_time.keys()))

    def variable_names(self) -> set[str]:
        """All unique variable names in the index."""
        return {r.variable for records in self.by_valid_time.values() for r in records}

    def run_ids(self) -> list[tuple[str, str, datetime, float, float]]:
        """List of (provider, model, run_time, lat, lon) for each run."""
        result: list[tuple[str, str, datetime, float, float]] = []
        for (rt, lat, lon), records in self.by_run_time.items():
            if records:
                r = records[0]
                result.append((r.provider, r.model or "", rt, lat, lon))
        return result
