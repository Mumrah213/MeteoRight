"""Phase 2: Transformation pipeline.

Converts validated Phase 1 RawForecastResponse objects into the
canonical coordinate system:

    (provider, model, run_time, valid_time, variable) → value

Pipeline stages:
  1. raw_response → ForecastRun[]     (normalize per location)
  2. ForecastRun[] → CanonicalForecastRecord[] (flatten per time×variable)
  3. CanonicalForecastRecord[] → ForecastIndex  (build query indices)

Each stage is pure and deterministic. No provider-specific logic
lives here — only transformation rules that apply to all providers.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import Any

from ..schemas.forecast_raw import RawForecastResponse
from .models import CanonicalForecastRecord, ForecastRun

logger = logging.getLogger(__name__)


# ── Helpers ────────────────────────────────────────────────────────────────


def _parse_iso_time(s: str) -> datetime:
    """Parse an ISO-8601 timestamp string into a naive datetime (UTC)."""
    dt = datetime.fromisoformat(s)
    if dt.tzinfo is not None:
        # Convert to UTC, then drop timezone info for canonical UTC-naive storage
        dt = dt.astimezone(UTC).replace(tzinfo=None)
    return dt


def _compute_lead_hours(
    run_time: datetime,
    valid_time: datetime,
) -> tuple[int, float]:
    """Compute lead time in integer hours.

    Returns:
        (lead_hours, drift) where drift is the fractional hour residual.

    Raises ValueError if lead time is non-integer or negative.
    """
    delta = valid_time - run_time
    total_seconds = delta.total_seconds()
    lead_hours = total_seconds / 3600

    # Fail fast on non-integer hours (irregular intervals)
    fractional = lead_hours - int(lead_hours)
    if abs(fractional) > 1e-6 and abs(fractional - 1.0) > 1e-6:
        raise ValueError(
            f"Lead time must be integer hours, got {lead_hours:.6f}h "
            f"(valid_time={valid_time}, run_time={run_time})"
        )

    lead_hours_int = int(round(lead_hours))
    if lead_hours_int < 0:
        raise ValueError(
            f"Lead time must be non-negative, got {lead_hours_int}h "
            f"(valid_time={valid_time}, run_time={run_time})"
        )

    return lead_hours_int, fractional


# ── Stage 1: RawResponse → ForecastRun ────────────────────────────────────


class RawToForecastRunError(Exception):
    """Raised when raw response cannot be converted to ForecastRun."""

    pass


def _extract_run_from_location(
    location_data: Any,  # RawLocationData
    raw_response: RawForecastResponse,
    time_granularity: str = "hourly",
) -> ForecastRun | None:
    """Extract a ForecastRun from one location's data.

    Returns None if the location has no data for the given granularity.
    """
    from ..schemas.forecast_raw import RawLocationData

    loc: RawLocationData = location_data

    # Select hourly or daily
    if time_granularity == "hourly":
        data = loc.hourly
        units = loc.hourly_units
    else:
        data = loc.daily
        units = loc.daily_units

    if data is None or not data:
        return None

    # Extract timestamp array
    time_key = "time"  # Provider standard
    times_raw = data.get(time_key)
    if not times_raw:
        return None

    # Extract run_time from request params or default to first available time
    run_time: datetime | None = None
    if raw_response.request_timestamp:
        try:
            run_time = _parse_iso_time(raw_response.request_timestamp)
        except (ValueError, TypeError):
            pass

    if run_time is None and times_raw:
        # Use first time entry as proxy for run_time
        run_time = _parse_iso_time(str(times_raw[0]))

    # Extract variables (all keys except 'time' and 'time_utc')
    variables: dict[str, list[float | None]] = {}
    for key, values in data.items():
        if key in ("time", "time_utc"):
            continue
        variables[key] = list(values) if values else []

    # Extract units
    units_dict: dict[str, str] = {}
    if units and isinstance(units, dict):
        for var_name in variables:
            units_dict[var_name] = units.get(var_name, "unknown")

    return ForecastRun(
        provider=raw_response.provider,
        model=raw_response.request_params.get("models", raw_response.request_params.get("model")),
        run_time=run_time,
        latitude=float(loc.latitude) if loc.latitude is not None else 0.0,
        longitude=float(loc.longitude) if loc.longitude is not None else 0.0,
        times=[str(t) for t in times_raw],
        variables=variables,
        units=units_dict,
        raw_ref=str(raw_response.raw_json) if raw_response.raw_json is not None else None,
        metadata={
            "endpoint": raw_response.endpoint,
            "request_params": raw_response.request_params,
            "time_granularity": time_granularity,
        },
    )


def raw_to_forecast_runs(
    response: RawForecastResponse,
    granularity: str = "hourly",
) -> list[ForecastRun]:
    """Convert a validated RawForecastResponse to one or more ForecastRuns.

    Each location in the response produces a separate ForecastRun (different
    lat/lon). The model is extracted from request_params.

    Raises RawToForecastRunError if conversion fails.
    """
    from ..schemas.forecast_raw import RawLocationData

    if not response.locations:
        raise RawToForecastRunError("Raw response has no locations to convert")

    runs: list[ForecastRun] = []

    for loc in response.locations:
        if not isinstance(loc, RawLocationData):
            loc = RawLocationData(**loc)  # Convert dict if needed

        run = _extract_run_from_location(loc, response, granularity)
        if run is not None:
            runs.append(run)

    if not runs:
        raise RawToForecastRunError("No valid data found in locations")

    logger.info(
        "Converted %d location(s) → %d ForecastRun(s)",
        len(response.locations),
        len(runs),
    )

    return runs


# ── Stage 2: ForecastRun → CanonicalForecastRecord[] ───────────────────────


class FlatteningError(Exception):
    """Raised when a ForecastRun cannot be flattened."""

    pass


def _flatten_run(run: ForecastRun) -> list[CanonicalForecastRecord]:
    """Flatten one ForecastRun into per-time-step, per-variable records.

    Each (time_index, variable) combination becomes one record.
    Null values are preserved — never dropped silently.

    Raises FlatteningError if the run is malformed.
    """
    n = run.time_count
    records: list[CanonicalForecastRecord] = []

    for i in range(n):
        # Parse valid_time from ISO string
        valid_time = _parse_iso_time(run.times[i])
        try:
            lead_hours_int, _ = _compute_lead_hours(run.run_time, valid_time)
        except ValueError as exc:
            raise FlatteningError(
                f"Invalid lead time at index {i} for {run.provider}: {exc}"
            ) from exc

        for var_name in run.variable_names:
            values = run.variables.get(var_name)
            value = values[i] if values and i < len(values) else None
            unit = run.units.get(var_name)

            records.append(
                CanonicalForecastRecord(
                    provider=run.provider,
                    model=run.model,
                    run_time=run.run_time,
                    valid_time=valid_time,
                    lead_hours=lead_hours_int,
                    latitude=run.latitude,
                    longitude=run.longitude,
                    variable=var_name,
                    value=value,
                    unit=unit,
                    raw_ref=run.raw_ref,
                    metadata=dict(run.metadata),
                )
            )

    return records


def forecast_runs_to_canonical(
    runs: list[ForecastRun],
) -> list[CanonicalForecastRecord]:
    """Convert multiple ForecastRuns into canonical records.

    Each run is flattened independently. The result is a flat list
    suitable for indexing.

    Raises FlatteningError if any run is malformed.
    """
    all_records: list[CanonicalForecastRecord] = []

    for run in runs:
        try:
            records = _flatten_run(run)
            all_records.extend(records)
        except Exception as exc:
            raise FlatteningError(
                f"Failed to flatten {run.provider} run at ({run.latitude}, {run.longitude}): {exc}"
            ) from exc

    logger.info(
        "Flattened %d ForecastRun(s) → %d CanonicalForecastRecord(s)",
        len(runs),
        len(all_records),
    )

    return all_records


# ── Stage 3: CanonicalForecastRecord[] → ForecastIndex ────────────────────

from .index import ForecastIndex, ForecastIndexError


def records_to_index(records: list[CanonicalForecastRecord]) -> ForecastIndex:
    """Build a queryable index from canonical records.

    Raises ForecastIndexError if records are invalid.
    """
    if not records:
        raise ForecastIndexError("No records to index")

    index = ForecastIndex.from_records(records)

    logger.info(
        "Built ForecastIndex: %d runs, %d valid_time buckets, %d lead_time buckets",
        len(index.by_run_time),
        len(index.by_valid_time),
        len(index.by_lead_time),
    )

    return index


# ── Full Pipeline ──────────────────────────────────────────────────────────


class CanonicalPipeline:
    """End-to-end transformation pipeline.

    Usage:
        pipeline = CanonicalPipeline()
        runs = pipeline.raw_to_runs(response)
        canonical = pipeline.runs_to_canonical(runs)
        index = pipeline.index(canonical)

    Each stage can be called independently for testing.
    """

    def raw_to_runs(
        self,
        response: RawForecastResponse,
        granularity: str = "hourly",
    ) -> list[ForecastRun]:
        """Stage 1: RawResponse → ForecastRun[]."""
        return raw_to_forecast_runs(response, granularity)

    def runs_to_canonical(self, runs: list[ForecastRun]) -> list[CanonicalForecastRecord]:
        """Stage 2: ForecastRun[] → CanonicalForecastRecord[]."""
        return forecast_runs_to_canonical(runs)

    def build_index(self, records: list[CanonicalForecastRecord]) -> ForecastIndex:
        """Stage 3: CanonicalForecastRecord[] → ForecastIndex."""
        return records_to_index(records)

    def transform(
        self,
        response: RawForecastResponse,
        granularity: str = "hourly",
    ) -> tuple[list[ForecastRun], list[CanonicalForecastRecord], ForecastIndex]:
        """Full pipeline: raw → runs → canonical → index.

        Returns (runs, records, index) for inspection.
        """
        runs = self.raw_to_runs(response, granularity)
        records = self.runs_to_canonical(runs)
        index = self.build_index(records)
        return runs, records, index
