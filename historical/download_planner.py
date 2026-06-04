"""Manifest-driven downloader for larger research datasets.

The planner separates *what to download* from *executing the download*. That
makes it possible to estimate request counts, review scope, and resume from a
checkpoint log after interruptions.
"""


import json
import logging
import time
from dataclasses import asdict, dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any, Literal

import pandas as pd

from .api import (
    fetch_historical_forecast,
    fetch_observations,
    fetch_previous_runs,
    fetch_single_run,
)
from .constants import DEFAULT_API_SLEEP
from .normalize import (
    json_to_forecast_df,
    json_to_observation_df,
    json_to_previous_runs_forecast_df,
)
from .storage import write_attribution, write_download_log, write_forecasts, write_observations
from verification.location import DEFAULT_MAX_LOCATION_DISTANCE_KM, haversine_km

logger = logging.getLogger(__name__)

DOWNLOAD_KIND = Literal["observations", "previous_runs", "historical_forecast", "single_runs"]


@dataclass(frozen=True)
class LocationSpec:
    name: str
    lat: float
    lon: float
    timezone: str = "UTC"
    max_grid_distance_km: float = DEFAULT_MAX_LOCATION_DISTANCE_KM

    @property
    def slug(self) -> str:
        return "".join(ch.lower() if ch.isalnum() else "_" for ch in self.name).strip("_")


@dataclass(frozen=True)
class DownloadTask:
    task_id: str
    kind: DOWNLOAD_KIND
    location: LocationSpec
    start_date: str
    end_date: str
    variables: tuple[str, ...]
    model: str | None = None
    lead_days: tuple[int, ...] = ()
    run_time: str | None = None
    forecast_hours: int | None = None
    past_hours: int | None = None


@dataclass(frozen=True)
class DownloadPlan:
    tasks: tuple[DownloadTask, ...]

    @property
    def request_count(self) -> int:
        return len(self.tasks)

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame([task_to_record(task) for task in self.tasks])

    def write_manifest(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps([task_to_record(task) for task in self.tasks], indent=2),
            encoding="utf-8",
        )
        return path


def task_to_record(task: DownloadTask) -> dict[str, Any]:
    record = asdict(task)
    record["location"] = asdict(task.location)
    return record


def _parse_date(value: str) -> date:
    return datetime.strptime(value, "%Y-%m-%d").date()


def _month_chunks(start_date: str, end_date: str, chunk_months: int = 1) -> list[tuple[str, str]]:
    """Split a date range into month-based inclusive chunks."""
    start = _parse_date(start_date)
    end = _parse_date(end_date)
    if end < start:
        raise ValueError(f"end_date {end_date!r} must be on or after start_date {start_date!r}")
    if chunk_months < 1:
        raise ValueError("chunk_months must be >= 1")

    chunks: list[tuple[str, str]] = []
    current = start.replace(day=1)
    while current <= end:
        month = current.month - 1 + chunk_months
        next_year = current.year + month // 12
        next_month = month % 12 + 1
        next_start = date(next_year, next_month, 1)
        chunk_start = max(current, start)
        chunk_end = min(next_start - timedelta(days=1), end)
        chunks.append((chunk_start.isoformat(), chunk_end.isoformat()))
        current = next_start
    return chunks


def _run_times(
    start_date: str,
    end_date: str,
    run_hours: tuple[int, ...],
    interval_days: int,
) -> list[datetime]:
    start = _parse_date(start_date)
    end = _parse_date(end_date)
    if interval_days < 1:
        raise ValueError("interval_days must be >= 1")

    result: list[datetime] = []
    current = start
    while current <= end:
        for hour in run_hours:
            result.append(datetime(current.year, current.month, current.day, hour, tzinfo=UTC))
        current += timedelta(days=interval_days)
    return result


def build_download_plan(
    locations: list[LocationSpec | dict[str, Any]],
    start_date: str,
    end_date: str,
    variables: tuple[str, ...],
    models: tuple[str, ...] = (),
    kinds: tuple[DOWNLOAD_KIND, ...] = ("observations", "previous_runs"),
    lead_days: tuple[int, ...] = (1, 2, 3, 4, 5, 6, 7),
    chunk_months: int = 1,
    single_run_hours: tuple[int, ...] = (0,),
    single_run_interval_days: int = 1,
    single_run_forecast_hours: int | None = None,
) -> DownloadPlan:
    """Build a resumable download plan and estimate request count."""
    locs = [
        loc
        if isinstance(loc, LocationSpec)
            else LocationSpec(
                name=loc["name"],
                lat=float(loc["lat"]),
                lon=float(loc["lon"]),
                timezone=loc.get("timezone", "UTC"),
                max_grid_distance_km=float(
                    loc.get("max_grid_distance_km", DEFAULT_MAX_LOCATION_DISTANCE_KM)
                ),
            )
        for loc in locations
    ]
    chunks = _month_chunks(start_date, end_date, chunk_months=chunk_months)
    tasks: list[DownloadTask] = []

    for loc in locs:
        for chunk_start, chunk_end in chunks:
            if "observations" in kinds:
                tasks.append(
                    DownloadTask(
                        task_id=f"obs:{loc.slug}:{chunk_start}:{chunk_end}",
                        kind="observations",
                        location=loc,
                        start_date=chunk_start,
                        end_date=chunk_end,
                        variables=variables,
                    )
                )

            for model in models:
                if "previous_runs" in kinds:
                    tasks.append(
                        DownloadTask(
                            task_id=f"prev:{loc.slug}:{model}:{chunk_start}:{chunk_end}:{','.join(map(str, lead_days))}",
                            kind="previous_runs",
                            location=loc,
                            start_date=chunk_start,
                            end_date=chunk_end,
                            variables=variables,
                            model=model,
                            lead_days=lead_days,
                        )
                    )

                if "historical_forecast" in kinds:
                    tasks.append(
                        DownloadTask(
                            task_id=f"histfc:{loc.slug}:{model}:{chunk_start}:{chunk_end}",
                            kind="historical_forecast",
                            location=loc,
                            start_date=chunk_start,
                            end_date=chunk_end,
                            variables=variables,
                            model=model,
                        )
                    )

        if "single_runs" in kinds:
            for model in models:
                for run_time in _run_times(
                    start_date, end_date, single_run_hours, single_run_interval_days
                ):
                    run_str = run_time.isoformat()
                    tasks.append(
                        DownloadTask(
                            task_id=f"single:{loc.slug}:{model}:{run_str}",
                            kind="single_runs",
                            location=loc,
                            start_date=run_time.date().isoformat(),
                            end_date=run_time.date().isoformat(),
                            variables=variables,
                            model=model,
                            run_time=run_str,
                            forecast_hours=single_run_forecast_hours,
                        )
                    )

    return DownloadPlan(tasks=tuple(tasks))


def _checkpoint_path(output_dir: Path) -> Path:
    return output_dir / "metadata" / "download_checkpoints.jsonl"


def _read_completed(output_dir: Path) -> set[str]:
    path = _checkpoint_path(output_dir)
    if not path.exists():
        return set()
    completed: set[str] = set()
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if record.get("status") == "success" and record.get("task_id"):
                completed.add(record["task_id"])
    return completed


def _write_checkpoint(
    output_dir: Path, task: DownloadTask, status: str, rows: int = 0, error: str | None = None
) -> None:
    path = _checkpoint_path(output_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "task_id": task.task_id,
        "kind": task.kind,
        "status": status,
        "rows": rows,
        "error": error,
        "timestamp": datetime.now(UTC).isoformat(),
    }
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(record) + "\n")


def execute_download_plan(
    plan: DownloadPlan,
    output_dir: str | Path,
    resume: bool = True,
    sleep_seconds: float = DEFAULT_API_SLEEP,
    max_tasks: int | None = None,
) -> dict[str, Any]:
    """Execute a plan with checkpoint-based resume support."""
    output_path = Path(output_dir)
    completed = _read_completed(output_path) if resume else set()
    result: dict[str, Any] = {
        "requests_planned": plan.request_count,
        "requests_attempted": 0,
        "requests_skipped": 0,
        "rows_written": 0,
        "errors": [],
    }

    tasks = list(plan.tasks)
    if max_tasks is not None:
        tasks = tasks[:max_tasks]

    for index, task in enumerate(tasks):
        if resume and task.task_id in completed:
            result["requests_skipped"] += 1
            continue

        try:
            rows = _execute_task(task, output_path)
            result["requests_attempted"] += 1
            result["rows_written"] += rows
            _write_checkpoint(output_path, task, "success", rows=rows)
            if sleep_seconds and index < len(tasks) - 1:
                time.sleep(sleep_seconds)
        except Exception as exc:
            logger.exception("Download task failed: %s", task.task_id)
            error = {"task_id": task.task_id, "kind": task.kind, "error": str(exc)}
            result["errors"].append(error)
            _write_checkpoint(output_path, task, "error", error=str(exc))

    return result


def _execute_task(task: DownloadTask, output_dir: Path) -> int:
    loc = task.location
    if task.kind == "observations":
        response = fetch_observations(
            loc.lat, loc.lon, task.start_date, task.end_date, task.variables
        )
        distance_km = _validate_response_location(response, task)
        df = json_to_observation_df(
            response,
            task.variables,
            **_location_kwargs(task, distance_km),
        )
        rows = write_observations(df, output_dir)
        write_download_log(output_dir, "observations", None, task.start_date, task.end_date, rows)
        write_attribution(
            output_dir,
            "observations",
            "open-meteo-archive",
            loc.lat,
            loc.lon,
            start_date=task.start_date,
            end_date=task.end_date,
        )
        return rows

    if not task.model:
        raise ValueError(f"Task {task.task_id} requires model")

    if task.kind == "previous_runs":
        response = fetch_previous_runs(
            loc.lat,
            loc.lon,
            task.start_date,
            task.end_date,
            task.variables,
            task.model,
            task.lead_days,
        )
        distance_km = _validate_response_location(response, task)
        df = json_to_previous_runs_forecast_df(
            response,
            task.model,
            task.variables,
            task.lead_days,
            **_location_kwargs(task, distance_km),
        )
        rows = write_forecasts(df, output_dir)
        write_download_log(
            output_dir, "forecasts", task.model, task.start_date, task.end_date, rows
        )
        write_attribution(
            output_dir,
            "forecasts",
            "open-meteo-previous-runs",
            loc.lat,
            loc.lon,
            task.model,
            task.start_date,
            task.end_date,
        )
        return rows

    if task.kind == "historical_forecast":
        response = fetch_historical_forecast(
            loc.lat,
            loc.lon,
            task.start_date,
            task.end_date,
            task.variables,
            task.model,
        )
        distance_km = _validate_response_location(response, task)
        # Stitched historical forecasts do not preserve original issue time.
        df = json_to_previous_runs_forecast_df(
            _as_day0_previous_runs_response(response, task.variables),
            task.model,
            task.variables,
            lead_days=(0,),
            **_location_kwargs(task, distance_km),
        )
        rows = write_forecasts(df, output_dir)
        write_download_log(
            output_dir, "forecasts", task.model, task.start_date, task.end_date, rows
        )
        write_attribution(
            output_dir,
            "forecasts",
            "open-meteo-historical-forecast",
            loc.lat,
            loc.lon,
            task.model,
            task.start_date,
            task.end_date,
        )
        return rows

    if task.kind == "single_runs":
        if not task.run_time:
            raise ValueError(f"Task {task.task_id} requires run_time")
        run_time = datetime.fromisoformat(task.run_time)
        response = fetch_single_run(
            loc.lat,
            loc.lon,
            run_time,
            task.variables,
            model=None if task.model == "ecmwf_ifs_single" else task.model,
            forecast_hours=task.forecast_hours,
            past_hours=task.past_hours,
        )
        distance_km = _validate_response_location(response, task)
        df = json_to_forecast_df(
            response,
            run_time,
            task.model,
            task.variables,
            **_location_kwargs(task, distance_km),
        )
        rows = write_forecasts(df, output_dir)
        write_download_log(
            output_dir, "forecasts", task.model, task.start_date, task.end_date, rows
        )
        write_attribution(
            output_dir,
            "forecasts",
            "open-meteo-single-runs",
            loc.lat,
            loc.lon,
            task.model,
            task.start_date,
            task.end_date,
        )
        return rows

    raise ValueError(f"Unknown task kind: {task.kind}")


def _location_kwargs(task: DownloadTask, distance_km: float) -> dict[str, Any]:
    loc = task.location
    return {
        "location_id": loc.slug,
        "location_name": loc.name,
        "requested_latitude": loc.lat,
        "requested_longitude": loc.lon,
        "location_distance_km": distance_km,
        "location_extra": {
            "requested_timezone": loc.timezone,
        },
    }


def _validate_response_location(response: dict[str, Any], task: DownloadTask) -> float:
    """Ensure the provider returned a grid point near the requested location."""
    loc = task.location
    try:
        returned_lat = float(response["latitude"])
        returned_lon = float(response["longitude"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            f"Task {task.task_id} response missing usable latitude/longitude"
        ) from exc

    distance_km = haversine_km(loc.lat, loc.lon, returned_lat, returned_lon)
    if distance_km > loc.max_grid_distance_km:
        raise ValueError(
            "Refusing to write downloaded data: provider returned a grid point too far "
            "from the requested location; "
            f"task_id={task.task_id}, requested=({loc.lat}, {loc.lon}), "
            f"returned=({returned_lat}, {returned_lon}), "
            f"distance_km={distance_km:.1f}, "
            f"max_allowed_km={loc.max_grid_distance_km:.1f}"
        )
    return distance_km


def _as_day0_previous_runs_response(response: dict, variables: tuple[str, ...]) -> dict:
    """Wrap a stitched historical forecast response as day-0 fixed lead data."""
    hourly = dict(response.get("hourly", {}))
    for variable in variables:
        if variable in hourly:
            hourly[f"{variable}_previous_day0"] = hourly[variable]
    wrapped = dict(response)
    wrapped["hourly"] = hourly
    return wrapped
