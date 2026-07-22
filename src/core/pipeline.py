"""Orchestration layer for the unified weather data pipeline.

This module ties together the historical (v2) and forecast (docker) subsystems:
- Data ingestion: fetch forecasts and observations from APIs
- Canonical transformation: normalize to shared coordinate system
- Parquet storage: partitioned, versioned writes
- Verification: validate forecasts against observations
- Analysis: metrics, anomalies, extremes, seasonal patterns

The orchestration layer operates at a high level:
1. Schedule runs (hourly/daily)
2. Trigger ingestion (historical + forecast)
3. Run canonical transformation
4. Store to parquet partitions
5. Run verification and validation
6. Generate reports and metrics

This module does NOT contain business logic — it coordinates
existing subsystems from src/historical and src/forecast.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from historical.api import fetch_observations, fetch_single_run
from historical.normalize import json_to_forecast_df, json_to_observation_df
from historical.storage import (
    write_attribution,
    write_download_log,
    write_forecasts,
    write_observations,
)

logger = logging.getLogger(__name__)


SUPPORTED_SINGLE_RUN_MODELS = {"ecmwf_ifs_single"}


class PipelineOrchestrator:
    """High-level orchestration of the unified pipeline.

    Coordinates between historical (v2) and forecast (docker) subsystems
    without duplicating business logic.

    Usage:
        orch = PipelineOrchestrator(output_dir="./data")
        result = orch.run_ingestion(
            location={"name": "Copenhagen", "lat": 55.605, "lon": 12.574},
            start_date="2024-01-01",
            end_date="2024-01-07",
            variables=["temperature_2m", "precipitation"],
            models=["ecmwf_ifs_single"],
        )
    """

    def __init__(self, output_dir: str | Path) -> None:
        self.output_dir = Path(output_dir)
        # Ensure output directories exist
        for subdir in ["raw/observations", "raw/forecasts", "intermediate", "cache"]:
            (self.output_dir / subdir).mkdir(parents=True, exist_ok=True)

    def run_ingestion(
        self,
        location: dict[str, Any],
        start_date: str,
        end_date: str,
        variables: tuple[str, ...] = ("temperature_2m", "precipitation"),
        models: list[str] | None = None,
        ingest_observations: bool = True,
    ) -> dict[str, Any]:
        """Run a complete ingestion cycle for observations and forecasts.

        Args:
            location: Dict with name, lat, lon.
            start_date: Start date (YYYY-MM-DD).
            end_date: End date (YYYY-MM-DD).
            variables: Tuple of variable names to fetch.
            models: List of model names for forecast ingestion.
            ingest_observations: Whether to fetch observations in this run.

        Returns:
            Dict with row counts, models ingested, and timing info.
        """
        models = models or []
        result: dict[str, Any] = {
            "forecast_rows": 0,
            "observation_rows": 0,
            "models_ingested": [],
            "errors": [],
            "started_at": datetime.now(UTC).isoformat(),
        }

        if ingest_observations:
            self._ingest_observations(location, start_date, end_date, variables, result)

        for model in models:
            self._ingest_forecast_model(location, start_date, end_date, variables, model, result)

        result["completed_at"] = datetime.now(UTC).isoformat()
        return result

    def _ingest_observations(
        self,
        location: dict[str, Any],
        start_date: str,
        end_date: str,
        variables: tuple[str, ...],
        result: dict[str, Any],
    ) -> None:
        """Fetch archive observations and write them to parquet."""
        try:
            obs_response = fetch_observations(
                location["lat"],
                location["lon"],
                start_date,
                end_date,
                variables,
                max_retries=3,
                retry_delay=1.0,
            )
            obs_df = json_to_observation_df(obs_response, variables)
            obs_rows = write_observations(obs_df, self.output_dir)
            write_download_log(
                self.output_dir,
                "observations",
                None,
                start_date,
                end_date,
                rows_downloaded=obs_rows,
            )
            write_attribution(
                self.output_dir,
                "observations",
                "open-meteo-archive",
                latitude=location["lat"],
                longitude=location["lon"],
                start_date=start_date,
                end_date=end_date,
            )
            result["observation_rows"] = obs_rows
            logger.info("Observations: %d rows", obs_rows)
        except Exception as e:
            logger.error("Observation ingestion failed: %s", e)
            result["errors"].append({"type": "observations", "error": str(e)})

    def _ingest_forecast_model(
        self,
        location: dict[str, Any],
        start_date: str,
        end_date: str,
        variables: tuple[str, ...],
        model: str,
        result: dict[str, Any],
    ) -> None:
        """Fetch archived forecast runs and write them to parquet."""
        if model not in SUPPORTED_SINGLE_RUN_MODELS:
            error = (
                f"Model '{model}' is not supported by Open-Meteo Single Runs; "
                f"supported models: {', '.join(sorted(SUPPORTED_SINGLE_RUN_MODELS))}"
            )
            logger.error("Forecast %s ingestion failed: %s", model, error)
            result["errors"].append({"type": f"forecast_{model}", "error": error})
            return

        rows_for_model = 0
        for issue_time in self._forecast_issue_times(start_date, end_date):
            try:
                response = fetch_single_run(
                    location["lat"],
                    location["lon"],
                    issue_time,
                    variables,
                    max_retries=3,
                    retry_delay=1.0,
                )
                df = json_to_forecast_df(response, issue_time, model, variables)
                forecast_rows = write_forecasts(df, self.output_dir)
                rows_for_model += forecast_rows
                result["forecast_rows"] += forecast_rows
                write_download_log(
                    self.output_dir,
                    "forecasts",
                    model,
                    issue_time.date().isoformat(),
                    issue_time.date().isoformat(),
                    rows_downloaded=forecast_rows,
                )
                logger.info(
                    "Forecast %s run %s: %d rows",
                    model,
                    issue_time.isoformat(),
                    forecast_rows,
                )
            except Exception as e:
                logger.error(
                    "Forecast %s run %s ingestion failed: %s", model, issue_time.isoformat(), e
                )
                result["errors"].append(
                    {
                        "type": f"forecast_{model}",
                        "run": issue_time.isoformat(),
                        "error": str(e),
                    }
                )

        if rows_for_model > 0:
            result["models_ingested"].append(model)
            write_attribution(
                self.output_dir,
                "forecasts",
                "open-meteo-single-runs",
                latitude=location["lat"],
                longitude=location["lon"],
                model=model,
                start_date=start_date,
                end_date=end_date,
            )

    @staticmethod
    def _forecast_issue_times(start_date: str, end_date: str) -> list[datetime]:
        """Generate UTC midnight issue times for each requested date."""
        start_dt = datetime.fromisoformat(start_date).replace(tzinfo=UTC)
        end_dt = datetime.fromisoformat(end_date).replace(tzinfo=UTC)
        if end_dt < start_dt:
            raise ValueError(f"end_date {end_date!r} must be on or after start_date {start_date!r}")

        issue_times = []
        current = start_dt
        while current <= end_dt:
            issue_times.append(current)
            current += timedelta(days=1)
        return issue_times

    def run_verification(
        self,
        location: dict[str, Any],
        start_date: str,
        end_date: str,
        models: list[str],
    ) -> dict[str, Any]:
        """Run verification for a set of models against observations.

        Uses v2 verification logic to compare forecast models
        against observations.
        """
        from verification.aligner import ForecastAligner
        from verification.validator import (
            ForecastValidator,
        )

        validator = ForecastValidator(
            location=location,
            start_date=start_date,
            end_date=end_date,
        )
        aligner = ForecastAligner(output_dir=self.output_dir)

        results = {}
        for model in models:
            try:
                aligned = aligner.align(model)
                validation = validator.validate(aligned)
                results[model] = {
                    "aligned_rows": len(aligned) if aligned is not None else 0,
                    "errors": [str(e) for e in validation] if validation else [],
                }
            except Exception as e:
                logger.error("Verification failed for %s: %s", model, e)
                results[model] = {"error": str(e)}

        return results

    def run_analysis(
        self,
        location: dict[str, Any],
        start_date: str,
        end_date: str,
        variable: str = "temperature_2m",
    ) -> dict[str, Any]:
        """Run analysis pipeline on historical data.

        Uses v2 analysis modules for anomalies, extremes,
        and seasonal pattern detection.
        """
        from analysis.anomalies import detect_anomalies
        from analysis.extremes import detect_extremes
        from analysis.seasonal_patterns import detect_seasonal_patterns

        results = {}

        try:
            anomalies = detect_anomalies(
                location,
                variable,
                start_date,
                end_date,
                output_dir=self.output_dir,
            )
            results["anomalies"] = len(anomalies) if anomalies else 0
        except Exception as e:
            logger.error("Anomaly detection failed: %s", e)
            results["anomalies"] = {"error": str(e)}

        try:
            extremes = detect_extremes(
                location,
                variable,
                start_date,
                end_date,
                output_dir=self.output_dir,
            )
            results["extremes"] = len(extremes) if extremes else 0
        except Exception as e:
            logger.error("Extreme detection failed: %s", e)
            results["extremes"] = {"error": str(e)}

        try:
            patterns = detect_seasonal_patterns(
                location,
                variable,
                start_date,
                end_date,
                output_dir=self.output_dir,
            )
            results["seasonal_patterns"] = len(patterns) if patterns else 0
        except Exception as e:
            logger.error("Seasonal pattern detection failed: %s", e)
            results["seasonal_patterns"] = {"error": str(e)}

        return results


# ===========================================================================
# Public API — top-level entry points for end users
# ===========================================================================


def download_historical(
    location: dict[str, Any],
    start_date: str,
    end_date: str,
    variables: tuple[str, ...] = ("temperature_2m", "precipitation"),
    output_dir: str | Path = "data",
) -> dict[str, Any]:
    """Download historical observations to parquet.

    Wrapper around v2 observation ingestion logic.
    Does NOT contain ingestion business logic — delegates to
    historical.api and historical.storage.

    Args:
        location: Dict with name, lat, lon.
        start_date: Start date (YYYY-MM-DD).
        end_date: End date (YYYY-MM-DD).
        variables: Tuple of variable names to fetch.
        output_dir: Destination data directory.

    Returns:
        Dict with row counts, errors, and timing.
    """
    orch = PipelineOrchestrator(output_dir)
    return orch.run_ingestion(
        location=location,
        start_date=start_date,
        end_date=end_date,
        variables=variables,
        models=[],
    )


def download_forecast(
    location: dict[str, Any],
    start_date: str,
    end_date: str,
    variables: tuple[str, ...] = ("temperature_2m", "precipitation"),
    models: list[str] | None = None,
    output_dir: str | Path = "data",
) -> dict[str, Any]:
    """Download forecast data from API providers to parquet.

    Wrapper around archived forecast run ingestion.
    Does NOT contain ingestion business logic — delegates to
    historical.api, historical.normalize, and historical.storage.

    Args:
        location: Dict with name, lat, lon.
        start_date: Start date (YYYY-MM-DD).
        end_date: End date (YYYY-MM-DD).
        variables: Tuple of variable names to fetch.
        models: List of model names for forecast ingestion.
        output_dir: Destination data directory.

    Returns:
        Dict with row counts, models ingested, and timing.
    """
    orch = PipelineOrchestrator(output_dir)
    return orch.run_ingestion(
        location=location,
        start_date=start_date,
        end_date=end_date,
        variables=variables,
        models=models or ["ecmwf_ifs_single"],
        ingest_observations=False,
    )


def build_dataset(
    location: dict[str, Any],
    start_date: str,
    end_date: str,
    variables: tuple[str, ...] = ("temperature_2m", "precipitation"),
    models: list[str] | None = None,
    run_verification: bool = True,
    run_analysis: bool = True,
    output_dir: str | Path = "data",
) -> dict[str, Any]:
    """Build a complete unified dataset.

    Orchestrates the full pipeline:
    1. Download historical observations (v2)
    2. Download forecasts (docker)
    3. Optionally run verification
    4. Optionally run analysis

    Does NOT contain ingestion or verification logic — delegates
    to underlying modules.

    Args:
        location: Dict with name, lat, lon.
        start_date: Start date (YYYY-MM-DD).
        end_date: End date (YYYY-MM-DD).
        variables: Tuple of variable names.
        models: List of forecast models.
        run_verification: Whether to run verification after ingestion.
        run_analysis: Whether to run analysis after ingestion.
        output_dir: Destination data directory.

    Returns:
        Dict with all pipeline results.
    """
    orch = PipelineOrchestrator(output_dir)
    result: dict[str, Any] = {}
    models = models or ["ecmwf_ifs_single"]

    # Step 1 & 2: Ingestion (observations + forecasts)
    result["ingestion"] = orch.run_ingestion(
        location=location,
        start_date=start_date,
        end_date=end_date,
        variables=variables,
        models=models,
    )

    # Step 3: Verification (optional)
    if run_verification:
        result["verification"] = orch.run_verification(
            location=location,
            start_date=start_date,
            end_date=end_date,
            models=models,
        )

    # Step 4: Analysis (optional)
    if run_analysis:
        for var in variables[:1]:  # analyze first variable
            result["analysis"] = orch.run_analysis(
                location=location,
                start_date=start_date,
                end_date=end_date,
                variable=var,
            )
            break  # only one analysis call for now

    return result


class ScheduledPipeline:
    """Pipeline with cron-style scheduling.

    Supports:
    - Hourly forecast ingestion (4 runs/day × N models)
    - Daily observation sync
    - Weekly verification cycles
    - Monthly analysis reports
    """

    def __init__(self, output_dir: str | Path, schedule: dict[str, str]) -> None:
        self.output_dir = Path(output_dir)
        self.schedule = schedule  # e.g., {"forecast": "0 */6 * * *", "obs": "0 0 * * *"}
        self.orchestrator = PipelineOrchestrator(output_dir)

    def run_now(self, trigger: str) -> dict[str, Any]:
        """Run a specific pipeline trigger immediately."""
        if trigger == "forecast":
            return self.orchestrator.run_ingestion(
                location=self._get_location(),
                start_date="2024-01-01",
                end_date=datetime.now(UTC).strftime("%Y-%m-%d"),
                variables=("temperature_2m", "precipitation"),
                models=["ecmwf_ifs_single"],
                ingest_observations=False,
            )
        elif trigger == "observations":
            return self.orchestrator.run_ingestion(
                location=self._get_location(),
                start_date="2024-01-01",
                end_date=datetime.now(UTC).strftime("%Y-%m-%d"),
                variables=("temperature_2m", "precipitation"),
                models=[],
            )
        elif trigger == "verification":
            return self.orchestrator.run_verification(
                location=self._get_location(),
                start_date="2024-01-01",
                end_date=datetime.now(UTC).strftime("%Y-%m-%d"),
                models=["ecmwf_ifs_single"],
            )
        elif trigger == "analysis":
            return self.orchestrator.run_analysis(
                location=self._get_location(),
                start_date="2024-01-01",
                end_date=datetime.now(UTC).strftime("%Y-%m-%d"),
                variable="temperature_2m",
            )
        else:
            raise ValueError(f"Unknown trigger: {trigger}")

    def _get_location(self) -> dict[str, Any]:
        """Get location config. In production, this reads from config."""
        return {"name": "Copenhagen", "lat": 55.605, "lon": 12.574}
