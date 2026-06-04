"""Phase 3: ERA5 observation provider via Open-Meteo Archive API.

Fetches hourly ERA5 reanalysis data and normalizes it into
CanonicalForecastRecord format with:
  - is_observation = True
  - model = "ERA5"
  - provider = "archive"

Observations align exactly to valid_time axis.
"""


from datetime import datetime
from typing import Any

import httpx
from meteo.transport.http_client import HTTPClient

from ..canonical.models import CanonicalForecastRecord


class ObservationProviderError(Exception):
    """Raised when observation fetching fails."""


class ObservationProvider:
    """Fetch ERA5 observations via Open-Meteo Archive API.

    Usage:
        provider = ObservationProvider()
        records = provider.fetch(
            latitude=55.6, longitude=12.6,
            start_date="2023-01-01", end_date="2023-01-07",
            variables=["temperature_2m"],
        )
    """

    ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/era5"

    def __init__(self, client: HTTPClient | None = None) -> None:
        self.client = client or HTTPClient()

    def fetch(
        self,
        latitude: float,
        longitude: float,
        start_date: str,
        end_date: str,
        variables: list[str],
        hourly_units: dict[str, str] | None = None,
    ) -> list[CanonicalForecastRecord]:
        """Fetch ERA5 observations for a single location.

        Args:
            latitude: Latitude coordinate
            longitude: Longitude coordinate
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            variables: List of variable names to fetch
            hourly_units: Optional units dict (auto-detected if omitted)

        Returns:
            List of CanonicalForecastRecord with is_observation=True
        """
        params: dict[str, Any] = {
            "latitude": latitude,
            "longitude": longitude,
            "start_date": start_date,
            "end_date": end_date,
            "hourly": ",".join(variables),
            "timezone": "UTC",
        }

        try:
            response = self.client.get(self.ARCHIVE_URL, params=params)
            response.raise_for_status()
        except httpx.HTTPError as e:
            raise ObservationProviderError(f"Failed to fetch ERA5 data: {e}") from e

        data = response.json()

        if not isinstance(data, dict):
            raise ObservationProviderError("Expected JSON object, got different type")

        return self._normalize_response(
            data,
            latitude=latitude,
            longitude=longitude,
        )

    def _normalize_response(
        self,
        response_data: dict[str, Any],
        latitude: float,
        longitude: float,
    ) -> list[CanonicalForecastRecord]:
        """Convert raw archive API response to CanonicalForecastRecord."""
        records: list[CanonicalForecastRecord] = []

        hourly = response_data.get("hourly", {})
        if not isinstance(hourly, dict):
            raise ObservationProviderError("Expected 'hourly' dict in archive response")

        times = hourly.get("time", [])
        if not times:
            raise ObservationProviderError("No time data in archive response")

        # Get units if available
        hourly_units = response_data.get("hourly_units", {})
        if not isinstance(hourly_units, dict):
            hourly_units = {}

        # Process each variable
        for var_name, var_values in hourly.items():
            if var_name == "time":
                continue
            if not isinstance(var_values, list):
                continue

            unit = hourly_units.get(var_name)

            for i, value in enumerate(var_values):
                if i >= len(times):
                    break

                # Parse timestamp
                ts = times[i]
                if isinstance(ts, str):
                    dt = datetime.fromisoformat(ts)
                elif isinstance(ts, (int, float)):
                    dt = datetime.utcfromtimestamp(ts)
                else:
                    continue

                # Normalize to UTC-naive
                if dt.tzinfo is not None:
                    dt = dt.replace(tzinfo=None)

                # ERA5 observations have run_time = valid_time (no forecast horizon)
                record = CanonicalForecastRecord(
                    provider="archive",
                    model="ERA5",
                    run_time=dt,  # For observations, run_time = valid_time
                    valid_time=dt,
                    lead_hours=0,
                    latitude=latitude,
                    longitude=longitude,
                    variable=var_name,
                    value=value,
                    unit=unit,
                    is_observation=True,
                    is_reanalysis=True,
                    raw_ref=f"era5:{latitude},{longitude},{var_name},{ts}",
                    metadata={
                        "observation_source": "era5",
                        "archive_response": True,
                        "is_reanalysis": True,
                        "provenance": {
                            "type": "reanalysis",
                            "product": "ERA5",
                            "provider": "ECMWF via Open-Meteo",
                            "resolution": "0.1 deg hourly",
                            "description": (
                                "ERA5 reanalysis is a global atmospheric "
                                "reanalysis product. It is NOT ground truth — "
                                "it has its own biases and resolution limits. "
                                "For surface validation, prefer in-situ "
                                "observations (station, METAR, SYNOP)."
                            ),
                        },
                    },
                )
                records.append(record)

        return records

    def fetch_for_records(
        self,
        forecast_records: list[CanonicalForecastRecord],
        variables: list[str] | None = None,
    ) -> list[CanonicalForecastRecord]:
        """Fetch ERA5 observations matching the time range and location
        of existing forecast records.

        Args:
            forecast_records: Existing forecast records to match against
            variables: Variables to fetch (defaults to all in forecasts)

        Returns:
            List of ERA5 observation records aligned to forecast times
        """
        if not forecast_records:
            return []

        # Determine unique locations and time range
        locations: set[tuple[float, float]] = set()
        min_time = None
        max_time = None

        for rec in forecast_records:
            locations.add((rec.latitude, rec.longitude))
            if min_time is None or rec.valid_time < min_time:
                min_time = rec.valid_time
            if max_time is None or rec.valid_time > max_time:
                max_time = rec.valid_time

        # Determine variables
        if variables is None:
            variables = list({rec.variable for rec in forecast_records})

        # Fetch observations for each location
        all_observations: list[CanonicalForecastRecord] = []

        for lat, lon in locations:
            start_date = min_time.strftime("%Y-%m-%d")
            end_date = max_time.strftime("%Y-%m-%d")

            try:
                obs = self.fetch(
                    latitude=lat,
                    longitude=lon,
                    start_date=start_date,
                    end_date=end_date,
                    variables=variables,
                )
                all_observations.extend(obs)
            except ObservationProviderError:
                # Log but continue — missing observations are explicit gaps
                pass

        return all_observations
