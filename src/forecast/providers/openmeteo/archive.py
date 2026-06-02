"""Open-Meteo Archive (Observations) API adapter.

The Archive API provides historical weather data — actual observations
and reanalysis, not forecasts. This is used as the ground truth
for forecast verification.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, ClassVar

from ...base_provider import BaseProvider
from ...http_client import HTTPClient
from ...schemas.capabilities import ProviderCapabilities
from ...schemas.errors import MeteoProviderError, ProviderError, ProviderErrorKind
from ...schemas.forecast_raw import RawForecastResponse, RawLocationData
from ...schemas.validation import ValidationResult
from ...utils.logging import timed_request
from ...validators.structural import StructuralValidator

logger = logging.getLogger(__name__)

ARCHIVE_ENDPOINT = "/v1/forecast"  # Same endpoint, different params


class OpenMeteoArchiveProvider(BaseProvider):
    """Open-Meteo Archive API adapter.

    Fetches historical weather data for verification against forecasts.

    Usage:
        provider = OpenMeteoArchiveProvider()
        response = provider.fetch_archive(
            lat=55.6,
            lon=12.6,
            start_date="2024-01-01",
            end_date="2024-01-07",
        )
    """

    DEFAULT_MODELS: ClassVar[tuple[str, ...]] = ("era5", "era5_hourly")
    DEFAULT_HOURLY: ClassVar[tuple[str, ...]] = (
        "temperature_2m",
        "dew_point_2m",
        "relative_humidity_2m",
        "precipitation",
        "wind_speed_10m",
        "wind_direction_10m",
        "pressure_msl",
    )

    @property
    def provider_name(self) -> str:
        return "openmeteo_archive"

    @property
    def supported_endpoints(self) -> list[str]:
        return [ARCHIVE_ENDPOINT]

    def __init__(self, client: HTTPClient | None = None, **kwargs: Any) -> None:
        super().__init__(client=client, **kwargs)

    # ------------------------------------------------------------------
    # Core fetch implementation
    # ------------------------------------------------------------------

    def fetch(
        self,
        endpoint: str,
        params: dict[str, Any],
        raw_dir: Path | str | None = None,
    ) -> RawForecastResponse:
        endpoint = endpoint or ARCHIVE_ENDPOINT
        params.setdefault("models", self.DEFAULT_MODELS)

        with timed_request(self.provider_name, endpoint, params):
            try:
                http_resp = self.client.get(endpoint, params)
            except Exception as exc:
                raise MeteoProviderError(
                    ProviderError(
                        kind=ProviderErrorKind.CONNECTION_TIMEOUT,
                        message=f"Request failed: {exc}",
                        provider=self.provider_name,
                        endpoint=endpoint,
                        parameters=params,
                    )
                ) from exc

        if not http_resp.ok:
            raise MeteoProviderError(
                ProviderError(
                    kind=ProviderErrorKind.HTTP_ERROR,
                    message=f"HTTP {http_resp.status_code}: {http_resp.text[:200]}",
                    provider=self.provider_name,
                    endpoint=endpoint,
                    parameters=params,
                    http_status_code=http_resp.status_code,
                )
            )

        try:
            raw_json = http_resp.json
        except Exception as exc:
            raise MeteoProviderError(
                ProviderError(
                    kind=ProviderErrorKind.INVALID_JSON,
                    message=f"Failed to parse JSON: {exc}",
                    provider=self.provider_name,
                    endpoint=endpoint,
                    parameters=params,
                )
            ) from exc

        return self._parse_response(raw_json, params, endpoint)

    def _parse_response(
        self,
        raw_json: dict[str, Any],
        params: dict[str, Any],
        endpoint: str,
    ) -> RawForecastResponse:
        locations = []
        for loc_data in raw_json.get("locations", []):
            locations.append(RawLocationData(**loc_data))

        if not locations and "latitude" in raw_json:
            locations = [RawLocationData(**raw_json)]

        response = RawForecastResponse(
            locations=locations,
            request_params=params,
            request_timestamp=datetime.now(UTC).isoformat(),
            provider=self.provider_name,
            endpoint=endpoint,
            raw_json=raw_json,
        )

        result = ValidationResult(
            valid=True,
            provider=self.provider_name,
            endpoint=endpoint,
        )
        structural_issues = StructuralValidator.pipeline().check(response)
        for issue in structural_issues:
            result.add_issue(issue)

        self.save_raw_response(response)
        return response

    # ------------------------------------------------------------------
    # Convenience methods
    # ------------------------------------------------------------------

    def fetch_archive(
        self,
        lat: float,
        lon: float,
        start_date: str,
        end_date: str,
        models: list[str] | None = None,
        hourly: list[str] | None = None,
        timezone: str = "UTC",
    ) -> RawForecastResponse:
        """Fetch historical weather data.

        Args:
            lat: Latitude.
            lon: Longitude.
            start_date: Start date (YYYY-MM-DD).
            end_date: End date (YYYY-MM-DD).
            models: Model names (default: ERA5).
            hourly: Hourly variables.
            timezone: Output timezone.
        """
        return self.fetch(
            endpoint="/v1/forecast",
            params={
                "latitude": lat,
                "longitude": lon,
                "start_date": start_date,
                "end_date": end_date,
                "models": models or self.DEFAULT_MODELS,
                "hourly": hourly or self.DEFAULT_HOURLY,
                "timezone": timezone,
            },
        )

    # ------------------------------------------------------------------
    # Capabilities
    # ------------------------------------------------------------------

    def capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities(
            provider=self.provider_name,
            supported_models=["era5", "era5_hourly", "era5_vertical_level"],
            supported_variables=[
                "temperature_2m",
                "dew_point_2m",
                "relative_humidity_2m",
                "precipitation",
                "wind_speed_10m",
                "wind_direction_10m",
                "pressure_msl",
            ],
            supports_hourly=True,
            supports_daily=True,
            supports_multi_location=True,
            supports_historical_runs=False,
            max_forecast_hours=None,  # Archive is historical only
            timezone_behavior="auto",
        )
