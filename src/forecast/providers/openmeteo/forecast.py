"""Open-Meteo Forecast API adapter.

The Forecast API provides operational forecasts from multiple models.
Unlike Single Runs, this endpoint supports model selection and
multiple models per request.
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

FORECAST_ENDPOINT = "/v1/forecast"


class OpenMeteoForecastProvider(BaseProvider):
    """Open-Meteo Forecast API adapter.

    Fetches operational forecasts from multiple models.

    Usage:
        provider = OpenMeteoForecastProvider()
        response = provider.fetch_forecast(
            lat=55.6,
            lon=12.6,
            models=["ecmwf_ifs_hres", "gfs"],
            hourly=["temperature_2m", "precipitation"],
        )
    """

    DEFAULT_MODELS: ClassVar[tuple[str, ...]] = ("ecmwf_ifs_hres", "gfs", "icon")
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
        return "openmeteo_forecast"

    @property
    def supported_endpoints(self) -> list[str]:
        return [FORECAST_ENDPOINT]

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
        endpoint = endpoint or FORECAST_ENDPOINT
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

        # Run structural validation
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

    def fetch_forecast(
        self,
        lat: float,
        lon: float,
        models: list[str] | None = None,
        hourly: list[str] | None = None,
        daily: list[str] | None = None,
        past_days: int = 0,
        timezone: str = "UTC",
    ) -> RawForecastResponse:
        """Fetch an operational forecast.

        Args:
            lat: Latitude.
            lon: Longitude.
            models: Model names (default: ECMWF IFS HRES, GFS, ICON).
            hourly: Hourly variables.
            daily: Daily variables.
            past_days: Include N days of past data for verification.
            timezone: Output timezone.
        """
        return self.fetch(
            endpoint="/v1/forecast",
            params={
                "latitude": lat,
                "longitude": lon,
                "models": models or self.DEFAULT_MODELS,
                "hourly": hourly or self.DEFAULT_HOURLY,
                "daily": daily,
                "past_days": past_days,
                "timezone": timezone,
            },
        )

    # ------------------------------------------------------------------
    # Capabilities
    # ------------------------------------------------------------------

    def capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities(
            provider=self.provider_name,
            supported_models=[
                "ecmwf_ifs_hres",
                "gfs",
                "icon",
                "icon_eu",
                "hrrr",
                "nam",
                "nbm",
                "ukmo_global",
                "ukmo_ukv",
                "gem_global",
                "aifs_0p25_single",
            ],
            supported_variables=[
                "temperature_2m",
                "dew_point_2m",
                "relative_humidity_2m",
                "precipitation",
                "wind_speed_10m",
                "wind_direction_10m",
                "pressure_msl",
                "cloud_cover",
            ],
            supports_hourly=True,
            supports_daily=True,
            supports_multi_location=True,
            supports_historical_runs=False,
            max_forecast_hours=384,
            timezone_behavior="auto",
            null_behavior={
                "unsupported_model": "returns_error_in_locations array",
                "description": "Forecast API reports unsupported models in the "
                "response structure (not silently)",
            },
        )
