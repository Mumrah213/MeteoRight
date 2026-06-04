"""Open-Meteo Single Runs API adapter.

The Single Runs API provides archived forecast runs — essentially
"what the model thought would happen" at some past date. This is
critical for forecast verification.

KNOWN BEHAVIOR:
- ONLY supports the ECMWF IFS model (single_runs endpoint)
- The `models` parameter fails silently — requesting other models
  returns HTTP 200 with empty or all-null arrays
- Hourly data only
- UTC timestamps
- Archived forecast runs from past dates

CRITICAL: HTTP 200 DOES NOT MEAN VALID DATA.
You MUST validate the response to detect silent model failures.
"""


import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ...base_provider import BaseProvider
from ...http_client import HttpClient
from ...schemas.capabilities import ProviderCapabilities
from ...schemas.errors import (
    MeteoProviderError,
    NullClassification,
    ProviderError,
    ProviderErrorKind,
)
from ...schemas.forecast_raw import RawForecastResponse, RawLocationData
from ...schemas.validation import ValidationResult
from ...utils.logging import timed_request
from ...validators.provider_specific import OpenMeteoValidator
from ...validators.structural import StructuralValidator

logger = logging.getLogger(__name__)

# Single Runs API endpoint (hardcoded — this is the only supported endpoint)
SINGLE_RUNS_ENDPOINT = "/v1/forecast"

# The ONLY model that works with single runs
SUPPORTED_MODEL = "ecmwf_ifs_single"


class OpenMeteoSingleRunsProvider(BaseProvider):
    """Open-Meteo Single Runs API adapter.

    Fetches archived forecast runs for verification against observations.

    Usage:
        provider = OpenMeteoSingleRunsProvider()
        response = provider.fetch(
            endpoint="/v1/forecast",
            params={
                "latitude": 55.6,
                "longitude": 12.6,
                "start_date": "2024-01-01",
                "end_date": "2024-01-01",
                "hourly": ["temperature_2m", "precipitation"],
            }
        )
    """

    @property
    def provider_name(self) -> str:
        return "openmeteo_single_runs"

    @property
    def supported_endpoints(self) -> list[str]:
        return [SINGLE_RUNS_ENDPOINT]

    def __init__(self, client: HttpClient | None = None, **kwargs: Any) -> None:
        super().__init__(client=client, **kwargs)
        # Single Runs defaults
        self._default_hourly = [
            "temperature_2m",
            "dew_point_2m",
            "precipitation",
            "wind_speed_10m",
            "wind_direction_10m",
            "wind_gusts_10m",
            "pressure_msl",
            "cloud_cover",
            "relative_humidity_2m",
            "apparent_temperature",
        ]

    # ------------------------------------------------------------------
    # Core fetch implementation
    # ------------------------------------------------------------------

    def fetch(
        self,
        endpoint: str,
        params: dict[str, Any],
        raw_dir: Path | str | None = None,
    ) -> RawForecastResponse:
        """Fetch a single forecast run.

        Pipeline:
        1. Ensure model is ECMWF IFS (Single Runs only supports this)
        2. Request data via transport layer
        3. Persist raw response
        4. Parse into typed schema
        5. Run structural + semantic + provider-specific validation
        6. Return RawForecastResponse
        """
        endpoint = endpoint or SINGLE_RUNS_ENDPOINT

        # Default to ECMWF IFS single — Single Runs only supports this
        params.setdefault("models", "ecmwf_ifs_single")

        response, validation = self._fetch_with_validation(endpoint, params)

        # Run additional provider-specific validation
        validation_issues = OpenMeteoValidator.pipeline().check(response)
        for issue in validation_issues:
            validation.add_issue(issue)

        # Persist raw response
        self.save_raw_response(response, raw_dir)

        return response

    def _fetch_with_validation(
        self,
        endpoint: str,
        params: dict[str, Any],
    ) -> tuple[RawForecastResponse, ValidationResult]:
        """Full fetch pipeline with error handling."""

        # Set defaults for single runs
        params.setdefault("timezone", "UTC")
        params.setdefault("past_days", 0)

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

        # Check HTTP status
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

        # Parse JSON
        try:
            raw_json = http_resp.json
        except Exception as exc:
            raise MeteoProviderError(
                ProviderError(
                    kind=ProviderErrorKind.INVALID_JSON,
                    message=f"Failed to parse response as JSON: {exc}",
                    provider=self.provider_name,
                    endpoint=endpoint,
                    parameters=params,
                    raw_response={
                        "status_code": http_resp.status_code,
                        "content_preview": http_resp.content[:200],
                    },
                )
            ) from exc

        # Parse into typed schema
        locations = []
        for loc_data in raw_json.get("locations", []):
            loc = RawLocationData(**loc_data)
            locations.append(loc)

        # If no locations array, try single location format
        if not locations and "latitude" in raw_json:
            locations = [RawLocationData(**raw_json)]

        # Build response object
        response = RawForecastResponse(
            locations=locations,
            request_params=params,
            request_timestamp=datetime.now(UTC).isoformat(),
            provider=self.provider_name,
            endpoint=endpoint,
            raw_json=raw_json,
        )

        # Run structural and semantic validation
        result = ValidationResult(
            valid=True,
            provider=self.provider_name,
            endpoint=endpoint,
        )

        # Structural checks
        structural_issues = StructuralValidator.pipeline().check(response)
        for issue in structural_issues:
            result.add_issue(issue)
        result.rules_passed += len(StructuralValidator.pipeline()._validators)

        # Semantic checks (basic)
        self._check_semantic_values(response)

        # Check for data presence
        has_any_data = False
        for loc in response.locations:
            if loc.hourly and any(
                isinstance(v, list) and len(v) > 0 and any(x is not None for x in v)
                for k, v in loc.hourly.items()
                if k not in ("time", "time_utc")
            ):
                has_any_data = True

        if not has_any_data:
            result.valid = False
            result.null_classification = NullClassification.ALL_NULL_RESPONSE.name

        return response, result

    # ------------------------------------------------------------------
    # Convenience methods
    # ------------------------------------------------------------------

    def fetch_single_run(
        self,
        lat: float,
        lon: float,
        date: str,
        hourly: list[str] | None = None,
    ) -> RawForecastResponse:
        """Fetch a single forecast run for a specific date.

        Args:
            lat: Latitude.
            lon: Longitude.
            date: Date string (YYYY-MM-DD) — the forecast was issued on this date.
            hourly: Hourly variables to fetch. Defaults to standard set.

        Example:
            # Get the ECMWF forecast that was issued on 2024-01-01
            resp = provider.fetch_single_run(55.6, 12.6, "2024-01-01")
        """
        return self.fetch(
            endpoint="/v1/forecast",
            params={
                "latitude": lat,
                "longitude": lon,
                "start_date": date,
                "end_date": date,
                "hourly": hourly or self._default_hourly,
                "timezone": "UTC",
            },
        )

    # ------------------------------------------------------------------
    # Capabilities
    # ------------------------------------------------------------------

    def capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities(
            provider=self.provider_name,
            supported_models=["ecmwf_ifs_single"],
            supported_variables=self._default_hourly,
            supports_hourly=True,
            supports_daily=False,
            supports_multi_location=False,
            supports_historical_runs=True,
            max_forecast_hours=240,
            timezone_behavior="utc",
            null_behavior={
                "unsupported_model": "returns_empty_or_all_null_arrays",
                "description": "Single Runs API only supports ECMWF IFS. Other models "
                "produce HTTP 200 with no data — always validate response.",
                "known_silent_failures": [
                    "ECMWF IFS model is the ONLY supported model",
                    "Other model names are silently rejected (HTTP 200, empty data)",
                    "Unsupported variables return null arrays",
                ],
            },
            known_limitations=[
                "Only supports ECMWF IFS single model",
                "Hourly data only (no daily)",
                "UTC timestamps only",
                "Model parameter silently fails for non-ECMWF models",
                "Must validate ALL responses for empty/null arrays",
            ],
        )
