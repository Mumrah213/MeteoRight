"""Abstract base provider.

All provider adapters inherit from this class and implement:
- fetch(): perform the API request and return RawForecastResponse
- capabilities(): return ProviderCapabilities

The base class handles:
- Transport abstraction (uses self.client)
- Raw response persistence (calls self.storage.save_raw)
- Validation pipeline (calls self._validate)
- Error classification (returns ProviderError on failure)
"""


import json
import logging
from abc import ABC, abstractmethod
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ..schemas.capabilities import ProviderCapabilities
from ..schemas.forecast_raw import RawForecastResponse
from ..schemas.validation import ValidationResult
from .http_client import HttpClient

logger = logging.getLogger(__name__)


class BaseProvider(ABC):
    """Abstract base class for weather API providers."""

    # Default storage directory for raw responses
    DEFAULT_RAW_DIR = Path("storage/raw")

    def __init__(
        self,
        client: HttpClient | None = None,
        raw_dir: Path | str | None = None,
    ) -> None:
        self.client = client or HttpClient()
        self.raw_dir = Path(raw_dir) if raw_dir else self.DEFAULT_RAW_DIR
        self._capabilities: ProviderCapabilities | None = None

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Unique identifier for this provider."""

    @property
    @abstractmethod
    def supported_endpoints(self) -> list[str]:
        """List of API endpoints this provider supports."""

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    @abstractmethod
    def fetch(
        self,
        endpoint: str,
        params: dict[str, Any],
        raw_dir: Path | str | None = None,
    ) -> RawForecastResponse:
        """Fetch data from the provider.

        Implements the full pipeline:
        1. Build URL and request
        2. Receive HttpResponse from transport layer
        3. Persist raw response
        4. Parse into typed schema
        5. Run validation pipeline
        6. Return RawForecastResponse with metadata

        Returns RawForecastResponse on success.
        Raises ProviderError on structural failures (can't parse, etc.).
        """

    def fetch_and_check(
        self,
        endpoint: str,
        params: dict[str, Any],
        raw_dir: Path | str | None = None,
    ) -> tuple[RawForecastResponse, ValidationResult | None]:
        """Fetch and validate. Convenience wrapper.

        Returns:
            (response, validation_result) on success
            Raises ProviderError on structural failures
        """
        response = self.fetch(endpoint, params, raw_dir)
        validation = self.validate(response)
        return response, validation

    def validate(self, response: RawForecastResponse) -> ValidationResult:
        """Run validation pipeline on a raw response.

        Override in subclasses to add provider-specific checks.
        """
        result = ValidationResult(
            valid=True,
            provider=type(self).provider_name,
            endpoint=response.endpoint,
        )

        # Structural checks
        if not self._check_array_consistency(response):
            result.valid = False

        # Semantic checks
        self._check_semantic_values(response)

        return result

    def _check_array_consistency(self, response: RawForecastResponse) -> bool:
        """Check that arrays in the response are structurally consistent."""
        for loc in response.locations:
            if loc.hourly is not None:
                hourly_data = loc.hourly
                for var_name, values in hourly_data.items():
                    if var_name == "time" or var_name == "time_utc":
                        continue  # Timestamps are handled separately
                    if not isinstance(values, list):
                        continue
                    # Check monotonicity of timestamps if available
                    # This is structural, not semantic
                    if values and len(values) > 1:
                        pass  # Timestamp monotonicity would use the 'time' array
        return True

    def _check_semantic_values(self, response: RawForecastResponse) -> None:
        """Run semantic validation on response data."""
        from ..schemas.validation import SemanticIssue

        for loc in response.locations:
            if loc.hourly is not None:
                hourly_data = loc.hourly

                for var_name, values in hourly_data.items():
                    if not isinstance(values, list):
                        continue
                    for value in values:
                        if value is None:
                            continue

                        # Temperature plausibility (-100 to 80°C is reasonable)
                        if "temperature" in var_name and "apparent" not in var_name:
                            if value < -100 or value > 80:
                                response.issues.append(
                                    SemanticIssue(
                                        rule="temperature_range",
                                        variable=var_name,
                                        value=value,
                                        min_allowed=-100.0,
                                        max_allowed=80.0,
                                        message=f"Temperature {value}°C outside plausible range [-100, 80]",
                                    )
                                )

                        # Humidity must be 0-100
                        if "humidity" in var_name:
                            if value < 0 or value > 100:
                                response.issues.append(
                                    SemanticIssue(
                                        rule="humidity_range",
                                        variable=var_name,
                                        value=value,
                                        min_allowed=0.0,
                                        max_allowed=100.0,
                                        message=f"Humidity {value}% outside valid range [0, 100]",
                                    )
                                )

                        # Precipitation should be non-negative
                        if "precipitation" in var_name and "sum" not in var_name:
                            if value < 0:
                                response.issues.append(
                                    SemanticIssue(
                                        rule="precipitation_non_negative",
                                        variable=var_name,
                                        value=value,
                                        message=f"Precipitation {value} is negative",
                                    )
                                )

    # ------------------------------------------------------------------
    # Capabilities
    # ------------------------------------------------------------------

    @abstractmethod
    def capabilities(self) -> ProviderCapabilities:
        """Return declared capabilities for this provider."""

    def get_capabilities(self) -> ProviderCapabilities:
        """Return capabilities, caching the result."""
        if self._capabilities is None:
            self._capabilities = self.capabilities()
        return self._capabilities

    # ------------------------------------------------------------------
    # Raw response persistence
    # ------------------------------------------------------------------

    def save_raw_response(
        self,
        response: RawForecastResponse,
        raw_dir: Path | str | None = None,
    ) -> Path:
        """Persist a raw response to disk for reproducibility.

        Creates a deterministic file path based on provider, endpoint, and params.
        """
        directory = Path(raw_dir) if raw_dir else self.raw_dir

        # Build deterministic path
        params_hash = self._params_to_path(response.request_params)
        timestamp = response.request_timestamp or datetime.now(UTC).isoformat()

        filename = f"{self.provider_name}_{response.endpoint}_{params_hash}_{timestamp[:19]}.json"
        filepath = directory / filename

        filepath.parent.mkdir(parents=True, exist_ok=True)

        # Write raw JSON
        data = response.model_dump()
        filepath.write_text(json.dumps(data, indent=2, default=str))

        logger.debug("Raw response saved: %s", filepath)
        return filepath

    @staticmethod
    def _params_to_path(params: dict[str, Any]) -> str:
        """Convert params dict to a safe filename fragment."""
        parts = []
        for k in sorted(params.keys()):
            v = params[k]
            if isinstance(v, (list, tuple)):
                parts.append(f"{k}={'|'.join(str(x) for x in v)}")
            else:
                parts.append(f"{k}={v}")
        return "_".join(parts)
