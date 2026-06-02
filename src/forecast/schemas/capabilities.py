"""Provider capability schemas.

Every provider exposes capabilities through typed models.
Capabilities can be:
- Pre-declared (known, stable facts about a provider)
- Empirically discovered (discovered through test queries)
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class ProviderCapabilities(BaseModel):
    """Typed description of what a provider supports.

    These are NOT assumptions — they are declared capabilities that
    can be verified empirically through test queries.

    Dimensions:
    - supported_models: Which forecast models are available
    - supported_variables: Which weather variables can be requested
    - supports_hourly: Whether hourly resolution is available
    - supports_daily: Whether daily resolution is available
    - supports_multi_location: Whether multiple locations per request work
    - supports_historical_runs: Whether archived forecast runs are accessible
    - max_forecast_hours: Maximum forecast horizon in hours
    - timezone_behavior: How timezone is handled (auto, utc, fixed)
    - null_behavior: What the provider returns for unsupported queries
    - known_limitations: Documented or discovered limitations
    """

    provider: str
    supported_models: list[str] = Field(
        default_factory=list,
        description="List of model identifiers this provider supports",
    )
    supported_variables: list[str] = Field(
        default_factory=list,
        description="List of variable names this provider supports",
    )
    supports_hourly: bool = Field(default=False)
    supports_daily: bool = Field(default=False)
    supports_multi_location: bool = Field(default=False)
    supports_historical_runs: bool = Field(default=False)
    max_forecast_hours: int | None = Field(default=None)
    timezone_behavior: str = Field(
        default="auto",
        description="How the provider handles timezones: 'utc', 'auto', or fixed offset",
    )
    null_behavior: dict[str, Any] = Field(
        default_factory=dict,
        description="What the provider returns for unsupported queries (discovered empirically)",
    )
    known_limitations: list[str] = Field(
        default_factory=list,
        description="Known limitations discovered through testing or documentation",
    )

    def merge(self, other: ProviderCapabilities) -> ProviderCapabilities:
        """Merge capabilities from two sources (e.g., declared + discovered)."""
        return ProviderCapabilities(
            provider=self.provider,
            supported_models=list(set(self.supported_models) | set(other.supported_models)),
            supported_variables=list(
                set(self.supported_variables) | set(other.supported_variables)
            ),
            supports_hourly=self.supports_hourly or other.supports_hourly,
            supports_daily=self.supports_daily or other.supports_daily,
            supports_multi_location=self.supports_multi_location or other.supports_multi_location,
            supports_historical_runs=self.supports_historical_runs
            or other.supports_historical_runs,
            max_forecast_hours=max(
                self.max_forecast_hours or 0,
                other.max_forecast_hours or 0,
            ),
            timezone_behavior=self.timezone_behavior
            if self.timezone_behavior != "auto"
            else other.timezone_behavior,
            null_behavior={**self.null_behavior, **other.null_behavior},
            known_limitations=list(set(self.known_limitations) | set(other.known_limitations)),
        )

    def model_dump(self, **kwargs: Any) -> dict[str, Any]:
        """Serialize, replacing None with explicit markers."""
        data = super().model_dump(**kwargs)
        if data.get("max_forecast_hours") is None:
            data["max_forecast_hours"] = -1  # Explicitly unknown
        return data
