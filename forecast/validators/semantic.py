"""Semantic validation rules.

These check whether data values are plausible — not whether the
structure is correct. A structurally valid response can still
contain impossible values.
"""


from typing import Any

from ..schemas.forecast_raw import RawForecastResponse
from ..schemas.validation import SemanticIssue
from ..validators.base import BaseValidator


class TemperatureValidator(BaseValidator):
    """Temperature must be within physically plausible range."""

    @property
    def rule_name(self) -> str:
        return "temperature_range"

    @property
    def severity(self):
        from ..schemas.validation import IssueSeverity

        return IssueSeverity.WARNING

    def check(self, response: RawForecastResponse) -> list[SemanticIssue]:
        issues: list[SemanticIssue] = []

        for loc in response.locations:
            if loc.hourly is not None:
                self._check_var(loc.hourly, issues)

        return issues

    def _check_var(
        self,
        hourly: dict[str, Any],
        issues: list[SemanticIssue],
    ) -> None:
        for var_name, values in hourly.items():
            if "temperature" not in var_name:
                continue
            if not isinstance(values, list):
                continue

            for i, value in enumerate(values):
                if value is None:
                    continue

                # -100 to 80°C is physically plausible
                if value < -100 or value > 80:
                    issues.append(
                        SemanticIssue(
                            rule=self.rule_name,
                            variable=var_name,
                            value=value,
                            min_allowed=-100.0,
                            max_allowed=80.0,
                            message=f"Temperature {value}°C outside plausible range [-100, 80] at index {i}",
                        )
                    )


class HumidityValidator(BaseValidator):
    """Humidity must be between 0 and 100."""

    @property
    def rule_name(self) -> str:
        return "humidity_range"

    @property
    def severity(self):
        from ..schemas.validation import IssueSeverity

        return IssueSeverity.WARNING

    def check(self, response: RawForecastResponse) -> list[SemanticIssue]:
        issues: list[SemanticIssue] = []

        for loc in response.locations:
            if loc.hourly is not None:
                for var_name, values in loc.hourly.items():
                    if "humidity" not in var_name or not isinstance(values, list):
                        continue
                    for i, value in enumerate(values):
                        if value is None:
                            continue
                        if value < 0 or value > 100:
                            issues.append(
                                SemanticIssue(
                                    rule=self.rule_name,
                                    variable=var_name,
                                    value=value,
                                    min_allowed=0.0,
                                    max_allowed=100.0,
                                    message=f"Humidity {value}% outside valid range [0, 100] at index {i}",
                                )
                            )

        return issues


class PrecipitationValidator(BaseValidator):
    """Precipitation should be non-negative."""

    @property
    def rule_name(self) -> str:
        return "precipitation_non_negative"

    @property
    def severity(self):
        from ..schemas.validation import IssueSeverity

        return IssueSeverity.WARNING

    def check(self, response: RawForecastResponse) -> list[SemanticIssue]:
        issues: list[SemanticIssue] = []

        for loc in response.locations:
            if loc.hourly is not None:
                for var_name, values in loc.hourly.items():
                    if "precipitation" not in var_name or "sum" in var_name:
                        continue  # Daily sums can be checked differently
                    if not isinstance(values, list):
                        continue
                    for i, value in enumerate(values):
                        if value is None:
                            continue
                        if value < 0:
                            issues.append(
                                SemanticIssue(
                                    rule=self.rule_name,
                                    variable=var_name,
                                    value=value,
                                    message=f"Precipitation {value} is negative at index {i}",
                                )
                            )

        return issues


class PressureValidator(BaseValidator):
    """Surface pressure should be within plausible range (800-1100 hPa)."""

    @property
    def rule_name(self) -> str:
        return "pressure_range"

    @property
    def severity(self):
        from ..schemas.validation import IssueSeverity

        return IssueSeverity.WARNING

    def check(self, response: RawForecastResponse) -> list[SemanticIssue]:
        issues: list[SemanticIssue] = []

        for loc in response.locations:
            if loc.hourly is not None:
                for var_name, values in loc.hourly.items():
                    if "pressure" not in var_name or "msl" not in var_name:
                        continue
                    if not isinstance(values, list):
                        continue
                    for i, value in enumerate(values):
                        if value is None:
                            continue
                        if value < 800 or value > 1100:
                            issues.append(
                                SemanticIssue(
                                    rule=self.rule_name,
                                    variable=var_name,
                                    value=value,
                                    min_allowed=800.0,
                                    max_allowed=1100.0,
                                    message=f"Pressure {value} hPa outside plausible range [800, 1100] at index {i}",
                                )
                            )

        return issues


class SemanticValidator:
    """Convenience factory — returns a populated ValidatorPipeline."""

    @staticmethod
    def pipeline():
        from ..validators.base import ValidatorPipeline

        pipe = ValidatorPipeline()
        pipe.add(TemperatureValidator())
        pipe.add(HumidityValidator())
        pipe.add(PrecipitationValidator())
        pipe.add(PressureValidator())
        return pipe
