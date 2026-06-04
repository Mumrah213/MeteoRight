"""Structural validation rules.

These check the data structure itself — not the values.
"""


from typing import Any

from ..schemas.forecast_raw import RawForecastResponse
from ..schemas.validation import IssueSeverity, StructuralIssue
from ..validators.base import BaseValidator, ValidatorPipeline


class ArrayLengthConsistencyValidator(BaseValidator):
    """All arrays for a given variable should have the same length."""

    @property
    def rule_name(self) -> str:
        return "array_length_consistency"

    def check(self, response: RawForecastResponse) -> list[StructuralIssue]:
        issues: list[StructuralIssue] = []

        for loc in response.locations:
            if loc.hourly is not None:
                self._check_arrays(loc.hourly, loc.hourly_units, "hourly", issues)
            if loc.daily is not None:
                self._check_arrays(loc.daily, loc.daily_units, "daily", issues)

        return issues

    def _check_arrays(
        self,
        data: dict[str, Any],
        units: dict[str, str] | None,
        level: str,
        issues: list[StructuralIssue],
    ) -> None:
        # Find the reference array length (first non-time list)
        ref_len = None
        ref_var = None

        for var_name, values in data.items():
            if var_name in ("time", "time_utc"):
                continue  # Timestamps are not data arrays
            if not isinstance(values, list):
                continue

            if ref_len is None:
                ref_len = len(values)
                ref_var = var_name

        # Check all arrays match reference length
        for var_name, values in data.items():
            if not isinstance(values, list):
                continue

            if len(values) != ref_len:
                issues.append(
                    StructuralIssue(
                        rule=self.rule_name,
                        field=f"{level}.{var_name}",
                        message=(
                            f"Array length mismatch: {var_name} has {len(values)} values, "
                            f"but reference array {ref_var} has {ref_len}"
                        ),
                        raw_value={"expected": ref_len, "actual": len(values)},
                        severity=IssueSeverity.ERROR,
                    )
                )


class TimestampsMonotonicValidator(BaseValidator):
    """Timestamps must be strictly increasing."""

    @property
    def rule_name(self) -> str:
        return "timestamps_monotonic"

    def check(self, response: RawForecastResponse) -> list[StructuralIssue]:
        issues: list[StructuralIssue] = []

        for loc in response.locations:
            if loc.hourly is not None:
                times = loc.hourly.get("time") or loc.hourly.get("time_utc")
                if isinstance(times, list) and len(times) > 1:
                    self._check_monotonic(times, "hourly", issues)

            if loc.daily is not None:
                times = loc.daily.get("time") or loc.daily.get("time_utc")
                if isinstance(times, list) and len(times) > 1:
                    self._check_monotonic(times, "daily", issues)

        return issues

    def _check_monotonic(
        self,
        timestamps: list[str],
        level: str,
        issues: list[StructuralIssue],
    ) -> None:
        for i in range(1, len(timestamps)):
            if timestamps[i] <= timestamps[i - 1]:
                issues.append(
                    StructuralIssue(
                        rule=self.rule_name,
                        field=f"{level}.time",
                        message=f"Non-monotonic timestamps at index {i}: '{timestamps[i]}' <= '{timestamps[i - 1]}'",
                        raw_value={"index": i, "timestamp": timestamps[i]},
                        severity=IssueSeverity.ERROR,
                    )
                )
                break  # Report once


class RequiredKeysValidator(BaseValidator):
    """Check that required keys exist in the response structure."""

    @property
    def rule_name(self) -> str:
        return "required_keys"

    def check(self, response: RawForecastResponse) -> list[StructuralIssue]:
        issues: list[StructuralIssue] = []

        for loc in response.locations:
            if loc.hourly is not None:
                has_time = "time" in loc.hourly or "time_utc" in loc.hourly
                if not has_time and loc.hourly:  # Non-empty hourly dict
                    issues.append(
                        StructuralIssue(
                            rule=self.rule_name,
                            field="hourly.time",
                            message="Hourly data missing 'time' or 'time_utc' key",
                            severity=IssueSeverity.ERROR,
                        )
                    )

        return issues


class UnitsExistValidator(BaseValidator):
    """Check that units exist for data variables."""

    @property
    def rule_name(self) -> str:
        return "units_exist"

    def check(self, response: RawForecastResponse) -> list[StructuralIssue]:
        issues: list[StructuralIssue] = []

        for loc in response.locations:
            if loc.hourly is not None:
                data_vars = [k for k in loc.hourly.keys() if k not in ("time", "time_utc")]
                units = loc.hourly_units or {}
                missing = [v for v in data_vars if v not in units]
                for var in missing:
                    issues.append(
                        StructuralIssue(
                            rule=self.rule_name,
                            field=f"hourly.{var}.unit",
                            message=f"Missing units for hourly variable '{var}'",
                            severity=IssueSeverity.WARNING,
                        )
                    )

        return issues


class StructuralValidator:
    """Convenience factory — returns a populated ValidatorPipeline."""

    @staticmethod
    def pipeline() -> ValidatorPipeline:
        pipe = ValidatorPipeline()
        pipe.add(ArrayLengthConsistencyValidator())
        pipe.add(TimestampsMonotonicValidator())
        pipe.add(RequiredKeysValidator())
        pipe.add(UnitsExistValidator())
        return pipe
