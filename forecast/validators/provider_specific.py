"""Provider-specific validation rules.

These detect the specific silent failure modes of each provider.
For Open-Meteo, the critical issue is that HTTP 200 does NOT mean
valid data — unsupported models return empty or all-null arrays.
"""


from ..schemas.errors import NullClassification
from ..schemas.forecast_raw import RawForecastResponse
from ..schemas.validation import SemanticIssue, StructuralIssue
from ..validators.base import BaseValidator


class OpenMeteoNullDetectionValidator(BaseValidator):
    """Detect Open-Meteo silent failures.

    Open-Meteo's Single Runs API has a critical behavior:
    - It ONLY supports the ECMWF IFS model
    - Requesting other models returns HTTP 200 with empty or all-null arrays
    - There is NO error in the response structure — it looks valid
    - The only way to detect this is to examine the data values

    This validator detects:
    - ALL_NULL_RESPONSE: hourly arrays exist but every value is null
    - EMPTY_RESPONSE: hourly arrays are empty lists
    - PARTIAL_NULLS: some values are null (may be outside forecast horizon)
    """

    @property
    def rule_name(self) -> str:
        return "openmeteo_null_detection"

    def check(self, response: RawForecastResponse) -> list[SemanticIssue | StructuralIssue]:
        issues: list[SemanticIssue | StructuralIssue] = []
        classifications: list[NullClassification] = []

        for loc in response.locations:
            # Check hourly data
            hourly = loc.hourly
            if hourly is None:
                continue

            # Skip timestamp keys
            data_vars = {k: v for k, v in hourly.items() if k not in ("time", "time_utc")}

            if not data_vars:
                classifications.append(NullClassification.EMPTY_RESPONSE)
                issues.append(
                    StructuralIssue(
                        rule=self.rule_name,
                        field="hourly",
                        message="Hourly data is empty — no variables returned",
                        raw_value={"hourly_keys": list(hourly.keys())},
                    )
                )
                continue

            # Check each data variable
            all_null_vars = []
            empty_vars = []
            has_non_null = False

            for var_name, values in data_vars.items():
                if not isinstance(values, list):
                    continue

                if len(values) == 0:
                    empty_vars.append(var_name)
                    continue

                null_count = sum(1 for v in values if v is None)
                non_null_count = len(values) - null_count

                if null_count > 0:
                    issues.append(
                        SemanticIssue(
                            rule=self.rule_name,
                            variable=var_name,
                            value=None,
                            message=(
                                f"Variable '{var_name}' has {null_count}/{len(values)} null values "
                                f"({100 * null_count / len(values):.0f}%)"
                            ),
                        )
                    )

                if non_null_count > 0:
                    has_non_null = True

                if null_count == len(values):
                    all_null_vars.append(var_name)

            # Classify the overall pattern
            if not has_non_null and (all_null_vars or empty_vars):
                if empty_vars:
                    classifications.append(NullClassification.EMPTY_RESPONSE)
                elif all_null_vars:
                    classifications.append(NullClassification.ALL_NULL_RESPONSE)
                issues.append(
                    SemanticIssue(
                        rule=self.rule_name,
                        variable="all",
                        value=None,
                        message=(
                            f"All data variables are null — likely unsupported model or configuration. "
                            f"Null variables: {all_null_vars}"
                        ),
                    )
                )

            if not classifications and not any(
                isinstance(i, StructuralIssue) and i.severity.name == "ERROR" for i in issues
            ):
                # No null issue found — mark as OK
                pass

        # Store classification for downstream use
        if classifications:
            # Use the most severe classification
            severity_order = [
                NullClassification.EMPTY_RESPONSE,
                NullClassification.ALL_NULL_RESPONSE,
                NullClassification.PARTIAL_NULLS,
            ]
            for cls in severity_order:
                if cls in classifications:
                    response.null_classification = cls.name
                    break

        return issues


class OpenMeteoEmptyArrayValidator(BaseValidator):
    """Detect empty data arrays (not just null, but truly empty lists)."""

    @property
    def rule_name(self) -> str:
        return "openmeteo_empty_arrays"

    def check(self, response: RawForecastResponse) -> list[StructuralIssue]:
        issues: list[StructuralIssue] = []

        for loc in response.locations:
            if loc.hourly is not None:
                for var_name, values in loc.hourly.items():
                    if var_name in ("time", "time_utc"):
                        continue
                    if isinstance(values, list) and len(values) == 0:
                        issues.append(
                            StructuralIssue(
                                rule=self.rule_name,
                                field=f"hourly.{var_name}",
                                message=f"Hourly variable '{var_name}' has empty array (0 values)",
                                raw_value={"variable": var_name, "length": 0},
                            )
                        )

        return issues


class OpenMeteoSilentModelFailureValidator(BaseValidator):
    """Detect when a model query succeeded structurally but produced no data.

    This is the KEY validator for Open-Meteo's Single Runs API.
    The API returns HTTP 200 for unsupported models, but with empty
    or all-null arrays. We must detect this pattern.
    """

    @property
    def rule_name(self) -> str:
        return "openmeteo_silent_model_failure"

    def check(self, response: RawForecastResponse) -> list[StructuralIssue]:
        issues: list[StructuralIssue] = []

        # Check if model was specified in params
        model = response.request_params.get("models", "")
        if not model:
            return issues  # No model specified, can't classify

        has_data = False
        for loc in response.locations:
            if loc.hourly is not None:
                hourly_data = {k: v for k, v in loc.hourly.items() if k not in ("time", "time_utc")}
                for _var_name, values in hourly_data.items():
                    if (
                        isinstance(values, list)
                        and len(values) > 0
                        and any(v is not None for v in values)
                    ):
                        has_data = True
                        break

        if not has_data:
            issues.append(
                StructuralIssue(
                    rule=self.rule_name,
                    field="data",
                    message=(
                        f"Model '{model}' returned no data — this may indicate an unsupported model. "
                        f"Open-Meteo Single Runs API only supports the ECMWF IFS model."
                    ),
                    raw_value={
                        "requested_model": model,
                        "total_variables": sum(
                            1
                            for loc in response.locations
                            if loc.hourly
                            for k in loc.hourly
                            if k not in ("time", "time_utc")
                        ),
                        "null_classification": response.null_classification,
                    },
                )
            )

        return issues


class OpenMeteoValidator:
    """Convenience factory — returns a populated ValidatorPipeline for Open-Meteo."""

    @staticmethod
    def pipeline():
        from ..validators.base import ValidatorPipeline

        pipe = ValidatorPipeline()
        pipe.add(OpenMeteoNullDetectionValidator())
        pipe.add(OpenMeteoEmptyArrayValidator())
        pipe.add(OpenMeteoSilentModelFailureValidator())
        return pipe
