"""Phase 4: Fairness constraints for cross-model comparison.

Ensures that model comparisons are scientifically defensible by:
- Intersecting comparable datasets
- Exposing coverage limitations
- Warning about unequal sample sizes
- Preventing unfair comparisons (unequal horizons, missing variables)
- Never silently filtering or interpolating missing data
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from pydantic import BaseModel


class FairnessWarning(BaseModel):
    """One fairness constraint violation or warning."""

    code: str  # e.g., "UNEQUAL_HORIZON", "MISSING_VARIABLE", "LOW_COVERAGE"
    severity: str  # "error", "warning", "info"
    model_a: str | None = None
    model_b: str | None = None
    variable: str | None = None
    message: str
    recommendation: str = ""

    def is_blocking(self) -> bool:
        """True if this warning should prevent comparison."""
        return self.severity == "error"


class FairnessConstraintChecker:
    """Validate fairness constraints before model comparison.

    CRITICAL RULES:
    1. Models must share variables to be compared on those variables
    2. Models must have overlapping forecast horizons
    3. Coverage differences must be explicitly reported
    4. No silent filtering or interpolation
    5. Unequal sample sizes must be flagged

    Usage:
        checker = FairnessConstraintChecker(min_coverage=0.5)
        checker.check_comparison(
            model_a="ifs",
            model_a_data=ifs_records,
            model_b="gfs",
            model_b_data=gfs_records,
            variable="temperature_2m",
        )
        if checker.has_blocking_issues():
            raise BenchmarkError(checker.summary())
    """

    # Severity levels
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"

    # Warning codes
    UNEQUAL_HORIZON = "UNEQUAL_HORIZON"
    MISSING_VARIABLE = "MISSING_VARIABLE"
    LOW_COVERAGE = "LOW_COVERAGE"
    UNEQUAL_SAMPLE_SIZES = "UNEQUAL_SAMPLE_SIZES"
    OVERLAPPING_VALID_TIMES = "OVERLAPPING_VALID_TIMES"
    NO_COMMON_VARIABLES = "NO_COMMON_VARIABLES"

    def __init__(
        self,
        min_coverage: float = 0.0,
        min_sample_size: int = 1,
    ) -> None:
        """Initialize fairness checker.

        Args:
            min_coverage: Minimum coverage_fraction (0..1) to allow comparison
            min_sample_size: Minimum sample_size for stability
        """
        self.min_coverage = min_coverage
        self.min_sample_size = min_sample_size
        self.warnings: list[FairnessWarning] = []

    def check_comparison(
        self,
        model_a: str,
        model_a_data: list[Any],
        model_b: str,
        model_b_data: list[Any],
        variable: str,
        max_lead_hours_a: int | None = None,
        max_lead_hours_b: int | None = None,
    ) -> list[FairnessWarning]:
        """Check fairness constraints between two models.

        Args:
            model_a: First model name
            model_a_data: Forecast records for model A
            model_b: Second model name
            model_b_data: Forecast records for model B
            variable: Variable being compared
            max_lead_hours_a: Max forecast horizon for model A
            max_lead_hours_b: Max forecast horizon for model B

        Returns:
            List of FairnessWarning objects (may include blocking errors)
        """
        self.warnings = []

        # 1. Check variable support
        if not self._check_variable_support(model_a, model_b, variable, model_a_data, model_b_data):
            self.warnings.append(
                FairnessWarning(
                    code=self.NO_COMMON_VARIABLES,
                    severity=self.ERROR,
                    model_a=model_a,
                    model_b=model_b,
                    variable=variable,
                    message=(
                        f"Variable '{variable}' not supported by both models. "
                        f"Model A: {model_a}, Model B: {model_b}"
                    ),
                    recommendation=(
                        "Only compare models on variables supported by both. "
                        "Check capability matrix for supported variables."
                    ),
                )
            )

        # 2. Check horizon overlap
        self._check_horizon_overlap(
            model_a,
            model_b,
            variable,
            max_lead_hours_a,
            max_lead_hours_b,
        )

        # 3. Check coverage
        self._check_coverage(
            model_a,
            model_b,
            variable,
            model_a_data,
            model_b_data,
        )

        # 4. Check sample sizes
        self._check_sample_sizes(
            model_a,
            model_b,
            variable,
            model_a_data,
            model_b_data,
        )

        return self.warnings

    def check_capability_matrix(
        self,
        capability_matrix: Any,
        model_names: list[str],
        variable: str,
    ) -> list[FairnessWarning]:
        """Check fairness using the capability matrix.

        Args:
            capability_matrix: CapabilityMatrix instance
            model_names: List of model names to compare
            variable: Variable being compared

        Returns:
            List of FairnessWarning objects
        """
        self.warnings = []

        # Get models that support this variable
        [name for name in model_names if capability_matrix.get_supported_models(variable)]

        # Check if all models support the variable
        for name in model_names:
            cap = capability_matrix.models.get(name)
            if cap and variable not in cap.supported_variables:
                self.warnings.append(
                    FairnessWarning(
                        code=self.MISSING_VARIABLE,
                        severity=self.ERROR,
                        model_a=name,
                        message=(
                            f"Model '{name}' does not support variable '{variable}'. "
                            "Cannot include in fair comparison."
                        ),
                        recommendation=(
                            f"Exclude '{name}' from comparison or choose a variable "
                            "supported by all models."
                        ),
                    )
                )

        # Check horizon overlap between all pairs
        for i, name_a in enumerate(model_names):
            for name_b in model_names[i + 1 :]:
                cap_a = capability_matrix.models.get(name_a)
                cap_b = capability_matrix.models.get(name_b)
                if cap_a and cap_b:
                    max_a = cap_a.max_lead_hours
                    max_b = cap_b.max_lead_hours
                    if max_a is not None and max_b is not None:
                        if abs(max_a - max_b) > 24:
                            self.warnings.append(
                                FairnessWarning(
                                    code=self.UNEQUAL_HORIZON,
                                    severity=self.WARNING,
                                    model_a=name_a,
                                    model_b=name_b,
                                    variable=variable,
                                    message=(
                                        f"Models '{name_a}' (max {max_a}h) and "
                                        f"'{name_b}' (max {max_b}h) have significantly "
                                        f"different forecast horizons."
                                    ),
                                    recommendation=(
                                        "Limit comparison to the overlapping horizon "
                                        f"({min(max_a, max_b)}h)."
                                    ),
                                )
                            )

        return self.warnings

    def check_intersections(
        self,
        model_names: list[str],
        capability_matrix: Any,
        variables: list[str],
    ) -> dict[str, Any]:
        """Determine fair comparison subsets from capability matrix.

        Returns dict with:
        - common_variables: variables supported by ALL models
        - pairwise_variables: variables supported by each model pair
        - warnings: any fairness issues

        Args:
            model_names: List of model names
            capability_matrix: CapabilityMatrix instance
            variables: All candidate variables

        Returns:
            Dict with intersection analysis
        """
        common = capability_matrix.get_common_variables(model_names)
        pairwise: dict[tuple[str, str], list[str]] = {}

        for i, name_a in enumerate(model_names):
            for name_b in model_names[i + 1 :]:
                pairwise[(name_a, name_b)] = capability_matrix.get_intersecting_variables(
                    name_a, name_b
                )

        warnings = self.check_capability_matrix(
            capability_matrix, model_names, variables[0] if variables else ""
        )

        return {
            "common_variables": common,
            "pairwise_variables": pairwise,
            "warnings": warnings,
        }

    def _check_variable_support(
        self,
        model_a: str,
        model_b: str,
        variable: str,
        data_a: list[Any],
        data_b: list[Any],
    ) -> bool:
        """Check if both models have data for the variable."""
        has_a = any(getattr(r, "variable", None) == variable for r in data_a)
        has_b = any(getattr(r, "variable", None) == variable for r in data_b)
        return has_a and has_b

    def _check_horizon_overlap(
        self,
        model_a: str,
        model_b: str,
        variable: str,
        max_lead_a: int | None,
        max_lead_b: int | None,
    ) -> None:
        """Check forecast horizon overlap between models."""
        if max_lead_a is None or max_lead_b is None:
            self.warnings.append(
                FairnessWarning(
                    code=self.UNEQUAL_HORIZON,
                    severity=self.INFO,
                    model_a=model_a,
                    model_b=model_b,
                    variable=variable,
                    message="Forecast horizon unknown for one or both models.",
                    recommendation="Check capability matrix for max_lead_hours.",
                )
            )
        elif max_lead_a != max_lead_b:
            self.warnings.append(
                FairnessWarning(
                    code=self.UNEQUAL_HORIZON,
                    severity=self.WARNING,
                    model_a=model_a,
                    model_b=model_b,
                    variable=variable,
                    message=(
                        f"Models have different max horizons: "
                        f"{model_a}={max_lead_a}h, {model_b}={max_lead_b}h."
                    ),
                    recommendation=(
                        f"Limit comparison to min({min(max_lead_a, max_lead_b)}h) or less."
                    ),
                )
            )

    def _check_coverage(
        self,
        model_a: str,
        model_b: str,
        variable: str,
        data_a: list[Any],
        data_b: list[Any],
    ) -> None:
        """Check coverage for each model's data."""

        # Count non-null values for the variable
        def count_valid(data: list[Any]) -> int:
            count = 0
            for r in data:
                if (
                    getattr(r, "variable", None) == variable
                    and getattr(r, "value", None) is not None
                ):
                    count += 1
            return count

        valid_a = count_valid(data_a)
        valid_b = count_valid(data_b)

        if valid_a == 0:
            self.warnings.append(
                FairnessWarning(
                    code=self.LOW_COVERAGE,
                    severity=self.ERROR,
                    model_a=model_a,
                    variable=variable,
                    message=(
                        f"Model '{model_a}' has zero valid values for '{variable}'. Cannot compare."
                    ),
                    recommendation=f"Check data quality for {model_a} on {variable}.",
                )
            )

        if valid_b == 0:
            self.warnings.append(
                FairnessWarning(
                    code=self.LOW_COVERAGE,
                    severity=self.ERROR,
                    model_b=model_b,
                    variable=variable,
                    message=(
                        f"Model '{model_b}' has zero valid values for '{variable}'. Cannot compare."
                    ),
                    recommendation=f"Check data quality for {model_b} on {variable}.",
                )
            )

    def _check_sample_sizes(
        self,
        model_a: str,
        model_b: str,
        variable: str,
        data_a: list[Any],
        data_b: list[Any],
    ) -> None:
        """Check for significantly different sample sizes."""

        def count_valid(data: list[Any]) -> int:
            return sum(
                1
                for r in data
                if getattr(r, "variable", None) == variable
                and getattr(r, "value", None) is not None
            )

        count_a = count_valid(data_a)
        count_b = count_valid(data_b)

        # Warn if sample sizes differ by more than 50%
        if count_a > 0 and count_b > 0:
            ratio = min(count_a, count_b) / max(count_a, count_b)
            if ratio < 0.5:
                self.warnings.append(
                    FairnessWarning(
                        code=self.UNEQUAL_SAMPLE_SIZES,
                        severity=self.WARNING,
                        model_a=model_a,
                        model_b=model_b,
                        variable=variable,
                        message=(
                            f"Sample sizes differ significantly: "
                            f"{model_a}={count_a}, {model_b}={count_b} "
                            f"(ratio={ratio:.2f})."
                        ),
                        recommendation=(
                            "Results may be biased by unequal sample sizes. "
                            "Consider subsampling or reporting separate results."
                        ),
                    )
                )

    def has_blocking_issues(self) -> bool:
        """True if any ERROR-level warnings prevent comparison."""
        return any(w.severity == self.ERROR for w in self.warnings)

    def summary(self) -> dict[str, Any]:
        """Summary of all fairness warnings."""
        return {
            "total_warnings": len(self.warnings),
            "blocking_issues": sum(1 for w in self.warnings if w.severity == self.ERROR),
            "warnings_by_code": {
                code: [w.message for w in self.warnings if w.code == code]
                for code in {w.code for w in self.warnings}
            },
        }


class FairnessConstraintCheckerV2:
    """Extended fairness checker with coverage-parity and correlated-error support.

    Adds:
    - Sample-size ratio checking between all models
    - Coverage ratio checking between all models
    - Correlated error detection across models
    """

    def __init__(
        self,
        min_coverage: float = 0.0,
        min_sample_size: int = 1,
        max_coverage_ratio: float | None = None,
        max_sample_ratio: float | None = None,
    ) -> None:
        """Initialize extended fairness checker.

        Args:
            min_coverage: Minimum coverage_fraction (0..1)
            min_sample_size: Minimum samples for stability
            max_coverage_ratio: Maximum allowed ratio between largest
                                and smallest coverage_fraction. None = no limit.
            max_sample_ratio: Maximum allowed ratio between largest
                              and smallest sample_size. None = no limit.
        """
        self.min_coverage = min_coverage
        self.min_sample_size = min_sample_size
        self.max_coverage_ratio = max_coverage_ratio
        self.max_sample_ratio = max_sample_ratio
        self.warnings: list[FairnessWarning] = []

    def check_all(
        self,
        coverage_fractions: dict[str, float],
        sample_sizes: dict[str, int] | None = None,
        variables: dict[str, list[int]] | None = None,
    ) -> list[FairnessWarning]:
        """Run all fairness checks together.

        Convenience method to check coverage, sample sizes, and
        horizon overlap in a single call.

        Args:
            coverage_fractions: Dict mapping model_name -> coverage_fraction
            sample_sizes: Dict mapping model_name -> sample_size
            variables: Dict mapping variable_name -> list of horizons

        Returns:
            Combined list of all FairnessWarning objects
        """
        self.warnings = []

        # 1. Check minimum coverage
        for model_name, coverage in coverage_fractions.items():
            if coverage < self.min_coverage:
                self.warnings.append(
                    FairnessWarning(
                        code="LOW_COVERAGE",
                        severity=self.ERROR,
                        model_a=model_name,
                        message=(
                            f"Model '{model_name}' has coverage {coverage:.2%} "
                            f"(below threshold {self.min_coverage:.2%}). "
                            f"Comparison is unfair."
                        ),
                        recommendation="Increase data coverage or lower threshold.",
                    )
                )

        # 2. Check coverage ratio between models
        if self.max_coverage_ratio is not None and len(coverage_fractions) >= 2:
            max_cov = max(coverage_fractions.values())
            min_cov = min(coverage_fractions.values())
            if min_cov > 0 and (max_cov / min_cov) > self.max_coverage_ratio:
                self.warnings.append(
                    FairnessWarning(
                        code="COVERAGE_IMBALANCE",
                        severity=self.WARNING,
                        message=(
                            f"Coverage ratio {max_cov:.2%}/{min_cov:.2%} = "
                            f"{max_cov / min_cov:.2f}x exceeds limit "
                            f"({self.max_coverage_ratio}). "
                            f"Rankings may be biased."
                        ),
                        recommendation="Stratify comparison by coverage or filter models.",
                    )
                )

        # 3. Check sample sizes
        if sample_sizes:
            for model_name, size in sample_sizes.items():
                if size < self.min_sample_size:
                    self.warnings.append(
                        FairnessWarning(
                            code="INSUFFICIENT_SAMPLES",
                            severity=self.ERROR,
                            model_a=model_name,
                            message=(
                                f"Model '{model_name}' has only {size} samples "
                                f"(minimum: {self.min_sample_size}). "
                                f"Metric may be unreliable."
                            ),
                            recommendation="Gather more data or lower min_sample_size.",
                        )
                    )

            if len(sample_sizes) >= 2:
                max_size = max(sample_sizes.values())
                min_size = min(sample_sizes.values())
                ratio_limit = self.max_sample_ratio or 10.0
                if min_size > 0 and (max_size / min_size) > ratio_limit:
                    self.warnings.append(
                        FairnessWarning(
                            code="SAMPLE_SIZE_IMBALANCE",
                            severity=self.WARNING,
                            message=(
                                f"Sample size ratio {max_size}/{min_size} = "
                                f"{max_size / min_size:.1f}x exceeds limit "
                                f"({ratio_limit}). "
                                f"Rankings may be unreliable."
                            ),
                            recommendation="Require minimum sample sizes or subsample.",
                        )
                    )

        # 4. Check horizon overlap if variables provided
        if variables and len(variables) >= 2:
            horizon_sets: dict[str, list[int]] = {}
            for var_name, horizons in variables.items():
                horizon_sets[var_name] = sorted(horizons)

            var_names = sorted(horizon_sets.keys())
            for i in range(len(var_names)):
                for j in range(i + 1, len(var_names)):
                    v1, v2 = var_names[i], var_names[j]
                    h1, h2 = horizon_sets[v1], horizon_sets[v2]
                    set1, set2 = set(h1), set(h2)
                    intersection = set1 & set2
                    if len(intersection) < len(set1) and len(intersection) < len(set2):
                        self.warnings.append(
                            FairnessWarning(
                                code="HORIZON_MISMATCH",
                                severity=self.INFO,
                                variable=v1,
                                message=(
                                    f"Variable '{v1}' (horizons {h1}) and "
                                    f"'{v2}' (horizons {h2}) have mismatched "
                                    f"horizons (intersection: {sorted(intersection)}). "
                                    f"Direct comparison is limited."
                                ),
                                recommendation="Compare models within overlapping horizons only.",
                            )
                        )

        return self.warnings

    def is_blocking(self) -> bool:
        """Check if any warnings are blocking."""
        return any(w.severity == self.ERROR for w in self.warnings)

    @staticmethod
    def analyze_correlated_errors(
        alignments_by_model: dict[str, list[Any]],
    ) -> dict[str, Any]:
        """Detect correlated errors across models.

        If multiple models share the same type of error (e.g., both
        consistently under-predict at a certain location or time),
        this indicates a shared model weakness rather than independent
        errors.

        Args:
            alignments_by_model: Dict mapping model_name -> AlignmentRecord list

        Returns:
            Dict with correlation analysis results including:
            - shared_error_locations: Locations where multiple models err together
            - shared_error_types: Types of errors shared across models
            - warning: Description if correlated errors are detected
        """
        # Collect error signs per model per location
        location_errors: dict[tuple[float, float], dict[str, str]] = defaultdict(dict)
        for model_name, alignments in alignments_by_model.items():
            for a in alignments:
                if not hasattr(a, "is_fully_matched") or not a.is_fully_matched:
                    continue
                forecast = getattr(a, "forecast_value", None)
                observation = getattr(a, "observation_value", None)
                if forecast is None or observation is None:
                    continue
                error = forecast - observation
                sign = "over" if error > 0 else "under"
                lat = getattr(a, "latitude", 0.0)
                lon = getattr(a, "longitude", 0.0)
                location_errors[(lat, lon)][model_name] = sign

        # Find locations where models agree on error direction
        shared_error_locations: list[dict[str, Any]] = []
        for (lat, lon), model_signs in location_errors.items():
            if len(model_signs) < 2:
                continue
            signs = list(model_signs.values())
            if all(s == signs[0] for s in signs):
                shared_error_locations.append(
                    {
                        "location": (lat, lon),
                        "error_type": f"All models {signs[0]}-predict",
                        "models": list(model_signs.keys()),
                    }
                )

        # Aggregate into a warning if significant correlated errors exist
        warning: str | None = None
        if shared_error_locations:
            warning = (
                f"Detected {len(shared_error_locations)} location(s) where "
                f"multiple models share the same error direction. "
                f"This indicates correlated model weakness, not independent skill."
            )

        return {
            "shared_error_locations": shared_error_locations,
            "total_shared_locations": len(shared_error_locations),
            "warning": warning,
        }
