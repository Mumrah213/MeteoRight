"""Phase 3: Lead-Time Degradation Analysis.

Analyzes how forecast error grows with lead time.

For each variable:
  - compute error vs lead_time curve
  - produce monotonic degradation profile (if applicable)
  - detect instability points

No plotting required — just structured outputs.

Core principle:
  Forecasts degrade over time. This module quantifies that degradation.
"""

from __future__ import annotations

from typing import Any

from .alignment import AlignmentRecord
from .metrics import MetricsEngine
from .models import DegradationPoint, DegradationProfile


class DegradationAnalysisError(Exception):
    """Raised when degradation analysis fails."""


class LeadTimeDegradationAnalyzer:
    """Analyze error growth with lead time.

    For each variable at each location:
    1. Group aligned records by lead_time
    2. Compute metric (MAE, RMSE, or bias) per lead_time
    3. Build degradation curve (sample-size aware)
    4. Detect monotonicity and instability points
    5. Flag sparse bins where metrics are unreliable

    Output:
      DegradationProfile with list of DegradationPoint objects.
      Points with sample_size < min_sample_size are marked unstable.

    Core principle:
    Forecasts degrade over time. But sparse lead-time bins can produce
    misleading metrics. This analyzer flags such cases.
    """

    def __init__(self, metric: str = "mae", min_sample_size: int = 1) -> None:
        """Initialize degradation analyzer.

        Args:
            metric: One of "mae", "rmse", "bias"
            min_sample_size: Minimum samples per lead-time bin for
                            is_stable=True. Bins below this threshold
                            are flagged as unstable.
        """
        if min_sample_size < 1:
            raise ValueError("min_sample_size must be >= 1")
        self.metric = metric
        self._min_sample_size = min_sample_size
        self.metrics_engine = MetricsEngine(min_sample_size)

    def analyze(
        self,
        alignments: list[AlignmentRecord],
        variable: str,
        location: tuple[float, float] | None = None,
    ) -> DegradationProfile:
        """Analyze lead-time degradation for one variable at one location.

        Args:
            alignments: List of AlignmentRecord objects
            variable: Variable name to analyze
            location: Optional (latitude, longitude) filter

        Returns:
            DegradationProfile with error curve and analysis
        """
        # Filter alignments
        filtered = [a for a in alignments if a.variable == variable]
        if location is not None:
            filtered = [
                a for a in filtered if a.latitude == location[0] and a.longitude == location[1]
            ]

        if not filtered:
            return DegradationProfile(
                variable=variable,
                location=location or (0.0, 0.0),
                metric_type=self.metric,
                points=[],
                is_monotonic=False,
                instability_points=[],
            )

        # Group by lead_time
        by_lead_time = self._group_by_lead_time(filtered)

        # Sort by lead_time
        sorted_leads = sorted(by_lead_time.keys())

        # Compute metric for each lead_time
        points: list[DegradationPoint] = []
        prev_value: float | None = None

        for lead_hours in sorted_leads:
            lead_alignments = by_lead_time[lead_hours]
            stats = self._compute_metric_for_group(lead_alignments)

            if stats is None:
                continue

            value = stats.get(self.metric)
            if value is None:
                continue

            # Check monotonicity
            is_monotonic = False
            if prev_value is not None:
                is_monotonic = value >= prev_value

            # Extract coverage info
            sample_size = stats.get("sample_size", 0)
            total_count = stats.get("total_count", sample_size)

            point = DegradationPoint(
                lead_hours=lead_hours,
                metric_type=self.metric,
                variable=variable,
                value=value,
                sample_size=sample_size,
                total_count=total_count,
                coverage_fraction=(sample_size / total_count if total_count > 0 else 0.0),
                is_monotonic_with_prev=is_monotonic,
                min_sample_size=self._min_sample_size,
            )
            points.append(point)
            prev_value = value

        # Detect instability points (where error decreases)
        instability_points = [p.lead_hours for p in points if not p.is_monotonic_with_prev]

        # Check if overall monotonic (no instability)
        is_monotonic = len(instability_points) == 0

        # Build summary
        summary = self._build_summary(points, is_monotonic, instability_points)

        return DegradationProfile(
            variable=variable,
            location=location or (filtered[0].latitude, filtered[0].longitude),
            metric_type=self.metric,
            points=points,
            is_monotonic=is_monotonic,
            instability_points=instability_points,
            summary=summary,
        )

    def analyze_all_variables(
        self,
        alignments: list[AlignmentRecord],
    ) -> list[DegradationProfile]:
        """Analyze degradation for all variables in alignments.

        Args:
            alignments: List of AlignmentRecord objects

        Returns:
            List of DegradationProfile objects (one per variable)
        """
        # Get unique variables
        variables = list({a.variable for a in alignments})

        profiles: list[DegradationProfile] = []
        for variable in sorted(variables):
            profile = self.analyze(alignments, variable)
            profiles.append(profile)

        return profiles

    def _group_by_lead_time(
        self, alignments: list[AlignmentRecord]
    ) -> dict[int, list[AlignmentRecord]]:
        """Group alignments by lead_hours."""
        groups: dict[int, list[AlignmentRecord]] = {}
        for a in alignments:
            if a.lead_hours not in groups:
                groups[a.lead_hours] = []
            groups[a.lead_hours].append(a)
        return groups

    def _compute_metric_for_group(
        self, alignments: list[AlignmentRecord]
    ) -> dict[str, float | None] | None:
        """Compute metrics for one lead_time group.

        Returns dict with metric values plus sample_size and total_count.
        total_count includes ALL records (matched + unmatched) at this
        lead time, enabling coverage_fraction computation.
        """
        if not alignments:
            return None

        total_count = len(alignments)

        # Use only matched records
        matched = [a for a in alignments if a.is_fully_matched]
        if not matched:
            return None

        # Compute error statistics
        errors = []
        for a in matched:
            error = a.forecast_value - a.observation_value
            errors.append(error)

        n = len(errors)
        mean_error = sum(errors) / n

        mae = sum(abs(e) for e in errors) / n
        rmse = (sum(e**2 for e in errors) / n) ** 0.5
        bias = mean_error
        variance = sum((e - mean_error) ** 2 for e in errors) / n

        return {
            "mae": mae,
            "rmse": rmse,
            "bias": bias,
            "error_variance": variance,
            "sample_size": n,
            "total_count": total_count,
        }

    def _build_summary(
        self,
        points: list[DegradationPoint],
        is_monotonic: bool,
        instability_points: list[int],
    ) -> dict[str, Any]:
        """Build summary statistics for the degradation profile."""
        if not points:
            return {}

        values = [p.value for p in points]
        min_value = min(values)
        max_value = max(values)
        min_lead = points[0].lead_hours
        max_lead = points[-1].lead_hours

        # Compute degradation rate (slope of linear fit)
        degradation_rate = None
        if len(points) >= 2:
            [p.lead_hours for p in points]
            # Simple slope: (last - first) / (last_lead - first_lead)
            if max_lead > min_lead:
                degradation_rate = (values[-1] - values[0]) / (max_lead - min_lead)

        return {
            "min_error": min_value,
            "max_error": max_value,
            "min_lead_hours": min_lead,
            "max_lead_hours": max_lead,
            "is_monotonic": is_monotonic,
            "instability_count": len(instability_points),
            "degradation_rate_per_hour": degradation_rate,
        }

    def find_fastest_degrading_variable(
        self,
        profiles: list[DegradationProfile],
    ) -> str | None:
        """Find the variable with fastest error growth.

        Uses degradation_rate from summary if available.

        Args:
            profiles: List of DegradationProfile objects

        Returns:
            Variable name with fastest degradation, or None
        """
        best_variable = None
        best_rate = None  # Higher rate = faster degradation

        for profile in profiles:
            rate = profile.summary.get("degradation_rate_per_hour")
            if rate is not None and (best_rate is None or rate > best_rate):
                best_rate = rate
                best_variable = profile.variable

        return best_variable

    def find_instability_variables(
        self,
        profiles: list[DegradationProfile],
    ) -> list[str]:
        """Find variables with instability points in their degradation curve.

        Args:
            profiles: List of DegradationProfile objects

        Returns:
            List of variable names with instability
        """
        return [p.variable for p in profiles if p.is_monotonic is False]
