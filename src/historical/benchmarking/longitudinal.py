"""Phase 4: Longitudinal performance tracking.

Tracks how model skill changes over time:
- Record benchmark results at regular intervals (monthly, seasonally)
- Compare model performance across months/seasons
- Identify trends (improving, stable, declining)
- Store historical skill records for reproducibility

Longitudinal tracking answers:
- "Has Model A improved or degraded since last quarter?"
- "Which models are most consistent across seasons?"
- "Are there temporal patterns in model bias?"
"""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime
from typing import Any

from ...forecast.verification.alignment import AlignmentEngine
from ...forecast.verification.metrics import MetricsEngine
from .models import (
    LongitudinalSkillRecord,
    LongitudinalTrend,
)


class LongitudinalTracker:
    """Track model performance over time.

    Supports:
    - Recording benchmark results at specific timestamps
    - Retrieving historical skill records
    - Trend analysis (improving, stable, declining)
    - Periodic benchmark scheduling

    Usage:
        tracker = LongitudinalTracker()
        # Record a benchmark result
        tracker.record(
            model_name="ifs",
            variable="temperature_2m",
            metric_type="mae",
            value=2.5,
            timestamp=datetime.now(),
            sample_size=1000,
            evaluation_window=(start_date, end_date),
        )
        # Analyze trends
        trend = tracker.get_trend("ifs", "temperature_2m", "mae")
    """

    def __init__(
        self,
        alignment_engine: AlignmentEngine | None = None,
        metrics_engine: MetricsEngine | None = None,
    ) -> None:
        """Initialize longitudinal tracker.

        Args:
            alignment_engine: Optional AlignmentEngine
            metrics_engine: Optional MetricsEngine
        """
        self.alignment_engine = alignment_engine or AlignmentEngine()
        self.metrics_engine = metrics_engine or MetricsEngine()
        self._records: dict[str, list[LongitudinalSkillRecord]] = defaultdict(list)

    def record(
        self,
        model_name: str,
        variable: str,
        metric_type: str,
        value: float | None,
        timestamp: datetime,
        sample_size: int = 0,
        coverage_fraction: float | None = None,
        evaluation_window: tuple[datetime, datetime] | None = None,
    ) -> LongitudinalSkillRecord:
        """Record a benchmark result for longitudinal tracking.

        Args:
            model_name: Model name
            variable: Variable evaluated
            metric_type: Metric type ("mae", "rmse", "bias")
            value: Metric value (float or None)
            timestamp: When this benchmark was computed
            sample_size: Number of matched pairs
            coverage_fraction: Coverage fraction (0..1)
            evaluation_window: (start, end) of evaluation period

        Returns:
            Recorded LongitudinalSkillRecord
        """
        record = LongitudinalSkillRecord(
            model_name=model_name,
            variable=variable,
            metric_type=metric_type,
            value=value,
            timestamp=timestamp,
            sample_size=sample_size,
            coverage_fraction=coverage_fraction,
            evaluation_window=evaluation_window,
        )
        key = f"{model_name}:{variable}:{metric_type}"
        self._records[key].append(record)
        self._records[key].sort(key=lambda r: r.timestamp)
        return record

    def record_from_evaluation(
        self,
        model_name: str,
        variable: str,
        metric_type: str,
        evaluation: Any,
        timestamp: datetime | None = None,
    ) -> LongitudinalSkillRecord:
        """Record from an existing ForecastEvaluation.

        Convenience method to record from MetricsEngine output.

        Args:
            model_name: Model name
            variable: Variable evaluated
            metric_type: Metric type
            evaluation: ForecastEvaluation object
            timestamp: Benchmark timestamp (defaults to now)

        Returns:
            Recorded LongitudinalSkillRecord
        """
        ts = timestamp or datetime.now()
        return self.record(
            model_name=model_name,
            variable=variable,
            metric_type=metric_type,
            value=evaluation.value,
            timestamp=ts,
            sample_size=evaluation.sample_size,
            coverage_fraction=evaluation.coverage_fraction,
            evaluation_window=evaluation.time_range,
        )

    def get_records(
        self,
        model_name: str,
        variable: str,
        metric_type: str,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
    ) -> list[LongitudinalSkillRecord]:
        """Get historical records for one model/variable/metric.

        Args:
            model_name: Model name
            variable: Variable evaluated
            metric_type: Metric type
            start_date: Filter from this date (inclusive)
            end_date: Filter to this date (inclusive)

        Returns:
            List of LongitudinalSkillRecord, sorted by timestamp
        """
        key = f"{model_name}:{variable}:{metric_type}"
        records = self._records.get(key, [])

        if start_date is None and end_date is None:
            return list(records)

        filtered: list[LongitudinalSkillRecord] = []
        for r in records:
            if start_date and r.timestamp < start_date:
                continue
            if end_date and r.timestamp > end_date:
                continue
            filtered.append(r)

        return filtered

    def _simple_linear_regression(self, x: list[float], y: list[float]) -> tuple[float, float]:
        """Simple linear regression: returns (slope, intercept).

        Uses the least-squares formula. If there's insufficient data
        or zero variance in x, returns (0.0, 0.0).

        Args:
            x: Independent variable values (normalized timestamps)
            y: Dependent variable values (metric values)

        Returns:
            (slope, intercept) of the best-fit line
        """
        n = len(x)
        if n < 2:
            return 0.0, 0.0

        mean_x = sum(x) / n
        mean_y = sum(y) / n

        numerator = sum((x[i] - mean_x) * (y[i] - mean_y) for i in range(n))
        denominator = sum((x[i] - mean_x) ** 2 for i in range(n))

        if denominator == 0:
            return 0.0, mean_y

        slope = numerator / denominator
        intercept = mean_y - slope * mean_x
        return slope, intercept

    def get_trend(
        self,
        model_name: str,
        variable: str,
        metric_type: str,
    ) -> LongitudinalTrend:
        """Analyze trend from historical records.

        Uses simple linear regression to detect trends. For MAE/RMSE:
        - Negative slope = improving (error decreasing over time)
        - Positive slope = declining (error increasing over time)
        - Near-zero slope = stable

        For Bias:
        - Slope approaching zero from non-zero = improving
        - Slope moving away from zero = declining

        Args:
            model_name: Model name
            variable: Variable evaluated
            metric_type: Metric type

        Returns:
            LongitudinalTrend with analysis results
        """
        records = self.get_records(model_name, variable, metric_type)

        trend = LongitudinalTrend(
            model_name=model_name,
            variable=variable,
            metric_type=metric_type,
            records=records,
        )

        if len(records) < 2:
            trend.trend_description = "Insufficient data for trend analysis (need >= 2 records)"
            return trend

        # Extract values and normalized timestamps
        valid_records = [(r.timestamp, r.value) for r in records if r.value is not None]
        if len(valid_records) < 2:
            trend.trend_description = "Insufficient valid values for trend analysis"
            return trend

        # Normalize timestamps to fractional days from first record
        base_time = valid_records[0][0]
        x = [(ts - base_time).total_seconds() / 86400.0 for ts, _ in valid_records]
        y = [val for _, val in valid_records]

        # Compute summary statistics
        trend.mean_value = sum(y) / len(y)
        trend.min_value = min(y)
        trend.max_value = max(y)

        # Linear regression
        slope, intercept = self._simple_linear_regression(x, y)

        # Compute R-squared for goodness of fit
        mean_y = sum(y) / len(y)
        ss_tot = sum((yi - mean_y) ** 2 for yi in y)
        ss_res = sum((yi - (slope * xi + intercept)) ** 2 for xi, yi in zip(x, y, strict=False))
        r_squared = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

        trend.r_squared = round(r_squared, 4)
        trend.slope = round(slope, 6)

        # Determine trend direction using regression slope
        # Use a significance threshold: slope must be non-zero relative to the mean
        mean_abs_value = abs(trend.mean_value) if trend.mean_value is not None else 1.0
        relative_slope = abs(slope) / mean_abs_value if mean_abs_value > 0 else abs(slope)

        # Threshold for significance: 5% change per 100 days
        significance_threshold = 0.05 / 100.0

        if metric_type in ("mae", "rmse", "error_variance"):
            # Lower is better
            if relative_slope > significance_threshold and slope < 0:
                trend.improving = True
                trend.trend_description = (
                    f"Improving (linear regression): {metric_type} declining "
                    f"at {abs(slope):.4f}/day, R²={r_squared:.3f}"
                )
            elif relative_slope > significance_threshold and slope > 0:
                trend.declining = True
                trend.trend_description = (
                    f"Declining (linear regression): {metric_type} increasing "
                    f"at {slope:.4f}/day, R²={r_squared:.3f}"
                )
            else:
                trend.stable = True
                trend.trend_description = (
                    f"Stable (linear regression): {metric_type} changed by "
                    f"{abs(slope):.4f}/day (insignificant), R²={r_squared:.3f}"
                )
        elif metric_type == "bias":
            # Closer to zero is better — check if |slope| indicates improvement
            if relative_slope > significance_threshold and slope < 0:
                trend.improving = True
                trend.trend_description = (
                    f"Improving (linear regression): bias decreasing "
                    f"at {slope:.4f}/day, R²={r_squared:.3f}"
                )
            elif relative_slope > significance_threshold and slope > 0:
                trend.declining = True
                trend.trend_description = (
                    f"Declining (linear regression): bias increasing "
                    f"at {slope:.4f}/day, R²={r_squared:.3f}"
                )
            else:
                trend.stable = True
                trend.trend_description = (
                    f"Stable (linear regression): bias changed by "
                    f"{abs(slope):.4f}/day (insignificant), R²={r_squared:.3f}"
                )

        return trend

    def compare_longitudinal(
        self,
        model_a: str,
        model_b: str,
        variable: str,
        metric_type: str,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
    ) -> dict[str, Any]:
        """Compare longitudinal performance of two models.

        Args:
            model_a: First model name
            model_b: Second model name
            variable: Variable evaluated
            metric_type: Metric type
            start_date: Filter from this date
            end_date: Filter to this date

        Returns:
            Dict with comparison results and trend analysis
        """
        records_a = self.get_records(model_a, variable, metric_type, start_date, end_date)
        records_b = self.get_records(model_b, variable, metric_type, start_date, end_date)

        trend_a = self.get_trend(model_a, variable, metric_type)
        trend_b = self.get_trend(model_b, variable, metric_type)

        # Compare most recent values
        latest_a = records_a[-1].value if records_a else None
        latest_b = records_b[-1].value if records_b else None

        return {
            "model_a": model_a,
            "model_b": model_b,
            "variable": variable,
            "metric_type": metric_type,
            "records_a": len(records_a),
            "records_b": len(records_b),
            "latest_value_a": latest_a,
            "latest_value_b": latest_b,
            "trend_a": {
                "direction": "improving"
                if trend_a.improving
                else "declining"
                if trend_a.declining
                else "stable",
                "description": trend_a.trend_description,
                "slope": trend_a.slope,
                "r_squared": trend_a.r_squared,
            },
            "trend_b": {
                "direction": "improving"
                if trend_b.improving
                else "declining"
                if trend_b.declining
                else "stable",
                "description": trend_b.trend_description,
                "slope": trend_b.slope,
                "r_squared": trend_b.r_squared,
            },
        }
