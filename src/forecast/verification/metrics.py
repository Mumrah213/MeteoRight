"""Phase 3: Metrics Engine.

Computes forecast verification metrics with grouping support:
  - MAE (Mean Absolute Error)
  - RMSE (Root Mean Squared Error)
  - Bias
  - Error Variance

All metrics support grouping by lead_time, variable, location, run_time.

Example usage:
    engine = MetricsEngine()
    mae_by_lead = engine.compute_mae(alignments, group_by="lead_time")
"""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime
from typing import Any

from .alignment import AlignmentRecord
from .error_computation import ErrorComputation
from .models import ForecastEvaluation


class MetricsEngineError(Exception):
    """Raised when metric computation fails."""


class MetricsEngine:
    """Compute forecast verification metrics with grouping.

    All metrics are computed from AlignmentRecord objects produced
    by the AlignmentEngine.

    Supported groupings:
    - "lead_time": group by lead_hours
    - "variable": group by variable name
    - "location": group by (latitude, longitude)
    - "run_time": group by run_time
    - None: aggregate across all records

    Coverage-aware metrics:
    - Each ForecastEvaluation includes sample_size, total_count,
      coverage_fraction, and is_stable flags.
    - Minimum sample_size thresholds can be set via set_min_sample_size().
    """

    def __init__(self, min_sample_size: int = 1) -> None:
        """Initialize metrics engine.

        Args:
            min_sample_size: Minimum sample_size for is_stable=True.
                             If a group has fewer samples than this,
                             is_stable=False (metric may be unreliable).
        """
        self.error_computation = ErrorComputation()
        self._min_sample_size = min_sample_size

    def set_min_sample_size(self, n: int) -> None:
        """Set minimum sample size threshold for metric stability.

        Args:
            n: Minimum sample_size required for is_stable=True.
               A metric computed from fewer samples will have
               is_stable=False and should be flagged in output.
        """
        if n < 1:
            raise ValueError("min_sample_size must be >= 1")
        self._min_sample_size = n

    def get_min_sample_size(self) -> int:
        """Return current minimum sample size threshold."""
        return self._min_sample_size

    def compute_mae(
        self,
        alignments: list[AlignmentRecord],
        group_by: str | None = None,
    ) -> list[ForecastEvaluation]:
        """Compute Mean Absolute Error with optional grouping.

        MAE = mean(|forecast - observation|)

        Args:
            alignments: List of AlignmentRecord objects
            group_by: One of "lead_time", "variable", "location", "run_time"

        Returns:
            List of ForecastEvaluation objects
        """
        return self._compute_metric(alignments, "mae", group_by)

    def compute_rmse(
        self,
        alignments: list[AlignmentRecord],
        group_by: str | None = None,
    ) -> list[ForecastEvaluation]:
        """Compute Root Mean Squared Error with optional grouping.

        RMSE = sqrt(mean((forecast - observation)²))

        Args:
            alignments: List of AlignmentRecord objects
            group_by: One of "lead_time", "variable", "location", "run_time"

        Returns:
            List of ForecastEvaluation objects
        """
        return self._compute_metric(alignments, "rmse", group_by)

    def compute_bias(
        self,
        alignments: list[AlignmentRecord],
        group_by: str | None = None,
    ) -> list[ForecastEvaluation]:
        """Compute bias with optional grouping.

        Bias = mean(forecast - observation)

        Args:
            alignments: List of AlignmentRecord objects
            group_by: One of "lead_time", "variable", "location", "run_time"

        Returns:
            List of ForecastEvaluation objects
        """
        return self._compute_metric(alignments, "bias", group_by)

    def compute_error_variance(
        self,
        alignments: list[AlignmentRecord],
        group_by: str | None = None,
    ) -> list[ForecastEvaluation]:
        """Compute error variance with optional grouping.

        Error Variance = mean((error - mean(error))²)

        Args:
            alignments: List of AlignmentRecord objects
            group_by: One of "lead_time", "variable", "location", "run_time"

        Returns:
            List of ForecastEvaluation objects
        """
        return self._compute_metric(alignments, "error_variance", group_by)

    def compute_by_season(
        self,
        alignments: list[AlignmentRecord],
        metric_type: str = "mae",
    ) -> list[ForecastEvaluation]:
        """Compute metrics grouped by meteorological season (DJF, MAM, JJA, SON).

        CRITICAL: Never aggregate across seasons blindly. Seasonal
        differences can dominate inter-run comparisons.

        Args:
            alignments: List of AlignmentRecord objects (must have
                       temporal metadata: month, season derived)
            metric_type: One of "mae", "rmse", "bias", "error_variance"

        Returns:
            List of ForecastEvaluation objects, one per active season.
        """
        from .models import _month_to_season

        groups: dict[str, list[AlignmentRecord]] = defaultdict(list)
        for a in alignments:
            season = getattr(a, "season", "")
            if not season:
                season = _month_to_season(a.valid_time.month)
            groups[season].append(a)

        results: list[ForecastEvaluation] = []
        for season_code in sorted(groups.keys()):
            group_alignments = groups[season_code]
            result = self._compute_single_metric(
                group_alignments, metric_type, group_key=season_code
            )
            # Override season metadata
            result.metadata["season"] = season_code
            if result.time_range:
                result.metadata["season_start"] = result.time_range[0].isoformat()
                result.metadata["season_end"] = result.time_range[1].isoformat()
            results.append(result)

        return results

    def compute_by_month(
        self,
        alignments: list[AlignmentRecord],
        metric_type: str = "mae",
    ) -> list[ForecastEvaluation]:
        """Compute metrics grouped by calendar month.

        Args:
            alignments: List of AlignmentRecord objects
            metric_type: One of "mae", "rmse", "bias", "error_variance"

        Returns:
            List of ForecastEvaluation objects, one per active month.
        """
        groups: dict[int, list[AlignmentRecord]] = defaultdict(list)
        for a in alignments:
            month = getattr(a, "month", a.valid_time.month)
            groups[month].append(a)

        results: list[ForecastEvaluation] = []
        for month_num in sorted(groups.keys()):
            group_alignments = groups[month_num]
            result = self._compute_single_metric(group_alignments, metric_type, group_key=month_num)
            result.metadata["month"] = month_num
            results.append(result)

        return results

    def compute_all_metrics(
        self,
        alignments: list[AlignmentRecord],
        group_by: str | None = None,
    ) -> list[ForecastEvaluation]:
        """Compute all metrics in one pass.

        Args:
            alignments: List of AlignmentRecord objects
            group_by: One of "lead_time", "variable", "location", "run_time"

        Returns:
            List of ForecastEvaluation objects (4 per group)
        """
        results: list[ForecastEvaluation] = []
        for metric_type in ("mae", "rmse", "bias", "error_variance"):
            results.extend(self._compute_metric(alignments, metric_type, group_by))
        return results

    def _compute_metric(
        self,
        alignments: list[AlignmentRecord],
        metric_type: str,
        group_by: str | None = None,
    ) -> list[ForecastEvaluation]:
        """Internal: compute one metric with optional grouping."""
        if not alignments:
            return []

        # Group alignments
        if group_by == "lead_time":
            groups = self._group_by_lead_time(alignments)
        elif group_by == "variable":
            groups = self._group_by_variable(alignments)
        elif group_by == "location":
            groups = self._group_by_location(alignments)
        elif group_by == "run_time":
            groups = self._group_by_run_time(alignments)
        else:
            # No grouping — one aggregate result
            result = self._compute_single_metric(alignments, metric_type)
            if result.value is not None:
                return [result]
            return []

        # Compute metric for each group
        results: list[ForecastEvaluation] = []
        for group_key, group_alignments in groups.items():
            result = self._compute_single_metric(group_alignments, metric_type, group_key)
            # Skip groups with no matched data (None metric value)
            if result.value is not None:
                results.append(result)

        return results

    def _compute_single_metric(
        self,
        alignments: list[AlignmentRecord],
        metric_type: str,
        group_key: tuple | int | str | None = None,
    ) -> ForecastEvaluation:
        """Compute one metric for one group of alignments."""
        stats = ErrorComputation.compute_error_statistics(alignments)

        # Extract lead_time from first alignment in group
        lead_time = None
        if alignments:
            lead_time = alignments[0].lead_hours

        # Extract location from first alignment
        location = None
        if alignments:
            a = alignments[0]
            location = (a.latitude, a.longitude)

        # Extract time range from all alignments
        time_range = None
        if alignments:
            times = [a.valid_time for a in alignments]
            time_range = (min(times), max(times))

        # Determine variable_type from variable name
        variable_type = self._infer_variable_type(alignments[0].variable if alignments else "")

        # Build ForecastEvaluation with coverage tracking
        return ForecastEvaluation(
            variable=alignments[0].variable if alignments else "",
            lead_time=lead_time,
            metric_type=metric_type,
            value=stats.get(metric_type),
            sample_size=stats.get("sample_size", 0),
            total_count=len(alignments),
            coverage_fraction=stats.get("sample_size", 0) / len(alignments) if alignments else 0.0,
            min_sample_size=self._min_sample_size,
            variable_type=variable_type,
            location=location,
            time_range=time_range,
            metadata={"group_key": group_key},
        )

    @staticmethod
    def _infer_variable_type(variable_name: str) -> str | None:
        """Infer semantic variable type from variable name.

        Used for distinguishing metric semantics:
        - Temperature errors are absolute (Kelvin/Celsius)
        - Precipitation errors have different distribution properties
        - Humidity errors are bounded (0-100%)

        Args:
            variable_name: Variable name like "temperature_2m",
                          "precipitation_total", "relative_humidity_2m"

        Returns:
            Inferred type: "temperature", "precipitation", "humidity",
            "pressure", "wind", or None
        """
        name_lower = variable_name.lower()
        if "temp" in name_lower:
            return "temperature"
        if "precip" in name_lower:
            return "precipitation"
        if "humidity" in name_lower or "humid" in name_lower:
            return "humidity"
        if "pressure" in name_lower or "pres" in name_lower:
            return "pressure"
        if "wind" in name_lower:
            return "wind"
        return None

    def _group_by_lead_time(
        self, alignments: list[AlignmentRecord]
    ) -> dict[int, list[AlignmentRecord]]:
        """Group alignments by lead_hours."""
        groups: dict[int, list[AlignmentRecord]] = defaultdict(list)
        for a in alignments:
            groups[a.lead_hours].append(a)
        return dict(groups)

    def _group_by_variable(
        self, alignments: list[AlignmentRecord]
    ) -> dict[str, list[AlignmentRecord]]:
        """Group alignments by variable name."""
        groups: dict[str, list[AlignmentRecord]] = defaultdict(list)
        for a in alignments:
            groups[a.variable].append(a)
        return dict(groups)

    def _group_by_location(
        self, alignments: list[AlignmentRecord]
    ) -> dict[tuple[float, float], list[AlignmentRecord]]:
        """Group alignments by (latitude, longitude)."""
        groups: dict[tuple[float, float], list[AlignmentRecord]] = defaultdict(list)
        for a in alignments:
            key = (a.latitude, a.longitude)
            groups[key].append(a)
        return dict(groups)

    def _group_by_run_time(
        self, alignments: list[AlignmentRecord]
    ) -> dict[datetime | None, list[AlignmentRecord]]:
        """Group alignments by forecast run_time."""
        groups: dict[datetime | None, list[AlignmentRecord]] = defaultdict(list)
        for a in alignments:
            if a.forecast_run_time:
                groups[a.forecast_run_time].append(a)
        return dict(groups)

    def get_metric_by_group(
        self,
        alignments: list[AlignmentRecord],
        metric_type: str,
        group_by: str,
        group_key: Any,
    ) -> ForecastEvaluation | None:
        """Get a specific metric value for one group.

        Args:
            alignments: All alignment records
            metric_type: One of "mae", "rmse", "bias", "error_variance"
            group_by: Grouping dimension
            group_key: Key to look up (depends on group_by)

        Returns:
            ForecastEvaluation or None if group not found
        """
        all_metrics = self._compute_metric(alignments, metric_type, group_by)
        for m in all_metrics:
            if m.metadata.get("group_key") == group_key:
                return m
        return None
