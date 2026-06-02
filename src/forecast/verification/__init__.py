"""Phase 3: Forecast verification and evaluation layer.

This module provides:

1. ObservationProvider — fetches ERA5 observations via Archive API
2. AlignmentEngine — matches forecasts against observations by key
3. ErrorComputation — computes error metrics (bias, MAE, RMSE, variance)
4. LeadTimeDegradation — analyzes error growth with lead time
5. ForecastEvaluator — orchestrates the full verification pipeline

Core principle:
  error = forecast_value - observation_value

All outputs are structured, traceable, and explicit about gaps.
"""

from .alignment import AlignmentEngine
from .degradation import LeadTimeDegradationAnalyzer
from .error_computation import ErrorComputation
from .evaluator import ForecastEvaluator
from .metrics import MetricsEngine
from .models import (
    AlignmentRecord,
    EvaluationResult,
    ForecastEvaluation,
    VerificationInput,
)
from .observation_provider import ObservationProvider

__all__ = [
    "AlignmentEngine",
    "AlignmentRecord",
    "ErrorComputation",
    "EvaluationResult",
    "ForecastEvaluation",
    "ForecastEvaluator",
    "LeadTimeDegradationAnalyzer",
    "MetricsEngine",
    "ObservationProvider",
    "VerificationInput",
]
