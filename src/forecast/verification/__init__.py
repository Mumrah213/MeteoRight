"""Forecast verification primitives.

This module provides:

1. AlignmentEngine — matches forecasts against observations by key
2. ErrorComputation — computes error metrics (bias, MAE, RMSE, variance)
3. MetricsEngine — aggregates errors into grouped evaluations

Core principle:
  error = forecast_value - observation_value

All outputs are structured, traceable, and explicit about gaps.
"""

from .alignment import AlignmentEngine
from .error_computation import ErrorComputation
from .metrics import MetricsEngine
from .models import (
    AlignmentRecord,
    EvaluationResult,
    ForecastEvaluation,
    VerificationInput,
)

__all__ = [
    "AlignmentEngine",
    "AlignmentRecord",
    "ErrorComputation",
    "EvaluationResult",
    "ForecastEvaluation",
    "MetricsEngine",
    "VerificationInput",
]
