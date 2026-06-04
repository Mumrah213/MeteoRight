"""Phase 3: Forecast Evaluator — orchestrates the verification pipeline.

Orchestrates the full verification workflow:
  1. Align forecasts and observations
  2. Compute errors
  3. Compute metrics with grouping
  4. Analyze lead-time degradation
  5. Produce EvaluationResult output

Usage:
    evaluator = ForecastEvaluator()
    result = evaluator.evaluate(
        forecasts=forecast_records,
        observations=observation_records,
        group_by="lead_time",
    )
"""


from typing import Any

from ..canonical.models import CanonicalForecastRecord
from .alignment import AlignmentEngine
from .degradation import LeadTimeDegradationAnalyzer
from .error_computation import ErrorComputation
from .metrics import MetricsEngine
from .models import EvaluationResult, ForecastEvaluation


class ForecastEvaluatorError(Exception):
    """Raised when evaluation fails."""


class ForecastEvaluator:
    """Orchestrate the full forecast verification pipeline.

    Core workflow:
    1. Create VerificationInput from forecasts and observations
    2. Align on (latitude, longitude, valid_time, variable)
    3. Compute errors
    4. Compute metrics (MAE, RMSE, bias, variance) with grouping
    5. Analyze lead-time degradation
    6. Return EvaluationResult

    All outputs are structured, traceable, and explicit about gaps.
    """

    def __init__(
        self,
        alignment_engine: AlignmentEngine | None = None,
        metrics_engine: MetricsEngine | None = None,
        degradation_analyzer: LeadTimeDegradationAnalyzer | None = None,
    ) -> None:
        """Initialize the evaluator with optional sub-components.

        Args:
            alignment_engine: Optional AlignmentEngine instance
            metrics_engine: Optional MetricsEngine instance
            degradation_analyzer: Optional LeadTimeDegradationAnalyzer instance
        """
        self.alignment_engine = alignment_engine or AlignmentEngine()
        self.metrics_engine = metrics_engine or MetricsEngine()
        self.degradation_analyzer = degradation_analyzer or LeadTimeDegradationAnalyzer()
        self.error_computation = ErrorComputation()

    def evaluate(
        self,
        forecasts: list[CanonicalForecastRecord],
        observations: list[CanonicalForecastRecord],
        group_by: str | None = "lead_time",
        degradation_metric: str = "mae",
    ) -> EvaluationResult:
        """Run full verification pipeline.

        Args:
            forecasts: Forecast records (is_observation=False)
            observations: Observation records (is_observation=True)
            group_by: Metric grouping dimension
                      ("lead_time", "variable", "location", "run_time", None)
            degradation_metric: Metric for degradation analysis
                                ("mae", "rmse", "bias")

        Returns:
            EvaluationResult with alignments, metrics, and degradation profiles
        """
        # Validate inputs
        forecast_records = [f for f in forecasts if not f.is_observation]
        obs_records = [o for o in observations if o.is_observation]

        if not forecast_records:
            raise ForecastEvaluatorError("No forecast records to evaluate")

        # Step 1: Align forecasts and observations
        alignments = self.alignment_engine.align(forecast_records, obs_records)

        # Step 2: Count matches and gaps
        matched = [a for a in alignments if a.is_fully_matched]
        forecast_only = [a for a in alignments if a.match_status == "forecast_only"]
        observation_only = [a for a in alignments if a.match_status == "observation_only"]

        # Step 3: Compute metrics with grouping
        evaluations: list[ForecastEvaluation] = []
        # group_by=None means aggregate (no grouping), not "skip metrics"
        evaluations.extend(self.metrics_engine.compute_all_metrics(alignments, group_by))

        # Step 4: Analyze lead-time degradation
        degradation_profiles = self.degradation_analyzer.analyze_all_variables(alignments)

        # Build result
        return EvaluationResult(
            input_metadata={
                "forecast_count": len(forecast_records),
                "observation_count": len(obs_records),
                "group_by": group_by,
                "degradation_metric": degradation_metric,
            },
            alignments=alignments,
            matched_count=len(matched),
            forecast_only_count=len(forecast_only),
            observation_only_count=len(observation_only),
            evaluations=evaluations,
            degradation_profiles=degradation_profiles,
        )

    def evaluate_single_variable(
        self,
        forecasts: list[CanonicalForecastRecord],
        observations: list[CanonicalForecastRecord],
        variable: str,
        group_by: str | None = "lead_time",
    ) -> EvaluationResult:
        """Evaluate a single variable.

        Args:
            forecasts: Forecast records (is_observation=False)
            observations: Observation records (is_observation=True)
            variable: Variable name to evaluate
            group_by: Metric grouping dimension

        Returns:
            EvaluationResult with metrics for one variable only
        """
        forecast_records = [f for f in forecasts if not f.is_observation]
        obs_records = [o for o in observations if o.is_observation]

        # Filter to single variable
        forecast_records = [f for f in forecast_records if f.variable == variable]
        obs_records = [o for o in obs_records if o.variable == variable]

        if not forecast_records or not obs_records:
            raise ForecastEvaluatorError(f"No records for variable '{variable}'")

        return self.evaluate(forecast_records, obs_records, group_by)

    def evaluate_multiple_runs(
        self,
        all_forecasts: list[CanonicalForecastRecord],
        observations: list[CanonicalForecastRecord],
        group_by: str = "run_time",
    ) -> EvaluationResult:
        """Evaluate across multiple forecast runs.

        Compares all forecast runs against the same observation set.

        Args:
            all_forecasts: All forecast records from multiple runs
            observations: Observation records (is_observation=True)
            group_by: Must be "run_time" for this method

        Returns:
            EvaluationResult with metrics grouped by run_time
        """
        return self.evaluate(
            all_forecasts,
            observations,
            group_by=group_by,
        )

    def compare_runs(
        self,
        forecast_run_a: list[CanonicalForecastRecord],
        forecast_run_b: list[CanonicalForecastRecord],
        observations: list[CanonicalForecastRecord],
    ) -> dict[str, Any]:
        """Compare two forecast runs against the same observations.

        Computes metrics for both runs and compares them.

        IMPORTANT: This method validates that the two runs do not share
        overlapping valid_time windows with different run_times.
        Overlapping valid_times with distinct run_times indicate
        the same atmospheric event being forecast by different runs,
        which is valid for comparison but should be noted.

        Args:
            forecast_run_a: First forecast run records
            forecast_run_b: Second forecast run records
            observations: Observation records

        Returns:
            Dict with metrics for both runs, comparison, and
            overlap warnings if runs share valid_time windows.
        """
        # Validate runs are distinct
        run_a_times = {r.run_time for r in forecast_run_a}
        run_b_times = {r.run_time for r in forecast_run_b}

        is_same_run = run_a_times == run_b_times and len(run_a_times) == 1

        # Check for overlapping valid_times
        a_valid_times = {r.valid_time for r in forecast_run_a}
        b_valid_times = {r.valid_time for r in forecast_run_b}
        shared_valid_times = a_valid_times & b_valid_times

        overlap_info = {
            "is_same_run": is_same_run,
            "shared_valid_times": len(shared_valid_times),
            "warning": None,
        }

        if is_same_run:
            overlap_info["warning"] = (
                "RUN_A and RUN_B appear to be the same forecast run "
                "(identical run_times). Comparison is not meaningful."
            )
        elif shared_valid_times:
            overlap_info["warning"] = (
                f"Runs share {len(shared_valid_times)} valid_time points. "
                "These represent the same atmospheric events forecast by "
                "different runs — this is expected for operational "
                "verification but should not be treated as independent samples."
            )

        # Evaluate each run
        result_a = self.evaluate(forecast_run_a, observations, group_by=None)
        result_b = self.evaluate(forecast_run_b, observations, group_by=None)

        # Find corresponding evaluations
        evals_a = [e for e in result_a.evaluations if e.metric_type == "mae"]
        evals_b = [e for e in result_b.evaluations if e.metric_type == "mae"]

        comparison: dict[str, Any] = {
            "overlap": overlap_info,
            "run_a": {
                "forecast_count": len(forecast_run_a),
                "observation_count": len(observations),
                "match_count": result_a.matched_count,
            },
            "run_b": {
                "forecast_count": len(forecast_run_b),
                "observation_count": len(observations),
                "match_count": result_b.matched_count,
            },
        }

        for ea in evals_a:
            for eb in evals_b:
                if ea.variable == eb.variable:
                    comparison[ea.variable] = {
                        "run_a_mae": ea.value,
                        "run_b_mae": eb.value,
                        "difference": (ea.value or 0) - (eb.value or 0),
                        "run_b_better": (eb.value or float("inf")) < (ea.value or float("inf")),
                        "run_a_stable": ea.is_stable,
                        "run_b_stable": eb.is_stable,
                    }

        return comparison
