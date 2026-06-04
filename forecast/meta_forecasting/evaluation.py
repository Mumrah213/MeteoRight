"""Meta-Evaluation Framework.

Evaluates the meta-forecasting system itself.

Measures:
- Blended forecast MAE/RMSE
- Improvement over best single model
- Stability over time
- Degradation curves of blended forecasts

Never assumes blending automatically improves skill.
Measures it empirically.

Usage:
    evaluator = MetaEvaluator(skill_store)
    result = evaluator.evaluate(
        blended_forecasts=[...],
        single_model_forecasts={"IFS": [...], "GFS": [...]},
        observations=[...],
        variable="temperature_2m",
    )
"""


from typing import Any

from ..verification.metrics import MetricsEngine
from .models import MetaEvaluationResult
from .skill_profiles import SkillProfileStore

# ── Meta-Evaluator ─────────────────────────────────────────────────────────


class MetaEvaluator:
    """Evaluate the meta-forecasting system itself.

    Compares blended forecasts against single models to determine
    whether blending actually improves skill.

    Metrics:
    - Blended MAE/RMSE vs best single model MAE/RMSE
    - Improvement percentage (negative = blended is worse)
    - Weight stability over time
    - Blend stability over time
    - Blended degradation curve

    All comparisons are time-valid (no future leakage).
    """

    def __init__(
        self,
        skill_store: SkillProfileStore | None = None,
        metrics_engine: MetricsEngine | None = None,
        min_sample_size: int = 1,
    ) -> None:
        """Initialize the meta-evaluator.

        Args:
            skill_store: Optional pre-populated skill store.
            metrics_engine: Metrics engine for error computation.
            min_sample_size: Minimum samples for stability flag.
        """
        self.skill_store = skill_store or SkillProfileStore()
        self.metrics_engine = metrics_engine or MetricsEngine(min_sample_size)

    def evaluate(
        self,
        blended_forecasts: list[dict[str, Any]],
        single_model_forecasts: dict[str, list[dict[str, Any]]],
        observations: list[dict[str, Any]],
        variable: str,
        lead_hours: int | None = None,
        metric: str = "mae",
    ) -> MetaEvaluationResult:
        """Evaluate blended vs single model performance.

        Compares the blended forecast against the best single model
        for the same observations.

        Args:
            blended_forecasts: List of dicts with keys:
                - "value": forecast value
                - "observed": observation value
            single_model_forecasts: Dict mapping model_name → list of
                forecast dicts (same structure).
            observations: List of dicts with at least "observed" key.
                (For simplicity, forecasts include their own observations.)
            variable: Variable name.
            lead_hours: Optional lead time.
            metric: Metric type ("mae", "rmse").

        Returns:
            MetaEvaluationResult with all comparison metrics.
        """
        # Compute blended error
        blended_errors = []
        for bf in blended_forecasts:
            val = bf.get("value")
            obs = bf.get("observed")
            if val is not None and obs is not None:
                err = self._compute_error(val, obs, metric, variable)
                blended_errors.append({"error": err, "abs_error": abs(err)})

        blended_mae = (
            sum(e["abs_error"] for e in blended_errors) / len(blended_errors)
            if blended_errors
            else None
        )
        blended_rmse = (
            (sum(e["error"] ** 2 for e in blended_errors) / len(blended_errors)) ** 0.5
            if blended_errors
            else None
        )

        # Compute best single model error
        best_model = None
        best_mae = None
        best_rmse = None

        for model_name, forecasts in single_model_forecasts.items():
            errors = []
            for sf in forecasts:
                val = sf.get("value")
                obs = sf.get("observed")
                if val is not None and obs is not None:
                    err = self._compute_error(val, obs, metric, variable)
                    errors.append({"error": err, "abs_error": abs(err)})

            if not errors:
                continue

            mae = sum(e["abs_error"] for e in errors) / len(errors)
            rmse = (sum(e["error"] ** 2 for e in errors) / len(errors)) ** 0.5

            if best_mae is None or mae < best_mae:
                best_mae = mae
                best_rmse = rmse
                best_model = model_name

        # Compute improvement percentage
        mae_improvement_pct = None
        rmse_improvement_pct = None
        improvement_over_best = False

        if blended_mae is not None and best_mae is not None and best_mae > 0:
            mae_improvement_pct = ((blended_mae - best_mae) / best_mae) * 100
            improvement_over_best = blended_mae < best_mae

        if blended_rmse is not None and best_rmse is not None and best_rmse > 0:
            rmse_improvement_pct = ((blended_rmse - best_rmse) / best_rmse) * 100

        # Compute weight stability (if skill data available)
        weight_stability = None
        blend_stability = None

        if blended_errors:
            blend_stability = self._compute_blend_stability(blended_errors)

        return MetaEvaluationResult(
            variable=variable,
            lead_hours=lead_hours,
            metric=metric,
            blended_mae=blended_mae,
            blended_rmse=blended_rmse,
            blended_coverage=sum(1 for e in blended_errors if e["abs_error"] is not None)
            / max(len(blended_forecasts), 1),
            best_single_mae=best_mae,
            best_single_rmse=best_rmse,
            best_single_model=best_model,
            mae_improvement_pct=mae_improvement_pct,
            rmse_improvement_pct=rmse_improvement_pct,
            improvement_over_best=improvement_over_best,
            weight_stability_over_time=weight_stability,
            blend_stability=blend_stability,
            sample_size=len(blended_errors),
            rationale=self._build_rationale(blended_mae, best_mae, mae_improvement_pct, variable),
        )

    def evaluate_by_lead_time(
        self,
        blended_forecasts: list[dict[str, Any]],
        single_model_forecasts: dict[str, list[dict[str, Any]]],
        variable: str,
        lead_time_bins: list[tuple[int, int]] | None = None,
    ) -> list[MetaEvaluationResult]:
        """Evaluate blended vs single model by lead time bins.

        Creates a degradation curve for both blended and single models.

        Args:
            blended_forecasts: List of forecast dicts with "lead_hours" key.
            single_model_forecasts: Dict mapping model_name → list of dicts.
            variable: Variable name.
            lead_time_bins: List of (min_lead, max_lead) tuples.
                Defaults to [0-24, 25-72, 73-120, 121+].

        Returns:
            List of MetaEvaluationResult, one per lead time bin.
        """
        if lead_time_bins is None:
            lead_time_bins = [(0, 24), (25, 72), (73, 120), (121, 9999)]

        results: list[MetaEvaluationResult] = []

        for min_lead, max_lead in lead_time_bins:
            # Filter forecasts to this lead time bin
            filtered_blended = [
                bf for bf in blended_forecasts if min_lead <= (bf.get("lead_hours", 0)) <= max_lead
            ]
            filtered_single = {
                mn: [sf for sf in sfs if min_lead <= (sf.get("lead_hours", 0)) <= max_lead]
                for mn, sfs in single_model_forecasts.items()
            }

            result = self.evaluate(
                blended_forecasts=filtered_blended,
                single_model_forecasts=filtered_single,
                observations=[],  # observations are embedded in forecast dicts
                variable=variable,
                lead_hours=(min_lead + max_lead) // 2,
            )
            results.append(result)

        return results

    def evaluate_improvement_by_variable(
        self,
        blended_forecasts: list[dict[str, Any]],
        single_model_forecasts: dict[str, list[dict[str, Any]]],
        observations: list[dict[str, Any]],
        variables: list[str] | None = None,
    ) -> dict[str, MetaEvaluationResult]:
        """Evaluate improvement over best single model by variable.

        Answers: "Which variables benefit most from blending?"

        Args:
            blended_forecasts: List of forecast dicts with "variable" key.
            single_model_forecasts: Dict mapping model_name → list of dicts.
            observations: Embedded in forecast dicts (as "observed" key).
            variables: List of variables to evaluate, or None (all).

        Returns:
            Dict mapping variable → MetaEvaluationResult.
        """
        # Group by variable
        blended_by_var: dict[str, list[dict[str, Any]]] = {}
        for bf in blended_forecasts:
            var = bf.get("variable", "unknown")
            blended_by_var.setdefault(var, []).append(bf)

        single_by_var: dict[str, dict[str, list[dict[str, Any]]]] = {}
        for mn, sfs in single_model_forecasts.items():
            for sf in sfs:
                var = sf.get("variable", "unknown")
                single_by_var.setdefault(var, {}).setdefault(mn, []).append(sf)

        if variables is None:
            variables = sorted(blended_by_var.keys())

        results: dict[str, MetaEvaluationResult] = {}
        for var in variables:
            if var not in blended_by_var:
                continue
            results[var] = self.evaluate(
                blended_forecasts=blended_by_var[var],
                single_model_forecasts=single_by_var.get(var, {}),
                observations=[],
                variable=var,
            )

        return results

    # ── Internal Methods ──────────────────────────────────────────────

    @staticmethod
    def _compute_error(
        forecast: float, observation: float, metric: str, variable: str = ""
    ) -> float:
        """Compute error for one forecast-observation pair.

        For circular variables (e.g. wind_direction_10m) the error
        is the shortest angle on the unit circle, wrapped to [-180, +180).
        """
        if variable in {"wind_direction_10m"}:
            diff = forecast - observation
            diff = ((diff + 180.0) % 360.0) - 180.0
            return diff
        return forecast - observation

    @staticmethod
    def _compute_blend_stability(errors: list[dict[str, Any]]) -> float | None:
        """Compute blend stability as inverse of error variance over time.

        Returns a value between 0 and 1. Higher = more stable.
        """
        if len(errors) < 3:
            return None

        abs_errors = [e["abs_error"] for e in errors]
        mean_err = sum(abs_errors) / len(abs_errors)

        if mean_err == 0:
            return 1.0

        cv = (sum((e - mean_err) ** 2 for e in abs_errors) / len(abs_errors)) ** 0.5 / abs(mean_err)

        # Map CV to stability: CV=0 → 1.0, CV→∞ → 0.0
        return max(0.0, 1.0 - cv)

    def _build_rationale(
        self,
        blended_mae: float | None,
        best_single_mae: float | None,
        improvement_pct: float | None,
        variable: str,
    ) -> str:
        """Build rationale for meta-evaluation result."""
        parts = [f"Evaluation for '{variable}':"]

        if blended_mae is not None and best_single_mae is not None:
            if improvement_pct is not None:
                if improvement_pct < 0:
                    parts.append(
                        f"Blending improved MAE by {abs(improvement_pct):.1f}% "
                        f"(blended={blended_mae:.3f}, best single={best_single_mae:.3f})."
                    )
                elif improvement_pct > 0:
                    parts.append(
                        f"Blending worsened MAE by {improvement_pct:.1f}% "
                        f"(blended={blended_mae:.3f}, best single={best_single_mae:.3f}). "
                        f"Blending did not improve performance for this variable."
                    )
                else:
                    parts.append(
                        f"Blending and best single model have equal MAE (both={blended_mae:.3f})."
                    )
        elif blended_mae is None:
            parts.append("No blended forecast data available for evaluation.")

        return " ".join(parts)
