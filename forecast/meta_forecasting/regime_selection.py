"""Regime-Aware Model Selection.

Supports adaptive model selection based on:
- Season (DJF, MAM, JJA, SON)
- Region (e.g., "northern_europe", "tropical", "mid_latitudes")
- Weather regime (future extensibility: storm, precipitation_event, extreme_weather)

Architecture is designed for future expansion:
- Add new condition types without breaking existing logic
- Each regime has its own selection rules
- Rules are explicit and explainable

Usage:
    selector = RegimeAwareModelSelector(skill_store)
    selector.add_season_rule("JJA", "temperature_2m", recommended=["IFS", "ICON"])
    selector.add_region_rule("tropical", "precipitation", recommended=["GFS"])
    selection = selector.select(
        model_names=["IFS", "GFS", "ICON"],
        context={"season": "JJA", "region": "tropical"},
    )
"""


from typing import Any

from .models import (
    RegimeAwareSelection,
    RegimeCondition,
)
from .skill_profiles import SkillProfileStore

# ── Regime-Aware Selector ─────────────────────────────────────────────────


class RegimeAwareModelSelector:
    """Adaptive model selection by regime/conditions.

    Selects which models to trust most given the current context
    (season, region, weather regime).

    Selection rules are:
    1. Explicit recommendations from regime conditions
    2. Historical skill as tiebreaker
    3. Default: all models equally weighted

    All selections are explainable with full rationale.
    """

    def __init__(
        self,
        skill_store: SkillProfileStore | None = None,
    ) -> None:
        """Initialize the regime-aware selector.

        Args:
            skill_store: Pre-populated skill profile store for tiebreaking.
        """
        self.skill_store = skill_store or SkillProfileStore()
        # Condition type → variable → list of conditions
        self._rules: dict[str, dict[str, list[RegimeCondition]]] = {
            "season": {},
            "region": {},
            "storm": {},
            "precipitation_event": {},
            "extreme_weather": {},
        }
        # Model recommendations per condition chain
        self._recommendations: dict[str, list[str]] = {}

    def add_season_rule(
        self,
        season: str,
        variable: str,
        recommended: list[str] | None = None,
        excluded: list[str] | None = None,
        condition_metadata: dict[str, Any] | None = None,
    ) -> None:
        """Add a seasonal rule for model selection.

        Args:
            season: Season code (DJF, MAM, JJA, SON).
            variable: Variable this rule applies to.
            recommended: Model names recommended for this context.
            excluded: Model names excluded for this context.
            condition_metadata: Additional condition metadata.
        """
        key = f"{season}_{variable}"
        self._rules.setdefault("season", {}).setdefault(variable, []).append(
            RegimeCondition(
                condition_type="season",
                condition_value=season,
                condition_metadata=condition_metadata or {},
            )
        )
        if recommended:
            self._recommendations[key] = recommended
        if excluded:
            self._recommendations.setdefault(f"{key}_excluded", [])
            self._recommendations[f"{key}_excluded"] = excluded

    def add_region_rule(
        self,
        region: str,
        variable: str,
        recommended: list[str] | None = None,
        excluded: list[str] | None = None,
        condition_metadata: dict[str, Any] | None = None,
    ) -> None:
        """Add a regional rule for model selection.

        Args:
            region: Region name (e.g., "northern_europe").
            variable: Variable this rule applies to.
            recommended: Model names recommended for this context.
            excluded: Model names excluded for this context.
            condition_metadata: Additional condition metadata.
        """
        key = f"{region}_{variable}"
        self._rules.setdefault("region", {}).setdefault(variable, []).append(
            RegimeCondition(
                condition_type="region",
                condition_value=region,
                condition_metadata=condition_metadata or {},
            )
        )
        if recommended:
            self._recommendations[key] = recommended
        if excluded:
            self._recommendations.setdefault(f"{key}_excluded", [])
            self._recommendations[f"{key}_excluded"] = excluded

    def select(
        self,
        model_names: list[str],
        context: dict[str, Any],
        variable: str | None = None,
    ) -> RegimeAwareSelection:
        """Select models based on current regime/conditions.

        Searches all applicable rules and returns the best model
        selection with full rationale.

        Selection priority:
        1. Explicit recommendations from matching conditions
        2. Historical skill rankings (from skill store)
        3. Default: all models equally weighted

        Args:
            model_names: All available models to choose from.
            context: Context dict with keys like "season", "region", etc.
            variable: Variable context (affects rule matching).

        Returns:
            RegimeAwareSelection with selected models and rationale.
        """
        selected: list[dict[str, Any]] = []
        excluded: list[str] = []
        conditions_applied: list[RegimeCondition] = []

        # Check all condition types
        for cond_type in ("season", "region", "storm", "precipitation_event", "extreme_weather"):
            rules = self._rules.get(cond_type, {})
            for var, conditions in rules.items():
                if variable is not None and var != variable:
                    continue

                for cond in conditions:
                    if cond.matches(context):
                        conditions_applied.append(cond)
                        key = f"{cond.condition_value}_{var}"

                        # Check recommendations
                        recs = self._recommendations.get(key, [])
                        if recs:
                            for model_name in recs:
                                if model_name in model_names:
                                    reason = (
                                        f"Recommended by {cond_type} rule "
                                        f"for {var} in {cond.condition_value}"
                                    )
                                    selected.append(
                                        {
                                            "model_name": model_name,
                                            "reason": reason,
                                            "weight": 1.0,
                                        }
                                    )

                        # Check exclusions
                        excluded_key = f"{key}_excluded"
                        excluded_models = self._recommendations.get(excluded_key, [])
                        for model_name in excluded_models:
                            if model_name in model_names and model_name not in excluded:
                                excluded.append(model_name)

        # If no explicit recommendations, fall back to historical skill
        if not selected and variable:
            skill_selections = self._fallback_to_skill(model_names, variable, context)
            for sel in skill_selections:
                selected.append(sel)

        # Build rationale
        rationale = self._build_rationale(selected, excluded, conditions_applied, context)

        return RegimeAwareSelection(
            context=context,
            selected_models=selected,
            excluded_models=excluded,
            rationale=rationale,
            regime_conditions_applied=conditions_applied,
        )

    def _fallback_to_skill(
        self,
        model_names: list[str],
        variable: str,
        context: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """Fall back to historical skill rankings.

        Returns the top-ranked model(s) for the given context.
        """
        season = context.get("season", "")
        region = context.get("region", "")

        ranking = self.skill_store.get_skill_ranking(
            variable=variable,
            season=season,
            region=region,
            metric_type="mae",
        )

        if not ranking:
            return []

        # Return the top model(s) with equal weight
        top_model = ranking[0][0]  # Best model name
        return [
            {"model_name": top_model, "reason": "Best historical skill in context", "weight": 1.0}
        ]

    @staticmethod
    def _build_rationale(
        selected: list[dict[str, Any]],
        excluded: list[str],
        conditions: list[RegimeCondition],
        context: dict[str, Any],
    ) -> str:
        """Build human-readable rationale for the selection."""
        parts = []

        if conditions:
            applied = [f"{c.condition_type}={c.condition_value}" for c in conditions]
            parts.append(f"Conditions applied: {', '.join(applied)}.")

        if excluded:
            parts.append(f"Excluded: {', '.join(excluded)}.")

        if selected:
            model_names = [s["model_name"] for s in selected]
            reasons = [s.get("reason", "") for s in selected]
            parts.append(
                f"Selected: {', '.join(model_names)}. {' | '.join(r for r in reasons if r)}."
            )
        else:
            parts.append("No explicit regime rules matched; all models considered equally.")

        return " ".join(parts)

    def clear_rules(self, condition_type: str | None = None) -> None:
        """Clear rules. If condition_type is None, clears all rules."""
        if condition_type is None:
            for key in self._rules:
                self._rules[key] = {}
        else:
            self._rules.pop(condition_type, None)
