"""Historical Skill Profile System.

Tracks historical performance of models by:
- model
- variable
- lead_time
- season
- region

Profiles are persistent, temporally updatable, and causal
(only use data available before the evaluation time).

Usage:
    profiles = SkillProfileStore()
    profiles.update_from_evaluation(evaluation, model_name="IFS")
    profile = profiles.get("IFS")
"""


from collections import defaultdict
from datetime import datetime
from typing import Any

from ..verification.models import ForecastEvaluation
from .models import (
    SkillProfile,
    SkillProfileEntry,
)

# ── Skill Profile Store ────────────────────────────────────────────────────


class SkillProfileStore:
    """Persistent store for model skill profiles.

    Organized by: model_name → SkillProfile → (variable, lead_time, season, region) → entry

    Supports:
    - Adding new profile entries from evaluations
    - Temporal updates (new evaluations extend the profile)
    - Querying by context (variable, lead_time, season, region)
    - Retrieving best models per context

    CRITICAL: Profile entries must be created with data available BEFORE
    the target evaluation time to avoid lookahead bias.
    """

    def __init__(self) -> None:
        self._profiles: dict[str, SkillProfile] = {}

    @property
    def model_names(self) -> list[str]:
        """Return all model names with profiles."""
        return sorted(self._profiles.keys())

    def get_profile(self, model_name: str) -> SkillProfile | None:
        """Get the skill profile for a model."""
        return self._profiles.get(model_name)

    def add_model(self, model_name: str) -> SkillProfile:
        """Register a new model in the store. Returns its profile."""
        if model_name not in self._profiles:
            self._profiles[model_name] = SkillProfile(model_name=model_name)
        return self._profiles[model_name]

    def remove_model(self, model_name: str) -> None:
        """Remove a model's profile."""
        self._profiles.pop(model_name, None)

    def update_from_evaluation(
        self,
        evaluation: ForecastEvaluation,
        model_name: str,
        region: str = "",
    ) -> int:
        """Add or update a skill profile entry from an evaluation result.

        The entry is keyed by (variable, lead_time, season, region).
        If an entry already exists with the same key, it is replaced
        (newer evaluations overwrite older ones).

        Args:
            evaluation: A ForecastEvaluation from the verification pipeline.
            model_name: Which model this evaluation belongs to.
            region: Optional region label.
        """
        profile = self.add_model(model_name)

        # Extract season from evaluation metadata or time_range
        season = ""
        if evaluation.metadata:
            season = evaluation.metadata.get("season", "")
        if not season and evaluation.time_range:
            mid_time = (evaluation.time_range[0] + evaluation.time_range[1]) / 2
            # Derive season from mid-point time
            month = mid_time.month
            if month in (12, 1, 2):
                season = "DJF"
            elif month in (3, 4, 5):
                season = "MAM"
            elif month in (6, 7, 8):
                season = "JJA"
            elif month in (9, 10, 11):
                season = "SON"

        entry = SkillProfileEntry(
            model_name=model_name,
            variable=evaluation.variable,
            lead_hours=evaluation.lead_time,
            season=season,
            region=region,
            metric_type=evaluation.metric_type,
            value=evaluation.value,
            sample_size=evaluation.sample_size,
            coverage_fraction=evaluation.coverage_fraction,
            is_stable=evaluation.is_stable,
            min_sample_size=evaluation.min_sample_size,
            updated_at=datetime.now(),
            evaluation_window=evaluation.time_range,
        )

        profile.add_entry(
            variable=evaluation.variable,
            lead_hours=evaluation.lead_time,
            season=season,
            region=region,
            entry=entry,
        )

        return 1

    def update_from_evaluations(
        self,
        evaluations: list[ForecastEvaluation],
        model_name: str,
        region: str = "",
    ) -> int:
        """Add or update multiple profile entries at once.

        Args:
            evaluations: List of ForecastEvaluation objects.
            model_name: Which model these evaluations belong to.
            region: Optional region label.

        Returns:
            Number of entries updated.
        """
        count = 0
        for ev in evaluations:
            self.update_from_evaluation(ev, model_name, region)
            count += 1
        return count

    def get_best_model(
        self,
        variable: str,
        lead_hours: int | None = None,
        season: str = "",
        region: str = "",
        metric_type: str = "mae",
    ) -> tuple[str, float | None] | None:
        """Find the best model for a given context.

        Searches all models and returns the one with the lowest MAE/RMSE.
        Only considers stable entries with non-null values.

        Args:
            variable: Variable to look up.
            lead_hours: Optional lead time filter.
            season: Optional season filter.
            region: Optional region filter.
            metric_type: Metric type (mae, rmse, bias).

        Returns:
            (model_name, value) tuple, or None if no data.
        """
        best_model = None
        best_value = None

        for profile in self._profiles.values():
            entries = profile.get_entries_for_context(
                variable=variable,
                lead_hours=lead_hours,
                season=season,
                region=region,
                fallback=True,
            )
            for entry in entries:
                if entry.metric_type != metric_type:
                    continue
                if entry.value is None:
                    continue
                if not entry.is_stable:
                    continue  # Skip unstable entries for ranking
                if best_value is None or entry.value < best_value:
                    best_value = entry.value
                    best_model = profile.model_name

        if best_model is not None:
            return (best_model, best_value)
        return None

    def get_skill_ranking(
        self,
        variable: str,
        lead_hours: int | None = None,
        season: str = "",
        region: str = "",
        metric_type: str = "mae",
    ) -> list[tuple[str, float | None]]:
        """Rank all models for a given context.

        Models are ranked by lowest error (MAE/RMSE).
        Only stable entries with non-null values are included.

        Args:
            variable: Variable to look up.
            lead_hours: Optional lead time filter.
            season: Optional season filter.
            region: Optional region filter.
            metric_type: Metric type.

        Returns:
            List of (model_name, value) tuples, sorted by value ascending.
        """
        rankings: list[tuple[str, float | None]] = []

        for profile in self._profiles.values():
            entries = profile.get_entries_for_context(
                variable=variable,
                lead_hours=lead_hours,
                season=season,
                region=region,
                fallback=True,
            )
            for entry in entries:
                if entry.metric_type != metric_type:
                    continue
                if entry.value is None:
                    continue
                if not entry.is_stable:
                    continue
                rankings.append((profile.model_name, entry.value))
                break  # Only one entry per model (the most specific)

        # Sort by value ascending (lower error = better)
        rankings.sort(key=lambda x: x[1] if x[1] is not None else float("inf"))
        return rankings

    def get_all_profiles(self) -> dict[str, SkillProfile]:
        """Return all profiles."""
        return dict(self._profiles)

    def get_model_summary(self, model_name: str) -> dict[str, Any]:
        """Get a human-readable summary of a model's skill profile."""
        profile = self._profiles.get(model_name)
        if not profile:
            return {}

        summaries: dict[str, Any] = {
            "model_name": model_name,
            "total_entries": len(profile.entries),
            "variables": defaultdict(list),
        }

        for entry in profile.entries.values():
            summaries["variables"][entry.variable].append(
                {
                    "lead_hours": entry.lead_hours,
                    "season": entry.season,
                    "region": entry.region,
                    "metric_type": entry.metric_type,
                    "value": entry.value,
                    "sample_size": entry.sample_size,
                    "is_stable": entry.is_stable,
                }
            )

        # Convert defaultdict to dict
        summaries["variables"] = dict(summaries["variables"])
        return summaries

    def to_dict(self) -> dict[str, Any]:
        """Serialize all profiles to a dict (for persistence)."""
        return {
            model_name: {
                "model_name": profile.model_name,
                "entries": {
                    f"{e.variable}_{e.lead_hours}_{e.season}_{e.region}": e.model_dump()
                    for e in profile.entries.values()
                },
            }
            for model_name, profile in self._profiles.items()
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SkillProfileStore:
        """Deserialize profiles from a dict."""
        store = cls()
        for model_name, profile_data in data.items():
            store.add_model(model_name)
            profile = store._profiles[model_name]
            for key_str, entry_data in profile_data["entries"].items():
                entry = SkillProfileEntry(**entry_data)
                parts = key_str.rsplit("_", 3)
                if len(parts) == 4:
                    variable, lead, season, region = parts
                    lead_hours = int(lead) if lead != "None" else None
                else:
                    continue  # Skip malformed keys
                profile.add_entry(variable, lead_hours, season, region, entry)
        return store
