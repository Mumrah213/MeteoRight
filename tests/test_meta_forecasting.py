"""Tests for the adaptive meta-forecasting layer.

Covers the pieces that carry real logic: dynamic weighting, weighted
blending, confidence scoring, ensemble agreement, and the meta-evaluation
that reports whether blending actually beat the best single model.
"""

from __future__ import annotations

import random
from datetime import UTC, datetime, timedelta

import pytest

from src.forecast.canonical.models import CanonicalForecastRecord
from src.forecast.meta_forecasting import (
    DynamicWeightingEngine,
    MetaEvaluator,
    MetaForecastingOrchestrator,
    SkillProfileStore,
)
from src.forecast.meta_forecasting.models import SkillProfileEntry

RUN_TIME = datetime(2024, 1, 1, tzinfo=UTC)
VARIABLE = "temperature_2m"
LEAD_HOURS = 24

# Model MAEs used across tests: icon_eu best, ecmwf worst.
MODEL_SKILL = {"icon_eu": 1.0, "gfs": 2.0, "ecmwf": 4.0}


def _record(model: str, value: float, lead_hours: int = LEAD_HOURS) -> CanonicalForecastRecord:
    return CanonicalForecastRecord(
        provider="openmeteo",
        model=model,
        run_time=RUN_TIME,
        valid_time=RUN_TIME + timedelta(hours=lead_hours),
        lead_hours=lead_hours,
        latitude=55.6,
        longitude=12.5,
        variable=VARIABLE,
        value=value,
        unit="°C",
    )


def _seeded_store(skill: dict[str, float] | None = None) -> SkillProfileStore:
    """Skill store pre-populated with one stable MAE entry per model."""
    store = SkillProfileStore()
    for model, mae in (skill or MODEL_SKILL).items():
        store.add_model(model)
        store.get_profile(model).add_entry(
            VARIABLE,
            LEAD_HOURS,
            "",
            "",
            SkillProfileEntry(
                model_name=model,
                variable=VARIABLE,
                lead_hours=LEAD_HOURS,
                metric_type="mae",
                value=mae,
                sample_size=500,
                is_stable=True,
            ),
        )
    return store


def _orchestrator() -> MetaForecastingOrchestrator:
    orch = MetaForecastingOrchestrator()
    for model, mae in MODEL_SKILL.items():
        orch.skill_store.add_model(model)
        orch.skill_store.get_profile(model).add_entry(
            VARIABLE,
            LEAD_HOURS,
            "",
            "",
            SkillProfileEntry(
                model_name=model,
                variable=VARIABLE,
                lead_hours=LEAD_HOURS,
                metric_type="mae",
                value=mae,
                sample_size=500,
                is_stable=True,
            ),
        )
    return orch


# ── Dynamic weighting ───────────────────────────────────────────────────────


class TestDynamicWeighting:
    def test_weights_sum_to_one(self):
        engine = DynamicWeightingEngine(skill_store=_seeded_store())
        weights = engine.compute_weights(
            list(MODEL_SKILL), variable=VARIABLE, lead_hours=LEAD_HOURS
        )
        assert sum(w.weight for w in weights) == pytest.approx(1.0)

    def test_weights_are_non_negative(self):
        engine = DynamicWeightingEngine(skill_store=_seeded_store())
        weights = engine.compute_weights(
            list(MODEL_SKILL), variable=VARIABLE, lead_hours=LEAD_HOURS
        )
        assert all(w.weight >= 0.0 for w in weights)

    def test_better_model_gets_more_weight(self):
        """Ordering by weight must match ordering by historical skill."""
        engine = DynamicWeightingEngine(skill_store=_seeded_store())
        weights = engine.compute_weights(
            list(MODEL_SKILL), variable=VARIABLE, lead_hours=LEAD_HOURS
        )
        by_model = {w.model_name: w.weight for w in weights}
        assert by_model["icon_eu"] > by_model["gfs"] > by_model["ecmwf"]

    def test_contributions_are_per_model(self):
        """Regression: a stale loop variable gave every model the same
        contributions, so the explanations did not match the weights."""
        engine = DynamicWeightingEngine(skill_store=_seeded_store())
        weights = engine.compute_weights(
            list(MODEL_SKILL), variable=VARIABLE, lead_hours=LEAD_HOURS
        )
        contributions = {w.historical_skill_contribution for w in weights}
        assert len(contributions) == len(weights)

    def test_contributions_follow_skill_order(self):
        engine = DynamicWeightingEngine(skill_store=_seeded_store())
        weights = {
            w.model_name: w.historical_skill_contribution
            for w in engine.compute_weights(
                list(MODEL_SKILL), variable=VARIABLE, lead_hours=LEAD_HOURS
            )
        }
        assert weights["icon_eu"] > weights["gfs"] > weights["ecmwf"]

    def test_equal_skill_gives_equal_weights(self):
        engine = DynamicWeightingEngine(skill_store=_seeded_store({"a": 2.0, "b": 2.0, "c": 2.0}))
        weights = engine.compute_weights(["a", "b", "c"], variable=VARIABLE, lead_hours=LEAD_HOURS)
        assert {round(w.weight, 6) for w in weights} == {round(1 / 3, 6)}

    def test_every_weight_has_a_rationale(self):
        engine = DynamicWeightingEngine(skill_store=_seeded_store())
        weights = engine.compute_weights(
            list(MODEL_SKILL), variable=VARIABLE, lead_hours=LEAD_HOURS
        )
        assert all(w.rationale for w in weights)

    def test_no_skill_data_yields_equal_weights(self):
        """With an empty store no model is favoured over another."""
        engine = DynamicWeightingEngine(skill_store=SkillProfileStore())
        weights = engine.compute_weights(
            list(MODEL_SKILL), variable=VARIABLE, lead_hours=LEAD_HOURS
        )
        assert sum(w.weight for w in weights) == pytest.approx(1.0)
        assert len({round(w.weight, 6) for w in weights}) == 1


# ── Blending ────────────────────────────────────────────────────────────────


class TestBlending:
    def test_blend_is_pulled_toward_the_skilful_model(self):
        """A skill-weighted blend must beat the naive mean of 14.0."""
        orch = _orchestrator()
        blended = orch.blend_forecasts(
            {
                "icon_eu": [_record("icon_eu", 10.0)],
                "gfs": [_record("gfs", 12.0)],
                "ecmwf": [_record("ecmwf", 20.0)],
            },
            variable=VARIABLE,
            lead_hours=LEAD_HOURS,
        )
        assert len(blended) == 1
        assert blended[0].value < 14.0

    def test_blend_within_contributor_range(self):
        orch = _orchestrator()
        blended = orch.blend_forecasts(
            {m: [_record(m, v)] for m, v in [("icon_eu", 10.0), ("gfs", 12.0), ("ecmwf", 20.0)]},
            variable=VARIABLE,
            lead_hours=LEAD_HOURS,
        )[0]
        assert 10.0 <= blended.value <= 20.0

    def test_identical_inputs_blend_to_same_value(self):
        orch = _orchestrator()
        blended = orch.blend_forecasts(
            {m: [_record(m, 7.5)] for m in MODEL_SKILL},
            variable=VARIABLE,
            lead_hours=LEAD_HOURS,
        )[0]
        assert blended.value == pytest.approx(7.5)

    def test_coverage_reflects_missing_models(self):
        """A model with a null value must not count as an active contributor."""
        orch = _orchestrator()
        blended = orch.blend_forecasts(
            {
                "icon_eu": [_record("icon_eu", 10.0)],
                "gfs": [_record("gfs", 12.0)],
                "ecmwf": [_record("ecmwf", None)],
            },
            variable=VARIABLE,
            lead_hours=LEAD_HOURS,
        )[0]
        assert blended.active_models == 2
        assert blended.total_models == 3
        assert blended.coverage_fraction == pytest.approx(2 / 3)

    def test_blend_records_weighting_rationale(self):
        orch = _orchestrator()
        blended = orch.blend_forecasts(
            {m: [_record(m, 10.0)] for m in MODEL_SKILL},
            variable=VARIABLE,
            lead_hours=LEAD_HOURS,
        )[0]
        assert blended.weighting_rationale


# ── Confidence and agreement ────────────────────────────────────────────────


class TestConfidenceAndAgreement:
    def test_confidence_is_a_fraction(self):
        orch = _orchestrator()
        blended = orch.blend_forecasts(
            {m: [_record(m, v)] for m, v in [("icon_eu", 10.0), ("gfs", 12.0), ("ecmwf", 20.0)]},
            variable=VARIABLE,
            lead_hours=LEAD_HOURS,
        )[0]
        confidence = orch.get_confidence(blended)
        assert 0.0 <= confidence.confidence_score <= 1.0
        assert confidence.rationale

    def test_agreement_beats_disagreement_in_confidence(self):
        """Tightly clustered models should score higher than scattered ones."""
        orch = _orchestrator()

        def confidence_for(values: dict[str, float]) -> float:
            blended = orch.blend_forecasts(
                {m: [_record(m, v)] for m, v in values.items()},
                variable=VARIABLE,
                lead_hours=LEAD_HOURS,
            )[0]
            return orch.get_confidence(blended).confidence_score

        agree = confidence_for({"icon_eu": 10.0, "gfs": 10.1, "ecmwf": 9.9})
        disagree = confidence_for({"icon_eu": 10.0, "gfs": 18.0, "ecmwf": 2.0})
        assert agree > disagree

    def test_agreement_uses_forecasts_only(self):
        """Spread and consensus are derived from forecast values alone."""
        orch = _orchestrator()
        result = orch.analyze_agreement(
            {m: [_record(m, v)] for m, v in [("icon_eu", 10.0), ("gfs", 10.2), ("ecmwf", 9.8)]},
            variable=VARIABLE,
        )
        agreement = result[0] if isinstance(result, list) else result
        assert agreement.total_models == 3
        assert agreement.inter_model_std == pytest.approx(0.2, abs=0.05)
        assert 0.0 <= agreement.consensus_score <= 1.0
        assert agreement.interpretation


# ── Meta-evaluation ─────────────────────────────────────────────────────────


def _synthetic_run(seed: int, sigma_a: float, sigma_b: float, weight_a: float = 0.6):
    """Two models with independent Gaussian error around a shared truth."""
    rng = random.Random(seed)
    truth = [10 + 3 * rng.gauss(0, 1) for _ in range(300)]
    a = [t + rng.gauss(0, sigma_a) for t in truth]
    b = [t + rng.gauss(0, sigma_b) for t in truth]
    blend = [weight_a * x + (1 - weight_a) * y for x, y in zip(a, b, strict=True)]
    pairs = lambda values: [  # noqa: E731
        {"value": v, "observed": t} for v, t in zip(values, truth, strict=True)
    ]
    return MetaEvaluator().evaluate(
        blended_forecasts=pairs(blend),
        single_model_forecasts={"icon_eu": pairs(a), "gfs": pairs(b)},
        observations=[{"observed": t} for t in truth],
        variable=VARIABLE,
        lead_hours=LEAD_HOURS,
        metric="mae",
    )


class TestMetaEvaluation:
    def test_blending_helps_with_independent_errors(self):
        """Averaging independent errors should beat either model alone."""
        result = _synthetic_run(seed=7, sigma_a=2.0, sigma_b=2.5)
        assert result.improvement_over_best is True
        assert result.blended_mae < result.best_single_mae
        assert result.mae_improvement_pct < 0  # negative = lower error

    def test_blending_can_hurt_and_says_so(self):
        """Diluting a strong model with a weak one must be reported honestly."""
        result = _synthetic_run(seed=11, sigma_a=0.5, sigma_b=6.0)
        assert result.improvement_over_best is False
        assert result.blended_mae > result.best_single_mae
        assert result.mae_improvement_pct > 0
        assert "worsened" in (result.rationale or "").lower()

    def test_best_single_model_is_identified(self):
        result = _synthetic_run(seed=7, sigma_a=2.0, sigma_b=2.5)
        assert result.best_single_model == "icon_eu"

    def test_sample_size_is_reported(self):
        result = _synthetic_run(seed=7, sigma_a=2.0, sigma_b=2.5)
        assert result.sample_size == 300
