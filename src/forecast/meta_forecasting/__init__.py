"""Adaptive Meta-Forecasting and Dynamic Model Blending.

This module provides an interpretable, reproducible, and leakage-free
adaptive forecast intelligence layer on top of the benchmarking platform.

Components:
- Skill profiles: historical model performance tracking
- Dynamic weighting: adaptive model weights by context
- Forecast blending: weighted blend with full provenance
- Confidence scoring: explainable uncertainty estimation
- Ensemble agreement: inter-model spread and consensus
- Regime selection: seasonal/regional model adaptation
- Meta-evaluation: measures whether blending actually helps

Design principles:
- NO deep learning, NO black boxes
- ALL decisions are explainable and auditable
- Strict causal integrity (no future leakage)
- Deterministic and reproducible

Usage:
    from ..meta_forecasting import MetaForecastingOrchestrator

    orchestrator = MetaForecastingOrchestrator()
    blended = orchestrator.blend_forecasts(forecasts_by_model, variable="temperature_2m")
    confidence = orchestrator.get_confidence(blended)
"""

# Models
from .agreement import EnsembleAgreementAnalyzer
from .blending import ForecastBlender
from .confidence import ConfidenceScorer
from .evaluation import MetaEvaluator
from .models import (
    BlendedForecastRecord,
    EnsembleAgreement,
    ForecastConfidence,
    MetaEvaluationResult,
    ModelWeight,
    RegimeAwareSelection,
    RegimeCondition,
    SkillProfile,
    SkillProfileEntry,
)

# Orchestrator (main entry point)
from .orchestrator import MetaForecastingOrchestrator
from .regime_selection import RegimeAwareModelSelector

# Core components
from .skill_profiles import SkillProfileStore
from .weighting import DynamicWeightingEngine

__all__ = [
    # Models
    "SkillProfile",
    "SkillProfileEntry",
    "ModelWeight",
    "BlendedForecastRecord",
    "ForecastConfidence",
    "RegimeCondition",
    "RegimeAwareSelection",
    "EnsembleAgreement",
    "MetaEvaluationResult",
    # Components
    "SkillProfileStore",
    "DynamicWeightingEngine",
    "ForecastBlender",
    "ConfidenceScorer",
    "EnsembleAgreementAnalyzer",
    "RegimeAwareModelSelector",
    "MetaEvaluator",
    # Orchestrator
    "MetaForecastingOrchestrator",
]
