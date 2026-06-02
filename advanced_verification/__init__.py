"""Advanced Verification & Forecast Skill Analysis.

Transforms forecast error analysis into operational meteorological verification
by evaluating event-based forecast skill using confusion matrices and derived
skill scores.

Package Contents
----------------
thresholds : Threshold comparison semantics and operators
events     : Binary event generation from forecasts and observations
confusion  : Confusion matrix computation (hits, misses, false_alarms, correct_negatives)
skill_scores : Derived skill metrics (CSI, hit rate, FAR, precision, etc.)
baselines  : Baseline forecast models (climatology, persistence)
validation : Structured validation of skill outputs
io         : Read verification data and write skill parquet outputs
cli        : CLI integration (events, skill, compare commands)

Design Principles
-----------------
1. Scientific correctness — explicit formulas, documented semantics
2. Explicit threshold behavior — inclusive/exclusive clearly specified
3. Reproducibility — deterministic computations with structured validation
4. Reusable architecture — composable verification functions
5. Future-proof — supports ensembles and probabilistic verification later

Usage
-----
>>> from advanced_verification.events import generate_events
>>> from advanced_verification.confusion import compute_confusion_matrix
>>> from advanced_verification.skill_scores import compute_skill_scores
>>> # ... or use the CLI
>>> # weather-analyzer events --event frost --group-by lead_hours
>>> # weather-analyzer skill --event frost --group-by lead_hours,model
"""

__version__ = "2.1.0"
