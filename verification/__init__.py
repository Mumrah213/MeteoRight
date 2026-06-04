"""Weather Forecast Verification Tool.

Verifies forecast data against observations by producing a canonical
verification dataset that preserves full forecast provenance.

Core invariant: (forecast_issue_time, forecast_target_time, model)
is the scientific identity of every forecast row and is never lost.

Event-based verification (skill scores, confusion matrices):
    from verification.confusion import compute_confusion_matrix
    from verification.events import generate_events
    from verification.skill_scores import compute_skill_scores
"""

__version__ = "0.2.0"

# Event-based verification exports
from .baselines import compute_climatology_event_probability, compute_climatology_skill
from .confusion import compute_confusion_matrix, compute_confusion_counts
from .events import generate_events
from .event_io import load_verification, write_confusion, write_skill_scores
from .event_schema import (
    ALL_SKILL_METRICS,
    BUILTIN_EVENT_DEFS,
    BUILTIN_EVENTS,
    forecast_event_col,
    observed_event_col,
)
from .event_validation import validate_confusion, validate_skill_scores
from .skill_scores import compute_skill_scores
from .thresholds import compare_threshold, apply_threshold_series
