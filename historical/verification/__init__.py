"""Weather Forecast Verification Tool.

Verifies forecast data against observations by producing a canonical
verification dataset that preserves full forecast provenance.

Core invariant: (forecast_issue_time, forecast_target_time, model)
is the scientific identity of every forecast row and is never lost.
"""

__version__ = "0.2.0"
