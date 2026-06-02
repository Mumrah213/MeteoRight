"""Tests for data models — essential scientific tests only."""

from datetime import UTC, datetime

from downloader.models import ForecastRecord


def test_lead_hours_computed_from_timestamps():
    """Lead hours is computed from forecast_issue_time and forecast_target_time."""
    record = ForecastRecord(
        forecast_issue_time=datetime(2024, 3, 1, 0, 0, tzinfo=UTC),
        forecast_target_time=datetime(2024, 3, 1, 6, 0, tzinfo=UTC),
        model="test",
        latitude=55.0,
        longitude=13.0,
        elevation=0.0,
        variables={},
    )
    assert record.lead_hours == 6
