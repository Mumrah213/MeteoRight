"""Time utilities for weather data processing."""


from datetime import UTC, datetime


def utc_now() -> str:
    """Return current UTC time as ISO format string."""
    return datetime.now(UTC).isoformat()


def parse_iso_timestamp(ts: str | float | int) -> datetime:
    """Parse an ISO timestamp into a timezone-aware datetime.

    Handles:
    - ISO strings: "2024-01-15T00:00:00"
    - ISO with timezone: "2024-01-15T00:00:00+00:00"
    - Unix timestamps (float/int): 1705276800
    """
    if isinstance(ts, (int, float)):
        return datetime.fromtimestamp(ts, tz=UTC)

    # Assume ISO string
    try:
        return datetime.fromisoformat(ts)
    except ValueError as e:
        raise ValueError(f"Cannot parse timestamp: {ts!r}") from e


def ensure_utc(dt: datetime) -> datetime:
    """Ensure a datetime is timezone-aware in UTC."""
    if dt.tzinfo is None:
        return dt.replace(tzinfo=UTC)
    return dt.astimezone(UTC)
