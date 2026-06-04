"""API clients for Open-Meteo endpoints.

Provides thin wrappers around the Archive API (observations) and
Single Runs API (forecasts), with retry logic and error handling.
"""


import logging
import time
from datetime import datetime
from typing import Any

import httpx

from .constants import ARCHIVE_API, OPEN_METEO_API_KEY, SINGLE_RUNS_API

logger = logging.getLogger(__name__)


class ApiError(Exception):
    """Raised when the API returns an error response."""

    def __init__(self, message: str, status_code: int | None = None):
        super().__init__(message)
        self.status_code = status_code


def _build_params(
    lat: float,
    lon: float,
    variables: tuple[str, ...],
    timezone_str: str = "UTC",
    extra: dict[str, str] | None = None,
    api_key: str | None = None,
) -> dict[str, str]:
    """Build query parameters for Open-Meteo API requests."""
    params: dict[str, str] = {
        "latitude": str(lat),
        "longitude": str(lon),
        "hourly": ",".join(variables),
        "timezone": timezone_str,
        "timeformat": "iso8601",
    }
    if extra:
        params.update(extra)
    key = api_key or OPEN_METEO_API_KEY
    if key:
        params["apikey"] = key
    return params


def _handle_response(response: httpx.Response) -> dict[str, Any]:
    """Parse an API response, raising on errors."""
    if response.status_code != 200:
        try:
            body = response.json()
            reason = body.get("reason", response.text)
        except Exception:
            reason = response.text
        raise ApiError(
            f"API error (HTTP {response.status_code}): {reason}", status_code=response.status_code
        )

    try:
        return response.json()
    except Exception as e:
        raise ApiError(f"Invalid JSON response: {e}") from None


def _request_with_retry(
    client: httpx.Client,
    url: str,
    params: dict[str, str],
    max_retries: int = 3,
    retry_delay: float = 1.0,
) -> dict[str, Any]:
    """Make an HTTP request with exponential backoff retry."""
    for attempt in range(max_retries + 1):
        try:
            response = client.get(url, params=params)
            return _handle_response(response)
        except ApiError as exc:
            if exc.status_code == 429 and attempt < max_retries:
                wait = retry_delay * (2**attempt)
                logger.warning(
                    "Rate limited, waiting %.1fs before retry (attempt %d/%d)",
                    wait,
                    attempt + 1,
                    max_retries,
                )
                time.sleep(wait)
                continue
            raise
        except (httpx.ConnectError, httpx.ReadTimeout, httpx.NetworkError) as e:
            if attempt < max_retries:
                wait = retry_delay * (2**attempt)
                logger.warning(
                    "Network error (%s), waiting %.1fs before retry (attempt %d/%d)",
                    e,
                    wait,
                    attempt + 1,
                    max_retries,
                )
                time.sleep(wait)
                continue
            raise ApiError(f"Network error after {max_retries} retries: {e}") from None
        except Exception as e:
            raise ApiError(f"Unexpected error: {e}") from None

    # Should not reach here, but just in case
    raise ApiError("Max retries exceeded")


def _make_client(timeout: float) -> httpx.Client:
    """Create an httpx client that prefers IPv4.

    On many systems, DNS returns both IPv6 and IPv4 addresses (dual-stack),
    but the network has no IPv6 route. httpcore's connect_tcp raises on the
    first IPv6 failure instead of falling back to IPv4. We force IPv4 via
    local_address='0.0.0.0' to avoid this issue entirely.
    """
    return httpx.Client(
        timeout=timeout,
        http2=False,  # HTTP/2 can trigger additional connection issues
        transport=httpx.HTTPTransport(
            local_address="0.0.0.0",  # Force IPv4-only connections
        ),
    )


def fetch_observations(
    lat: float,
    lon: float,
    start_date: str,
    end_date: str,
    variables: tuple[str, ...],
    max_retries: int = 3,
    retry_delay: float = 1.0,
    api_key: str | None = None,
    url: str | None = None,
) -> dict[str, Any]:
    """Fetch historical observations from the Open-Meteo Archive API.

    Args:
        lat: Latitude.
        lon: Longitude.
        start_date: Start date (YYYY-MM-DD).
        end_date: End date (YYYY-MM-DD).
        variables: Tuple of hourly variable names.
        max_retries: Maximum number of retries.
        retry_delay: Base delay between retries (seconds).

    Returns:
        Parsed JSON response as a dict.

    Raises:
        ApiError: On API errors or invalid responses.
    """
    params = _build_params(
        lat,
        lon,
        variables,
        extra={
            "start_date": start_date,
            "end_date": end_date,
        },
        api_key=api_key,
    )

    with _make_client(30.0) as client:
        return _request_with_retry(client, url or ARCHIVE_API, params, max_retries, retry_delay)


def fetch_single_run(
    lat: float,
    lon: float,
    run_time: datetime,
    variables: tuple[str, ...],
    max_retries: int = 3,
    retry_delay: float = 1.0,
    api_key: str | None = None,
    url: str | None = None,
) -> dict[str, Any]:
    """Fetch a single forecast run from the Open-Meteo Single Runs API.

    Args:
        lat: Latitude.
        lon: Longitude.
        run_time: Model initialization time (UTC).
        variables: Tuple of hourly variable names.
        max_retries: Maximum number of retries.
        retry_delay: Base delay between retries (seconds).

    Returns:
        Parsed JSON response as a dict.

    Raises:
        ApiError: On API errors or invalid responses.
    """
    run_str = run_time.strftime("%Y-%m-%dT%H:%M")
    params = _build_params(
        lat,
        lon,
        variables,
        extra={
            "run": run_str,
        },
        api_key=api_key,
    )

    with _make_client(60.0) as client:
        return _request_with_retry(client, url or SINGLE_RUNS_API, params, max_retries, retry_delay)
