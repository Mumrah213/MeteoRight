"""HTTP transport layer — agnostic provider-transport boundary.

All HTTP interactions go through this layer. Provider adapters never
call `requests.get` or `httpx` directly.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import httpx

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class HTTPResponse:
    """Normalized HTTP response envelope.

    All provider adapters receive this type — never raw httpx responses.
    """

    status_code: int
    content: bytes
    headers: dict[str, str]
    url: str
    elapsed_ms: float

    @property
    def ok(self) -> bool:
        return 200 <= self.status_code < 300

    @property
    def json(self) -> Any:
        import json

        return json.loads(self.content)

    @property
    def text(self) -> str:
        return self.content.decode("utf-8", errors="replace")


class HTTPClient:
    """HTTP transport with timeout, retry, and structured logging.

    This is the ONLY place in the codebase that talks to the network.
    """

    def __init__(
        self,
        base_url: str | None = None,
        timeout: float = 60.0,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        user_agent: str = "meteo/0.1.0 (verification-platform)",
        api_key: str | None = None,
    ) -> None:
        resolved_base_url = (
            base_url or os.getenv("METEORIGHT_OPEN_METEO_BASE_URL") or "https://api.open-meteo.com"
        )
        self.base_url = resolved_base_url.rstrip("/")
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.user_agent = user_agent
        self.api_key = api_key or os.getenv("METEORIGHT_OPEN_METEO_API_KEY")

    def _build_url(self, path: str, params: dict[str, Any]) -> str:
        """Build a full URL with query parameters."""
        from urllib.parse import urlencode

        # Normalize lists to comma-separated strings (Open-Meteo convention)
        # Filter out None values — they cause API errors (e.g., daily=None → HTTP 400)
        normalized: dict[str, Any] = {}
        if self.api_key and "apikey" not in params:
            normalized["apikey"] = self.api_key
        for k, v in params.items():
            if v is None:
                continue
            if isinstance(v, (list, tuple)):
                normalized[k] = ",".join(str(x) for x in v)
            else:
                normalized[k] = v

        query = urlencode(normalized)
        return f"{self.base_url}{path}?{query}"

    def get(self, path: str, params: dict[str, Any] | None = None) -> HTTPResponse:
        """Perform a GET request with retries and structured logging."""
        url = self._build_url(path, params or {})
        logger.info("HTTP GET %s", url)

        last_exc: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            try:
                start = datetime.now(UTC)
                resp = httpx.get(url, timeout=self.timeout, follow_redirects=True)
                elapsed = (datetime.now(UTC) - start).total_seconds() * 1000

                headers = dict(resp.headers)
                response = HTTPResponse(
                    status_code=resp.status_code,
                    content=resp.content,
                    headers=headers,
                    url=url,
                    elapsed_ms=elapsed,
                )

                logger.info(
                    "HTTP %d %s [%.0fms] body=%d bytes",
                    resp.status_code,
                    path,
                    elapsed,
                    len(resp.content),
                )
                return response

            except (httpx.TimeoutException, httpx.ConnectError) as exc:
                last_exc = exc
                if attempt < self.max_retries:
                    logger.warning(
                        "Retry %d/%d for %s: %s",
                        attempt,
                        self.max_retries,
                        path,
                        exc,
                    )
                else:
                    logger.error(
                        "Max retries (%d) exhausted for %s: %s",
                        self.max_retries,
                        path,
                        exc,
                    )

        raise RuntimeError(f"Failed after {self.max_retries} retries: {last_exc}") from last_exc  # type: ignore[misc]
