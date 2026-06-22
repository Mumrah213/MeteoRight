"""Unit tests for the backend capability probe (no network).

Uses an httpx MockTransport injected via monkeypatching ``_make_client`` so the
probe logic is exercised against synthetic backend responses.
"""

from datetime import UTC, datetime, timedelta

import httpx
import pytest

from agent_tools import backend


def _make_mock_client(handler):
    """Build an httpx.Client backed by a MockTransport using ``handler``."""
    return httpx.Client(transport=httpx.MockTransport(handler))


@pytest.fixture
def patch_client(monkeypatch):
    """Patch backend._make_client to return a mock-transport client."""

    def _install(handler):
        monkeypatch.setattr(backend, "_make_client", lambda timeout: _make_mock_client(handler))

    return _install


def _hourly(n):
    return {"hourly": {"temperature_2m": [1.0] * n}}


def test_discovers_only_models_with_data(patch_client):
    today = datetime.now(UTC).date()
    have = {"ecmwf_ifs025", "dwd_icon"}

    from datetime import date

    lo = today - timedelta(days=25)
    hi = today - timedelta(days=15)

    def handler(request: httpx.Request) -> httpx.Response:
        model = request.url.params.get("models")
        day = date.fromisoformat(request.url.params.get("start_date"))
        # Give models in `have` data for a contiguous window ~15-25 days ago.
        if model in have and lo <= day <= hi:
            return httpx.Response(200, json=_hourly(24))
        return httpx.Response(200, json=_hourly(0))

    patch_client(handler)
    r = backend.describe_backend(
        candidate_models=("ecmwf_ifs025", "dwd_icon", "nonexistent_model"),
        discover_coverage=False,
    )
    assert r["error"] is None
    assert r["reachable"] is True
    names = {m["backend_name"] for m in r["models"]}
    assert names == have


def test_coverage_window_is_contiguous(patch_client):
    today = datetime.now(UTC).date()
    lo = today - timedelta(days=30)
    hi = today - timedelta(days=10)

    def handler(request: httpx.Request) -> httpx.Response:
        from datetime import date

        day = date.fromisoformat(request.url.params.get("start_date"))
        has = lo <= day <= hi
        return httpx.Response(200, json=_hourly(24 if has else 0))

    patch_client(handler)
    r = backend.describe_backend(
        candidate_models=("ecmwf_ifs025",), discover_coverage=True
    )
    assert r["error"] is None
    cov = r["models"][0]["coverage"]
    assert cov == {"start": lo.isoformat(), "end": hi.isoformat()}


def test_unreachable_backend(patch_client):
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused")

    patch_client(handler)
    r = backend.describe_backend(
        candidate_models=("ecmwf_ifs025",), discover_coverage=False
    )
    assert r["reachable"] is False
    assert r["models"] == []
    assert r["error"]["type"] == "BackendUnreachable"


def test_reachable_but_no_data(patch_client):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_hourly(0))

    patch_client(handler)
    r = backend.describe_backend(
        candidate_models=("ecmwf_ifs025",), discover_coverage=False
    )
    assert r["models"] == []
    assert r["error"]["type"] == "NoModelsWithData"
