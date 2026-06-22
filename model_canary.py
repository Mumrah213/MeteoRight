#!/usr/bin/env python3
"""Backend capability canary check.

Verifies that the configured Open-Meteo backend supports multi-model access.
When using the public Single Runs API, this will fail because all models
return identical data. When using a self-hosted or customer backend, this
should pass if the backend properly differentiates between models.

Usage:
    python model_canary.py [--base-url http://localhost:8080]
"""

import sys
from datetime import UTC, datetime

import httpx

from downloader.constants import SINGLE_RUNS_API

# Test parameters
LAT = 55.605
LON = 12.574
RUN_TIME = "2025-12-01T00:00"
VARIABLES = ("temperature_2m", "precipitation", "wind_speed_10m")

# Models to test — pick a diverse set
TEST_MODELS = ["ecmwf_ifs_hres", "gfs", "icon_eu", "hrrr"]

# Minimum number of distinct responses required to pass
MIN_DISTINCT_RESPONSES = 2


def fetch_model(
    url: str,
    model: str,
) -> tuple[int, dict | None, float]:
    """Fetch a single model run.

    Returns:
        (status_code, parsed_json_or_None, latency_ms)
    """
    # Use "models" (plural) for local/self-hosted backends,
    # "model" (singular) for public Single Runs API
    is_public = "single-runs-api.open-meteo.com" in url
    model_param = "model" if is_public else "models"
    params = {
        "latitude": str(LAT),
        "longitude": str(LON),
        "hourly": ",".join(VARIABLES),
        "timezone": "UTC",
        "run": RUN_TIME,
        model_param: model,
    }

    start = __import__("time").time()
    try:
        with httpx.Client(timeout=30.0, http2=False) as client:
            response = client.get(url, params=params)
        latency_ms = (__import__("time").time() - start) * 1000

        if response.status_code == 200:
            return response.status_code, response.json(), latency_ms
        else:
            return response.status_code, None, latency_ms
    except Exception:
        latency_ms = (__import__("time").time() - start) * 1000
        return 0, None, latency_ms


def responses_are_identical(data1: dict, data2: dict) -> bool:
    """Check if two API responses have identical forecast data.

    Compares only the hourly temperature_2m and precipitation arrays,
    ignoring metadata like generationtime_ms.
    """
    h1 = data1.get("hourly", {})
    h2 = data2.get("hourly", {})

    for var in ("temperature_2m", "precipitation", "wind_speed_10m"):
        v1 = h1.get(var, [])
        v2 = h2.get(var, [])
        if v1 != v2:
            return False

    return True


def run_canary(base_url: str | None = None) -> bool:
    """Run the canary check.

    Args:
        base_url: Override the configured backend URL.

    Returns:
        True if the canary passed (backend differentiates models).
    """
    url = base_url or SINGLE_RUNS_API
    is_public = "single-runs-api.open-meteo.com" in url

    print(f"Backend: {url}")
    print(f"Public API: {is_public}")
    print(f"Models to test: {TEST_MODELS}")
    print()

    if is_public:
        print("WARNING: Public Single Runs API detected.")
        print("  This API returns identical data for all models (ECMWF IFS only).")
        print("  Multi-model verification requires a self-hosted or customer backend.")
        print()

    results = []
    for model in TEST_MODELS:
        status, data, latency = fetch_model(url, model)
        results.append((model, status, data, latency))
        print(f"  {model:25s} HTTP {status}  {latency:.0f}ms")

    # Check 1: All models returned HTTP 200
    all_ok = all(s == 200 for _, s, _, _ in results)
    if not all_ok:
        print("\nFAIL: Not all models returned HTTP 200")
        return False

    # Check 2: Responses are not byte-identical across models
    distinct_count = 1
    for i in range(1, len(results)):
        if results[i][2] is None:
            continue
        if responses_are_identical(results[0][2], results[i][2]):
            print(f"  WARNING: {results[i][0]} returns identical data to {results[0][0]}")
        else:
            distinct_count += 1

    print(f"\nDistinct responses: {distinct_count}/{len(TEST_MODELS)}")

    if distinct_count < MIN_DISTINCT_RESPONSES:
        print("\nFAIL: Backend does not differentiate between models.")
        print(f"  Expected at least {MIN_DISTINCT_RESPONSES} distinct responses.")
        print("  This backend behaves like the public Single Runs API.")
        return False

    print("\nPASS: Backend differentiates between models.")
    return True


def main():
    base_url = None
    if "--base-url" in sys.argv:
        idx = sys.argv.index("--base-url")
        if idx + 1 < len(sys.argv):
            base_url = sys.argv[idx + 1]

    print(f"Canary check started at {datetime.now(UTC).strftime('%Y-%m-%d %H:%M UTC')}")
    print()

    # Capability summary from the shared runtime probe: which models the backend
    # actually serves and over what date range. Complements the differentiation
    # check below.
    if base_url:
        from agent_tools.backend import describe_backend

        forecast_url = base_url.rstrip("/")
        if not forecast_url.endswith("/forecast"):
            forecast_url = f"{forecast_url}/v1/forecast"
        desc = describe_backend(forecast_url)
        print(f"Capability probe ({forecast_url}): reachable={desc['reachable']}")
        for m in desc["models"]:
            cov = m["coverage"]
            window = f"{cov['start']}..{cov['end']}" if cov else "unknown"
            print(f"  {m['backend_name']:30s} data {window}")
        print()

    passed = run_canary(base_url)

    print()
    if passed:
        print("Result: PASS — backend supports multi-model access")
        return 0
    else:
        print("Result: FAIL — backend does not support multi-model access")
        print()
        print("To enable multi-model access:")
        print("  export METEORIGHT_OPEN_METEO_BASE_URL=http://your-openmeteo-host:8080")
        return 1


if __name__ == "__main__":
    sys.exit(main())
