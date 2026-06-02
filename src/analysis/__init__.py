"""Analysis-first exploration layer for weather data.

Optimized for:
- Rapid validation
- Interactive notebook workflows
- Exploratory weather analysis
- Forecast verification research
- Fast visualization and metric experimentation

This is NOT production dashboards. It is a scientific workbench.

Usage
-----
>>> from src.analysis import loaders
>>> from src.analysis import alignment
>>> from src.analysis import metrics

>>> forecast = loaders.load_forecast("icon_eu", "temperature_2m", start="2024-01")
>>> truth = loaders.load_historical("temperature_2m", start="2024-01")
>>> aligned = alignment.align(forecast, truth)
>>> mae = metrics.mae_by_leadtime(aligned)

Design principles
-----------------
1. Notebooks stay thin — reusable logic lives in modules
2. Loaders return clean, standardized pandas DataFrames
3. All canonical columns are consistently named
4. No duplication of ingestion logic
5. Explicit over clever
"""

from __future__ import annotations

from src.analysis import (
    alignment,
    loaders,
    metrics,
    plotting,
    validation,
)

__all__ = ["alignment", "loaders", "metrics", "plotting", "validation"]
