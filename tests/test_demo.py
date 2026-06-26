"""Smoke test for `meteoright demo` — the offline bundled-sample path.

Guarantees that a fresh install can render plots and tables from the committed
sample dataset with no network and no backend. If this breaks, the "clone and
see it work" promise is broken.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless; no display required


def test_demo_renders_offline(tmp_path: Path):
    from cli import _SAMPLE_VERIFICATION, main

    # The bundled sample must ship with the package.
    assert _SAMPLE_VERIFICATION.exists(), f"missing bundled sample: {_SAMPLE_VERIFICATION}"

    output = tmp_path / "demo_out"
    rc = main(["demo", "--output", str(output)])
    assert rc == 0

    # Lead-time degradation plot (the headline figure).
    lead_time_plot = output / "figures" / "lead_time" / "mae_by_lead_time_temperature_2m.png"
    assert lead_time_plot.exists(), "lead-time MAE plot was not produced"

    # An error-distribution figure.
    dist_dir = output / "figures" / "distributions"
    assert any(dist_dir.glob("error_histogram_temperature_2m_*.png")), "no distribution plots"

    # Metrics and tables.
    assert any((output / "metrics").glob("*.parquet")), "no metrics parquet written"
    assert (output / "tables" / "temperature_2m_error_quantiles.csv").exists()
