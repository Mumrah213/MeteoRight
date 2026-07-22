"""End-to-end tests for the `meteoright blend` command.

Exercises the full path: read verification parquet, learn per-model skill on a
training split, weight, blend the held-out rows, and report the comparison.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from cli import main


def _verification_parquet(
    path: Path,
    model_sigmas: dict[str, float],
    n: int = 400,
    seed: int = 42,
    observation_jitter: float = 0.0,
) -> Path:
    """Write verification rows for several models over shared targets.

    `observation_jitter` perturbs the observed value per model, reproducing
    real verification data where each model resolves the observation slightly
    differently (nearest station or interpolation choice).
    """
    rng = np.random.default_rng(seed)
    target = pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC")
    truth = 10 + 8 * np.sin(np.arange(n) / 24 * 2 * np.pi) + rng.normal(0, 1.5, n)

    frames = [
        pd.DataFrame(
            {
                "forecast_issue_time": target - pd.Timedelta("24h"),
                "forecast_target_time": target,
                "observation_time": target,
                "lead_hours": 24,
                "model": model,
                "latitude": 55.6,
                "longitude": 12.5,
                "forecast_temperature_2m": truth + rng.normal(0, sigma, n),
                "observed_temperature_2m": truth
                + (rng.normal(0, observation_jitter, n) if observation_jitter else 0.0),
            }
        )
        for model, sigma in model_sigmas.items()
    ]
    pd.concat(frames, ignore_index=True).to_parquet(path)
    return path


@pytest.fixture
def verification_file(tmp_path: Path) -> Path:
    return _verification_parquet(
        tmp_path / "verification.parquet",
        {"icon_eu": 1.8, "gfs": 2.6, "ecmwf_ifs": 2.1},
    )


def test_blend_runs_successfully(verification_file: Path):
    assert main(["blend", "--verification", str(verification_file)]) == 0


def test_blend_writes_report(verification_file: Path, tmp_path: Path):
    out = tmp_path / "report.json"
    main(["blend", "--verification", str(verification_file), "--output", str(out)])

    report = json.loads(out.read_text())
    assert report["variable"] == "temperature_2m"
    assert sorted(report["models"]) == ["ecmwf_ifs", "gfs", "icon_eu"]
    assert report["sample_size"] > 0


def test_weights_sum_to_one(verification_file: Path, tmp_path: Path):
    out = tmp_path / "report.json"
    main(["blend", "--verification", str(verification_file), "--output", str(out)])

    weights = json.loads(out.read_text())["weights"]
    assert sum(weights.values()) == pytest.approx(1.0)


def test_blending_helps_on_independent_errors(verification_file: Path, tmp_path: Path):
    """Independent per-model error is exactly the case blending should win."""
    out = tmp_path / "report.json"
    main(["blend", "--verification", str(verification_file), "--output", str(out)])

    report = json.loads(out.read_text())
    assert report["improvement_over_best"] is True
    assert report["blended_mae"] < report["best_single_mae"]
    assert report["mae_improvement_pct"] < 0  # negative = lower error


def test_dominant_model_is_not_diluted(tmp_path: Path):
    """A far weaker model must not drag a strong one down."""
    path = _verification_parquet(
        tmp_path / "skewed.parquet",
        {"good_model": 0.3, "bad_model": 7.0},
        seed=3,
    )
    out = tmp_path / "report.json"
    main(["blend", "--verification", str(path), "--output", str(out)])

    report = json.loads(out.read_text())
    assert report["weights"]["bad_model"] == pytest.approx(0.0, abs=1e-9)
    assert report["blended_mae"] <= report["best_single_mae"] * 1.01


def test_held_out_rows_are_not_used_for_training(verification_file: Path, tmp_path: Path):
    """Train and test must partition the shared targets without overlap."""
    out = tmp_path / "report.json"
    main(
        [
            "blend",
            "--verification",
            str(verification_file),
            "--train-fraction",
            "0.5",
            "--output",
            str(out),
        ]
    )

    report = json.loads(out.read_text())
    assert report["train_rows"] + report["sample_size"] == 400


def test_per_model_observations_still_align(tmp_path: Path):
    """Regression: real verification rows carry a slightly different observed
    value per model. Indexing the pivot on that column put every model on its
    own row, so no target was ever shared and blending aborted."""
    path = _verification_parquet(
        tmp_path / "jittered.parquet",
        {"icon_eu": 1.8, "gfs": 2.6, "ecmwf_ifs": 2.1},
        observation_jitter=0.4,
    )
    out = tmp_path / "report.json"
    assert main(["blend", "--verification", str(path), "--output", str(out)]) == 0

    report = json.loads(out.read_text())
    assert report["sample_size"] > 0
    assert sum(report["weights"].values()) == pytest.approx(1.0)


def test_single_model_is_rejected(tmp_path: Path):
    """Blending one model is meaningless and must fail loudly."""
    path = _verification_parquet(tmp_path / "one.parquet", {"icon_eu": 1.5})
    assert main(["blend", "--verification", str(path)]) == 1


def test_unknown_variable_is_rejected(verification_file: Path):
    assert (
        main(
            [
                "blend",
                "--verification",
                str(verification_file),
                "--variable",
                "not_a_variable",
            ]
        )
        == 1
    )
