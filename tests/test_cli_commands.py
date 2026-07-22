"""Smoke tests for every CLI subcommand module.

These exist because splitting cli.py into a package surfaced three modules
whose imports were incomplete (`pd` undefined in verify/advanced, sibling
handlers undefined in pipeline) — the kind of break nothing else caught,
since no test imported or ran those code paths.

Each test drives a command through `main()` on the bundled Copenhagen
sample, so an import error or a broken handler fails loudly.
"""

from __future__ import annotations

import ast
import importlib
from pathlib import Path

import pandas as pd
import pytest

from cli import main

REPO_ROOT = Path(__file__).resolve().parent.parent
VERIFICATION = REPO_ROOT / "examples/copenhagen_sample/verification/verification.parquet"

COMMAND_MODULES = [
    "advanced",
    "analyze",
    "blend",
    "download",
    "grid",
    "main",
    "metrics",
    "parser",
    "pipeline",
    "verify",
]


@pytest.mark.parametrize("module", COMMAND_MODULES)
def test_command_module_imports(module: str):
    """Every command module must import cleanly on its own."""
    importlib.import_module(f"meteoright.cli.{module}")


@pytest.mark.parametrize("module", COMMAND_MODULES)
def test_command_module_has_no_unbound_globals(module: str):
    """Every global a command module references must actually resolve.

    Catches the split-induced failure mode where a handler body still calls
    `pd` or a sibling `cmd_*` that no longer exists in the new module's
    namespace — invisible until that command runs.
    """
    import builtins

    mod = importlib.import_module(f"meteoright.cli.{module}")
    source = Path(mod.__file__).read_text()
    tree = ast.parse(source)

    assigned = {
        node.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store)
    }
    unresolved = set()
    for func in [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]:
        local = {a.arg for a in func.args.args}
        local |= {
            n.id for n in ast.walk(func) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)
        }
        for node in ast.walk(func):
            # names bound by `from x import y` / `import x` inside the function
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                local |= {(a.asname or a.name).split(".")[0] for a in node.names}
            # lambda parameters and comprehension targets
            elif isinstance(node, ast.Lambda):
                local |= {a.arg for a in node.args.args}
            elif isinstance(node, ast.comprehension):
                local |= {n.id for n in ast.walk(node.target) if isinstance(n, ast.Name)}
            # `except ... as e`
            elif isinstance(node, ast.ExceptHandler) and node.name:
                local.add(node.name)
        for node in ast.walk(func):
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
                name = node.id
                if (
                    name not in local
                    and name not in assigned
                    and not hasattr(mod, name)
                    and not hasattr(builtins, name)
                ):
                    unresolved.add(name)

    assert not unresolved, f"{module}.py references unbound names: {sorted(unresolved)}"


def test_help_exits_zero(capsys):
    assert main([]) == 0
    assert "MeteoRight" in capsys.readouterr().out


@pytest.mark.parametrize(
    "argv",
    [
        ["download", "--help"],
        ["verify", "--help"],
        ["metrics", "--help"],
        ["analyze", "--help"],
        ["pipeline", "--help"],
        ["blend", "--help"],
        ["grid-points", "--help"],
        ["advanced", "--help"],
    ],
)
def test_subcommand_help(argv: list[str]):
    """--help must reach every subparser without importing a broken module."""
    with pytest.raises(SystemExit) as excinfo:
        main(argv)
    assert excinfo.value.code == 0


def test_grid_points_runs(capsys):
    assert main(["grid-points", "--lat", "55.605", "--lon", "12.574", "--summary"]) == 0
    assert "Grid Points Summary" in capsys.readouterr().out


def test_metrics_then_analyze(tmp_path: Path):
    """metrics and analyze share a data contract — run them in sequence."""
    metrics_dir = tmp_path / "metrics"
    assert (
        main(
            [
                "metrics",
                "--verification",
                str(VERIFICATION),
                "--group-by",
                "lead_hours,model",
                "--variables",
                "temperature_2m",
                "--output",
                str(metrics_dir),
            ]
        )
        == 0
    )
    written = list(metrics_dir.glob("*.parquet"))
    assert written, "metrics wrote no parquet"

    analysis_dir = tmp_path / "analysis"
    assert (
        main(
            [
                "analyze",
                "--metrics",
                str(metrics_dir / "*.parquet"),
                "--verification",
                str(VERIFICATION),
                "--variables",
                "temperature_2m",
                "--plots",
                "lead_time,distributions",
                "--generate-tables",
                "--output",
                str(analysis_dir),
            ]
        )
        == 0
    )
    assert list(analysis_dir.rglob("*.png")), "analyze produced no figures"


def test_advanced_skill_runs(tmp_path: Path):
    assert (
        main(
            [
                "advanced",
                "skill",
                "--verification",
                str(VERIFICATION),
                "--event",
                "frost",
                "--output",
                str(tmp_path / "adv"),
            ]
        )
        == 0
    )
    assert list((tmp_path / "adv").glob("*.parquet"))


def test_advanced_without_subcommand_returns_error():
    assert main(["advanced"]) == 1


def test_verify_builds_verification_rows(tmp_path: Path):
    """`verify` needs raw forecast/observation partitions, so build them."""
    data_dir = tmp_path / "data"
    times = pd.date_range("2024-01-01", periods=24, freq="h", tz="UTC")

    observations = pd.DataFrame(
        {
            "observation_time": times,
            "latitude": 55.6,
            "longitude": 12.5,
            "elevation": 10.0,
            "temperature_2m": range(24),
        }
    )
    obs_path = data_dir / "observations" / "year=2024" / "month=01"
    obs_path.mkdir(parents=True)
    observations.to_parquet(obs_path / "data.parquet")

    forecasts = observations.rename(columns={"observation_time": "forecast_target_time"}).assign(
        forecast_issue_time=times[0],
        lead_hours=range(24),
        model="icon_eu",
    )
    fc_path = data_dir / "forecasts" / "model=icon_eu" / "year=2024" / "month=01"
    fc_path.mkdir(parents=True)
    forecasts.to_parquet(fc_path / "data.parquet")

    assert (
        main(
            [
                "verify",
                "--data-dir",
                str(data_dir),
                "--variables",
                "temperature_2m",
                "--output",
                str(tmp_path / "verification"),
            ]
        )
        == 0
    )
    assert list((tmp_path / "verification").glob("*.parquet"))
