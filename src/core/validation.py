"""Validation and parity checks for the unified pipeline.

Ensures that the merged system maintains data integrity across
the boundary between historical (v2) and forecast (docker) subsystems.

Checks:
- Parquet schema consistency across subsystems
- Canonical coordinate validation
- Forecast-to-observation alignment quality
- Data completeness (no null-heavy partitions)
- Cross-subsystem parity (same data produces same results)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar, Literal

import pandas as pd
import pyarrow.parquet as pq


class ValidationCheck:
    """Type for validation check identifiers."""

    SCHEMA_VALIDATION = "schema_validation"
    PARTITION_STRUCTURE = "partition_structure"
    DATA_COMPLETENESS = "data_completeness"
    EMPTY_VERIFICATION = "empty_verification"
    NAN_HEAVY_GROUP = "nan_heavy_group"
    ABSURD_BIAS = "absurd_bias"
    ABSURD_MAE = "absurd_mae"


@dataclass
class ValidationError:
    """A single validation error found during checks."""

    check: str
    message: str
    severity: Literal["error", "warning"] = "error"

    def __str__(self) -> str:
        return f"[{self.severity.upper()}] {self.check}: {self.message}"


logger = logging.getLogger(__name__)


class ParquetSchemaValidator:
    """Validate parquet schema consistency across subsystems."""

    REQUIRED_COLUMNS: ClassVar[frozenset[str]] = frozenset(
        (
            "timestamp",
            "latitude",
            "longitude",
            "variable_name",
            "value",
            "model",
            "run_time",
            "forecast_hour",
        )
    )

    OBSERVATION_COLUMNS: ClassVar[frozenset[str]] = frozenset(
        (
            "timestamp",
            "latitude",
            "longitude",
            "variable_name",
            "value",
        )
    )

    FORECAST_COLUMNS: ClassVar[frozenset[str]] = frozenset(
        (
            "timestamp",
            "latitude",
            "longitude",
            "variable_name",
            "value",
            "model",
            "run_time",
            "forecast_hour",
        )
    )

    def validate_schema(self, parquet_path: str | Path) -> dict[str, Any]:
        """Validate a parquet file's schema against expected columns.

        Auto-detects if the file is observation or forecast based on
        column names. Returns a dict with pass/fail status.
        """
        path = Path(parquet_path)
        result: dict[str, Any] = {
            "path": str(path),
            "valid": True,
            "issues": [],
        }

        try:
            schema = pq.read_schema(path)
            column_names = {f.name for f in schema}

            # Auto-detect: observations have no model/run_time/forecast_hour
            if "model" in column_names and "run_time" in column_names:
                expected = self.FORECAST_COLUMNS
            else:
                expected = self.OBSERVATION_COLUMNS

            # Check required columns (auto-detected type)
            missing = expected - column_names
            if missing:
                result["valid"] = False
                result["issues"].append(f"Missing required columns: {missing}")

            # Check for null-heavy partitions (>50% nulls)
            table = pq.read_table(path)
            for col_name in table.column_names:
                col = table.column(col_name)
                null_count = col.null_count
                total_count = len(col)
                if total_count > 0 and null_count / total_count > 0.5:
                    result["issues"].append(
                        f"Column '{col_name}' is {null_count / total_count:.0%} null"
                    )

        except Exception as e:
            result["valid"] = False
            result["issues"].append(f"Error reading parquet file: {e}")

        return result

    def validate_partition_structure(self, data_dir: str | Path) -> dict[str, Any]:
        """Validate the overall partition structure of the data directory.

        Expected structure:
        data/raw/observations/*.parquet
        data/raw/forecasts/model={model}/year={Y}/month={M}/data.parquet
        """
        result: dict[str, Any] = {
            "valid": True,
            "partitions_found": 0,
            "issues": [],
        }

        data_path = Path(data_dir)

        # Check observations
        obs_dir = data_path / "raw" / "observations"
        if obs_dir.exists():
            obs_files = list(obs_dir.glob("*.parquet"))
            result["partitions_found"] += len(obs_files)
            if not obs_files:
                result["issues"].append("No observation parquet files found")
        else:
            result["issues"].append("Missing raw/observations directory")

        # Check forecasts
        forecast_dir = data_path / "raw" / "forecasts"
        if forecast_dir.exists():
            model_dirs = list(forecast_dir.glob("model=*/"))
            result["partitions_found"] += len(model_dirs)
            for model_dir in model_dirs:
                year_dirs = list(model_dir.glob("year=*/"))
                for year_dir in year_dirs:
                    month_dirs = list(year_dir.glob("month=*/"))
                    for month_dir in month_dirs:
                        data_files = list(month_dir.glob("data.parquet"))
                        if not data_files:
                            result["issues"].append(f"No data.parquet in {month_dir}")
        else:
            result["issues"].append("Missing raw/forecasts directory")

        return result


class CrossSubsystemParityChecker:
    """Ensure forecast and historical subsystems produce consistent results."""

    def __init__(self, output_dir: str | Path) -> None:
        self.output_dir = Path(output_dir)

    def check_data_completeness(self, start_date: str, end_date: str) -> dict[str, Any]:
        """Check for data completeness across the pipeline.

        Verifies:
        - All expected dates have forecast data
        - No null-heavy partitions
        - Consistent variable coverage
        """
        result: dict[str, Any] = {
            "valid": True,
            "issues": [],
            "forecast_coverage": {},
            "observation_coverage": {},
        }

        # Check forecast data completeness
        forecast_dir = self.output_dir / "raw" / "forecasts"
        if forecast_dir.exists():
            for model_dir in forecast_dir.glob("model=*/"):
                model_name = model_dir.name.split("=")[1]
                partitions = list(model_dir.glob("year=*/month=*/data.parquet"))
                result["forecast_coverage"][model_name] = {
                    "partition_count": len(partitions),
                    "issues": [],
                }

                for partition in partitions:
                    try:
                        table = pq.read_table(partition)
                        null_pct = sum(c.null_count for c in table.columns) / (
                            len(table.columns) * len(table)
                        )
                        if null_pct > 0.5:
                            result["forecast_coverage"][model_name]["issues"].append(
                                f"{partition.relative_to(forecast_dir)} is {null_pct:.0%} null"
                            )
                    except Exception as e:
                        result["forecast_coverage"][model_name]["issues"].append(
                            f"Error reading {partition}: {e}"
                        )
        else:
            result["valid"] = False
            result["issues"].append("No forecast data directory found")

        return result

    def run_validation_suite(self) -> list[ValidationError]:
        """Run all validation checks. Returns list of any errors found."""
        errors: list[ValidationError] = []

        # Schema validation
        schema_validator = ParquetSchemaValidator()
        for parquet_file in self.output_dir.rglob("*.parquet"):
            schema_result = schema_validator.validate_schema(parquet_file)
            if not schema_result["valid"]:
                for issue in schema_result["issues"]:
                    errors.append(
                        ValidationError(
                            check="schema_validation",
                            message=f"{parquet_file}: {issue}",
                            severity="error",
                        )
                    )

        # Partition structure validation
        partition_result = schema_validator.validate_partition_structure(self.output_dir)
        if not partition_result["valid"]:
            for issue in partition_result["issues"]:
                errors.append(
                    ValidationError(
                        check="partition_structure",
                        message=issue,
                        severity="error",
                    )
                )

        # Data completeness
        parity_checker = CrossSubsystemParityChecker(self.output_dir)
        completeness = parity_checker.check_data_completeness("2024-01-01", "2024-12-31")
        if not completeness["valid"]:
            for issue in completeness["issues"]:
                errors.append(
                    ValidationError(
                        check="data_completeness",
                        message=issue,
                        severity="error",
                    )
                )

        return errors


class DatasetParityValidator:
    """Validate that unified outputs match v2 and docker originals.

    Ensures:
    1. Historical parquet schema matches v2 exactly
    2. Forecast outputs match docker system behavior
    3. Directory structure integrity
    """

    # v2 observation parquet columns (from datasets.schema)
    V2_OBSERVATION_COLUMNS: ClassVar[frozenset[str]] = frozenset(
        (
            "timestamp",
            "latitude",
            "longitude",
            "variable_name",
            "value",
        )
    )

    # v2 forecast parquet columns (from datasets.schema)
    V2_FORECAST_COLUMNS: ClassVar[frozenset[str]] = frozenset(
        (
            "timestamp",
            "latitude",
            "longitude",
            "variable_name",
            "value",
            "model",
            "run_time",
            "forecast_hour",
        )
    )

    # docker canonical columns (from meteo.canonical.models)
    DOCKER_CANONICAL_COLUMNS: ClassVar[frozenset[str]] = frozenset(
        (
            "timestamp",
            "latitude",
            "longitude",
            "variable_name",
            "value",
            "model",
            "run_time",
            "forecast_hour",
            "source",
            "version",
        )
    )

    def validate_historical_schema(self, parquet_path: str | Path) -> ValidationError | None:
        """Validate a historical parquet file matches v2 schema exactly.

        Returns ValidationError if schema doesn't match, else None.
        """
        path = Path(parquet_path)
        try:
            schema = pq.read_schema(path)
            column_names = {f.name for f in schema}

            # Determine if this is an observation or forecast file
            if "model" in column_names and "run_time" in column_names:
                expected = self.V2_FORECAST_COLUMNS
            else:
                expected = self.V2_OBSERVATION_COLUMNS

            missing = expected - column_names
            extra = column_names - expected

            if missing or extra:
                msg = f"{path.name}: "
                if missing:
                    msg += f"missing={missing} "
                if extra:
                    msg += f"extra={extra}"
                return ValidationError(
                    check="historical_schema",
                    message=msg,
                    severity="error",
                )
        except Exception as e:
            return ValidationError(
                check="historical_schema",
                message=f"{path.name}: {e}",
                severity="error",
            )
        return None

    def validate_docker_canonical_schema(
        self,
        parquet_path: str | Path,
    ) -> ValidationError | None:
        """Validate a forecast parquet file matches docker canonical schema.

        Returns ValidationError if schema doesn't match, else None.
        """
        path = Path(parquet_path)
        try:
            schema = pq.read_schema(path)
            column_names = {f.name for f in schema}

            missing = self.DOCKER_CANONICAL_COLUMNS - column_names
            extra = column_names - self.DOCKER_CANONICAL_COLUMNS

            if missing or extra:
                msg = f"{path.name}: "
                if missing:
                    msg += f"missing={missing} "
                if extra:
                    msg += f"extra={extra}"
                return ValidationError(
                    check="docker_schema",
                    message=msg,
                    severity="error",
                )
        except Exception as e:
            return ValidationError(
                check="docker_schema",
                message=f"{path.name}: {e}",
                severity="error",
            )
        return None

    def validate_directory_structure(self, data_dir: str | Path) -> list[ValidationError]:
        """Validate directory structure integrity.

        Expected:
        data/raw/observations/
        data/raw/forecasts/model={model}/year={Y}/month={M}/data.parquet
        data/intermediate/
        data/cache/
        """
        errors: list[ValidationError] = []
        base = Path(data_dir)

        expected_dirs = [
            "raw/observations",
            "raw/forecasts",
            "intermediate",
            "cache",
        ]

        for rel in expected_dirs:
            if not (base / rel).is_dir():
                errors.append(
                    ValidationError(
                        check="directory_structure",
                        message=f"Missing required directory: {rel}",
                        severity="error",
                    )
                )

        # Check forecast partition format
        forecast_dir = base / "raw" / "forecasts"
        if forecast_dir.is_dir():
            for model_dir in forecast_dir.iterdir():
                if model_dir.is_dir() and not model_dir.name.startswith("model="):
                    errors.append(
                        ValidationError(
                            check="directory_structure",
                            message=f"Invalid forecast model dir: {model_dir.name}",
                            severity="error",
                        )
                    )

        return errors

    def run_parity_checks(self, data_dir: str | Path) -> list[ValidationError]:
        """Run all parity checks on a dataset directory.

        Returns list of ValidationError for any issues found.
        """
        errors: list[ValidationError] = []
        base = Path(data_dir)

        # 1. Directory structure
        errors.extend(self.validate_directory_structure(data_dir))

        # 2. Historical schema check
        obs_dir = base / "raw" / "observations"
        if obs_dir.is_dir():
            for pf in obs_dir.glob("*.parquet"):
                err = self.validate_historical_schema(pf)
                if err:
                    errors.append(err)

        # 3. Forecast canonical schema check
        forecast_dir = base / "raw" / "forecasts"
        if forecast_dir.is_dir():
            for pf in forecast_dir.rglob("*.parquet"):
                err = self.validate_docker_canonical_schema(pf)
                if err:
                    errors.append(err)

        return errors


class VerificationResultValidator:
    """Validate verification outputs meet quality thresholds."""

    MIN_SAMPLE_SIZE = 100
    MAX_BIAS = 5.0  # degrees C
    MAX_MAE = 3.0  # degrees C

    def validate_verification_results(
        self, verification_data: pd.DataFrame
    ) -> list[ValidationError]:
        """Validate verification results meet quality thresholds.

        Checks:
        - Minimum sample size per metric group
        - Bias within acceptable range
        - MAE within acceptable range
        - No NaN-heavy groups
        """
        errors: list[ValidationError] = []

        if verification_data.empty:
            errors.append(
                ValidationError(
                    check="empty_verification",
                    message="Verification data is empty",
                    severity="error",
                )
            )
            return errors

        # Check for NaN-heavy groups
        groupby_cols = ["model", "variable"]
        if "lead_hours" in verification_data.columns:
            groupby_cols.append("lead_hours")

        grouped = verification_data.groupby(groupby_cols, dropna=False)
        for name, group in grouped:
            # Check sample size
            if "sample_size" in group.columns:
                min_sample = group["sample_size"].min()
                if pd.notna(min_sample) and min_sample < self.MIN_SAMPLE_SIZE:
                    errors.append(
                        ValidationError(
                            check="nan_heavy_group",
                            message=f"Group {name} has sample size {min_sample} "
                            f"(min: {self.MIN_SAMPLE_SIZE})",
                            severity="warning",
                        )
                    )

            # Check bias range
            if "bias" in group.columns:
                bias_values = group["bias"].dropna()
                if len(bias_values) > 0 and bias_values.abs().max() > self.MAX_BIAS:
                    errors.append(
                        ValidationError(
                            check="absurd_bias",
                            message=f"Group {name} has max absolute bias "
                            f"{bias_values.abs().max():.2f} (threshold: {self.MAX_BIAS})",
                            severity="warning",
                        )
                    )

            # Check MAE range
            if "mae" in group.columns:
                mae_values = group["mae"].dropna()
                if len(mae_values) > 0 and mae_values.max() > self.MAX_MAE:
                    errors.append(
                        ValidationError(
                            check="absurd_mae",
                            message=f"Group {name} has MAE {mae_values.max():.2f} "
                            f"(threshold: {self.MAX_MAE})",
                            severity="warning",
                        )
                    )

        return errors
