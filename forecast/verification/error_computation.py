"""Phase 3: Error Computation Layer.

Computes error metrics from aligned forecast-observation pairs:
  - error = forecast_value - observation_value
  - absolute_error = |error|
  - squared_error = error²
  - bias = mean(error)

Supports grouping by lead_time, variable, location, run_time.

All computations handle null values explicitly — unmatched or
null values are excluded from error calculations but tracked
in sample counts.

Circular variables (e.g., wind direction in degrees) use the
shortest-angle convention: the minimum signed difference on
the unit circle, wrapped to [-180, +180)."""


from .alignment import AlignmentRecord

# Variables treated as circular (0-360° phase angle)
CIRCULAR_VARIABLES = frozenset({"wind_direction_10m"})


def _circular_error(forecast: float, observation: float) -> float:
    """Compute shortest-angle error for a 0-360° circular variable.

    Returns a value in [-180, +180).  Positive means forecast
    is clockwise of observation; negative means counter-clockwise.
    """
    diff = forecast - observation
    # Wrap to [-180, +180)
    diff = ((diff + 180.0) % 360.0) - 180.0
    return diff


class ErrorComputationError(Exception):
    """Raised when error computation fails."""


class ErrorComputation:
    """Computes error metrics from aligned records.

    Core operation:
      error = forecast_value - observation_value

    For circular variables (e.g. wind_direction_10m) the error
    is computed as the shortest angle on the unit circle.

    All values must be non-null for error computation.
    Null or unmatched values are excluded but tracked.
    """

    @staticmethod
    def compute_errors(alignment: AlignmentRecord) -> dict[str, float | None]:
        """Compute all error metrics for one alignment record.

        Args:
            alignment: AlignmentRecord with forecast and observation values

        Returns:
            Dict with keys: 'error', 'absolute_error', 'squared_error', 'bias_contribution'
            Values are None if either forecast or observation is null.
        """
        if not alignment.is_fully_matched:
            return {
                "error": None,
                "absolute_error": None,
                "squared_error": None,
                "bias_contribution": None,
            }

        forecast_val = alignment.forecast_value
        obs_val = alignment.observation_value

        # Use circular error for phase-angle variables
        if alignment.variable in CIRCULAR_VARIABLES:
            error = _circular_error(forecast_val, obs_val)
        else:
            error = forecast_val - obs_val

        return {
            "error": error,
            "absolute_error": abs(error),
            "squared_error": error**2,
            "bias_contribution": error,
        }

    @staticmethod
    def compute_errors_batch(
        alignments: list[AlignmentRecord],
    ) -> list[dict[str, float | None]]:
        """Compute errors for a batch of alignment records.

        Args:
            alignments: List of AlignmentRecord objects

        Returns:
            List of error dicts (one per alignment)
        """
        return [ErrorComputation.compute_errors(a) for a in alignments]

    @staticmethod
    def compute_bias(
        alignments: list[AlignmentRecord],
    ) -> float | None:
        """Compute bias (mean error) from aligned records.

        Only considers fully matched records with non-null values.
        For circular variables, computes mean of circular errors
        (not recomputed here; the aligned table already has the
        per-record error column).

        Args:
            alignments: List of AlignmentRecord objects

        Returns:
            Bias value (mean forecast - observation), or None if no data
        """
        matched = [a for a in alignments if a.is_fully_matched]

        if not matched:
            return None

        errors = []
        for a in matched:
            if a.variable in CIRCULAR_VARIABLES:
                error = _circular_error(a.forecast_value, a.observation_value)
            else:
                error = a.forecast_value - a.observation_value
            errors.append(error)

        return sum(errors) / len(errors)

    @staticmethod
    def compute_absolute_errors(
        alignments: list[AlignmentRecord],
    ) -> list[float]:
        """Compute absolute errors from aligned records.

        Args:
            alignments: List of AlignmentRecord objects

        Returns:
            List of absolute error values (only for fully matched records)
        """
        errors: list[float] = []
        for a in alignments:
            if a.is_fully_matched:
                if a.variable in CIRCULAR_VARIABLES:
                    error = _circular_error(a.forecast_value, a.observation_value)
                else:
                    error = a.forecast_value - a.observation_value
                errors.append(abs(error))
        return errors

    @staticmethod
    def compute_squared_errors(
        alignments: list[AlignmentRecord],
    ) -> list[float]:
        """Compute squared errors from aligned records.

        Args:
            alignments: List of AlignmentRecord objects

        Returns:
            List of squared error values (only for fully matched records)
        """
        errors: list[float] = []
        for a in alignments:
            if a.is_fully_matched:
                if a.variable in CIRCULAR_VARIABLES:
                    error = _circular_error(a.forecast_value, a.observation_value)
                else:
                    error = a.forecast_value - a.observation_value
                errors.append(error**2)
        return errors

    @staticmethod
    def compute_error_statistics(
        alignments: list[AlignmentRecord],
    ) -> dict[str, float | None]:
        """Compute comprehensive error statistics.

        Args:
            alignments: List of AlignmentRecord objects

        Returns:
            Dict with keys: 'mean_error', 'mae', 'rmse', 'bias', 'error_variance'
        """
        matched = [a for a in alignments if a.is_fully_matched]

        if not matched:
            return {
                "mean_error": None,
                "mae": None,
                "rmse": None,
                "bias": None,
                "error_variance": None,
                "sample_size": 0,
            }

        errors = []
        abs_errors = []
        squared_errors = []

        for a in matched:
            if a.variable in CIRCULAR_VARIABLES:
                error = _circular_error(a.forecast_value, a.observation_value)
            else:
                error = a.forecast_value - a.observation_value
            errors.append(error)
            abs_errors.append(abs(error))
            squared_errors.append(error**2)

        n = len(errors)

        mean_error = sum(errors) / n
        mae = sum(abs_errors) / n
        rmse = (sum(squared_errors) / n) ** 0.5
        bias = mean_error
        variance = sum((e - mean_error) ** 2 for e in errors) / n

        return {
            "mean_error": mean_error,
            "mae": mae,
            "rmse": rmse,
            "bias": bias,
            "error_variance": variance,
            "sample_size": n,
        }
