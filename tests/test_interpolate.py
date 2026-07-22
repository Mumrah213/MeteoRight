"""Tests for forecast-to-observation interpolation.

The properties that matter: interpolation is exact at grid nodes, it never
leaves the range of its inputs, it handles wind direction as a circular
quantity, and it degrades to IDW rather than extrapolating off the grid edge.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from verification.interpolate import (
    cluster_coordinates,
    haversine_km,
    infer_grid_spacing,
    interpolate,
    interpolate_bilinear,
    interpolate_idw,
    interpolate_nearest,
    snap_to_grid,
    snap_to_lines,
)

# Latitudes as Open-Meteo actually returns them for one 0.25 degree column:
# four grid lines, each requested several times, each answer jittered.
JITTERED_LATS = np.array(
    [
        55.2485,
        55.2496,
        55.2516,
        55.2538,
        55.2542,
        55.4985,
        55.4992,
        55.5016,
        55.5034,
        55.5037,
        55.5041,
        55.7485,
        55.7492,
        55.7516,
        55.7534,
        55.7538,
        55.7541,
        55.9958,
        55.9987,
        55.9994,
        56.0018,
        56.0036,
        56.0039,
    ]
)

# A 2x2 cell of the IFS 0.25 degree grid near Copenhagen.
CELL_LATS = np.array([55.50, 55.50, 55.75, 55.75])
CELL_LONS = np.array([12.50, 12.75, 12.50, 12.75])
CELL_VALUES = np.array([10.0, 20.0, 30.0, 40.0])

METHODS = ["nearest", "bilinear", "idw"]


class TestHaversine:
    def test_zero_distance(self):
        assert haversine_km(55.6, 12.5, 55.6, 12.5) == pytest.approx(0.0)

    def test_known_separation(self):
        """0.25 degrees of latitude is ~27.8 km anywhere."""
        assert haversine_km(55.50, 12.5, 55.75, 12.5) == pytest.approx(27.8, abs=0.3)

    def test_symmetric(self):
        forward = haversine_km(55.5, 12.5, 56.0, 13.0)
        backward = haversine_km(56.0, 13.0, 55.5, 12.5)
        assert forward == pytest.approx(backward)


class TestGridInference:
    def test_infers_regular_spacing(self):
        assert infer_grid_spacing(np.array([55.25, 55.5, 55.75, 56.0])) == pytest.approx(0.25)

    def test_tolerates_api_jitter(self):
        """Open-Meteo returns cell coordinates with sub-kilometre jitter."""
        jittered = np.array([55.2485, 55.4985, 55.7516, 56.0039])
        assert infer_grid_spacing(jittered) == pytest.approx(0.25, abs=0.01)

    def test_handles_gaps_in_the_grid(self):
        """A missing row still implies the same underlying spacing."""
        assert infer_grid_spacing(np.array([55.25, 55.5, 56.0])) == pytest.approx(0.25)

    def test_single_coordinate_has_no_spacing(self):
        assert infer_grid_spacing(np.array([55.5])) is None

    def test_snap_removes_jitter(self):
        snapped = snap_to_grid(np.array([55.2485, 55.4985, 55.7516]), 0.25)
        assert snapped == pytest.approx([55.25, 55.50, 55.75])


class TestJitteredGrid:
    """Regression: Open-Meteo returns each grid cell with a little jitter.

    Taking the minimum raw gap read that jitter as the spacing (0.2416 instead
    of 0.25) and snapping then scattered one grid line across several
    coordinates, so the interpolation cube was built with the wrong corners.
    """

    def test_clusters_collapse_to_grid_lines(self):
        centres = cluster_coordinates(JITTERED_LATS)
        assert centres.size == 4
        assert centres == pytest.approx([55.25, 55.50, 55.75, 56.00], abs=0.01)

    def test_spacing_survives_jitter(self):
        """The failure mode: 0.2416 rather than 0.25."""
        assert infer_grid_spacing(JITTERED_LATS) == pytest.approx(0.25, abs=0.005)

    def test_snap_to_lines_is_idempotent(self):
        lines = cluster_coordinates(JITTERED_LATS)
        once = snap_to_lines(JITTERED_LATS, lines)
        twice = snap_to_lines(once, lines)
        assert once == pytest.approx(twice)

    def test_snap_to_lines_yields_one_value_per_line(self):
        lines = cluster_coordinates(JITTERED_LATS)
        snapped = snap_to_lines(JITTERED_LATS, lines)
        assert np.unique(snapped).size == 4

    def test_jittered_corners_still_interpolate(self):
        """A cell whose corners carry jitter must still interpolate cleanly."""
        lats = np.array([55.4985, 55.5016, 55.7492, 55.7534])
        lons = np.array([12.5008, 12.7551, 12.5022, 12.7558])
        values = np.array([10.0, 20.0, 30.0, 40.0])

        result = interpolate_bilinear(lats, lons, values, 55.625, 12.628)
        assert 10.0 <= result <= 40.0
        assert result == pytest.approx(25.0, abs=2.0)


@pytest.mark.parametrize("method", METHODS)
class TestExactnessAndBounds:
    def test_exact_at_grid_nodes(self, method: str):
        """Interpolating at a grid point must return that point's value."""
        for lat, lon, expected in zip(CELL_LATS, CELL_LONS, CELL_VALUES):
            result = interpolate(CELL_LATS, CELL_LONS, CELL_VALUES, lat, lon, method=method)
            assert result == pytest.approx(expected)

    def test_stays_within_input_range(self, method: str):
        """No method may invent a value outside its inputs."""
        result = interpolate(CELL_LATS, CELL_LONS, CELL_VALUES, 55.61, 12.63, method=method)
        assert CELL_VALUES.min() <= result <= CELL_VALUES.max()

    def test_uniform_field_is_preserved(self, method: str):
        """A constant field interpolates to that constant."""
        constant = np.full(4, 7.5)
        result = interpolate(CELL_LATS, CELL_LONS, constant, 55.6, 12.6, method=method)
        assert result == pytest.approx(7.5)

    def test_ignores_nan_grid_values(self, method: str):
        values = CELL_VALUES.copy()
        values[0] = np.nan
        result = interpolate(CELL_LATS, CELL_LONS, values, 55.74, 12.74, method=method)
        assert math.isfinite(result)

    def test_all_nan_returns_nan(self, method: str):
        result = interpolate(CELL_LATS, CELL_LONS, np.full(4, np.nan), 55.6, 12.6, method=method)
        assert math.isnan(result)


class TestBilinear:
    def test_cell_centre_is_the_mean(self):
        """Equidistant from all four corners: the plain average."""
        result = interpolate_bilinear(CELL_LATS, CELL_LONS, CELL_VALUES, 55.625, 12.625)
        assert result == pytest.approx(25.0)

    def test_edge_midpoint_averages_two_corners(self):
        """On the southern edge, only the two southern corners contribute."""
        result = interpolate_bilinear(CELL_LATS, CELL_LONS, CELL_VALUES, 55.50, 12.625)
        assert result == pytest.approx(15.0)

    def test_weights_follow_position(self):
        """Nearer the low corner means nearer its value."""
        near_low = interpolate_bilinear(CELL_LATS, CELL_LONS, CELL_VALUES, 55.53, 12.53)
        near_high = interpolate_bilinear(CELL_LATS, CELL_LONS, CELL_VALUES, 55.72, 12.72)
        assert near_low < 25.0 < near_high

    def test_falls_back_when_cell_incomplete(self):
        """Three corners cannot define a bilinear cell — IDW instead."""
        lats, lons, values = CELL_LATS[:3], CELL_LONS[:3], CELL_VALUES[:3]
        result = interpolate_bilinear(lats, lons, values, 55.6, 12.6)
        expected = interpolate_idw(lats, lons, values, 55.6, 12.6)
        assert result == pytest.approx(expected)

    def test_outside_the_grid_falls_back(self):
        """Never extrapolate a weather field past its own grid."""
        result = interpolate_bilinear(CELL_LATS, CELL_LONS, CELL_VALUES, 54.0, 11.0)
        expected = interpolate_idw(CELL_LATS, CELL_LONS, CELL_VALUES, 54.0, 11.0)
        assert result == pytest.approx(expected)


class TestIDW:
    def test_nearer_points_dominate(self):
        result = interpolate_idw(CELL_LATS, CELL_LONS, CELL_VALUES, 55.51, 12.51)
        assert result == pytest.approx(10.0, abs=3.0)

    def test_higher_power_sharpens(self):
        """A larger exponent concentrates weight on the closest point."""
        soft = interpolate_idw(CELL_LATS, CELL_LONS, CELL_VALUES, 55.55, 12.55, power=1.0)
        sharp = interpolate_idw(CELL_LATS, CELL_LONS, CELL_VALUES, 55.55, 12.55, power=6.0)
        assert abs(sharp - 10.0) < abs(soft - 10.0)

    def test_k_limits_the_neighbourhood(self):
        """k=1 is the nearest-neighbour answer."""
        idw = interpolate_idw(CELL_LATS, CELL_LONS, CELL_VALUES, 55.55, 12.55, k=1)
        nearest = interpolate_nearest(CELL_LATS, CELL_LONS, CELL_VALUES, 55.55, 12.55)
        assert idw == pytest.approx(nearest)


class TestCircularVariables:
    def test_wrap_around_north(self):
        """350 and 10 degrees average to 0, not 180."""
        lats = np.array([55.5, 55.5, 55.75, 55.75])
        lons = np.array([12.5, 12.75, 12.5, 12.75])
        directions = np.array([350.0, 10.0, 350.0, 10.0])

        result = interpolate_bilinear(lats, lons, directions, 55.625, 12.625, circular=True)
        assert min(result, 360.0 - result) < 1.0

    def test_linear_mode_would_be_wrong(self):
        """Contrast: without circular=True the same input averages to 180."""
        lats = np.array([55.5, 55.5, 55.75, 55.75])
        lons = np.array([12.5, 12.75, 12.5, 12.75])
        directions = np.array([350.0, 10.0, 350.0, 10.0])

        result = interpolate_bilinear(lats, lons, directions, 55.625, 12.625, circular=False)
        assert result == pytest.approx(180.0)

    def test_result_stays_in_range(self):
        lats = np.array([55.5, 55.5, 55.75, 55.75])
        lons = np.array([12.5, 12.75, 12.5, 12.75])
        directions = np.array([10.0, 80.0, 190.0, 300.0])

        result = interpolate_bilinear(lats, lons, directions, 55.6, 12.6, circular=True)
        assert 0.0 <= result < 360.0

    def test_idw_handles_wrap(self):
        lats = np.array([55.5, 55.5, 55.75, 55.75])
        lons = np.array([12.5, 12.75, 12.5, 12.75])
        directions = np.array([355.0, 5.0, 355.0, 5.0])

        result = interpolate_idw(lats, lons, directions, 55.625, 12.625, circular=True)
        assert min(result, 360.0 - result) < 2.0


class TestInputValidation:
    def test_mismatched_lengths_raise(self):
        with pytest.raises(ValueError, match="same length"):
            interpolate(np.array([55.5, 55.75]), np.array([12.5]), np.array([1.0]), 55.6, 12.6)

    def test_empty_grid_raises(self):
        with pytest.raises(ValueError, match="no grid points"):
            interpolate(np.array([]), np.array([]), np.array([]), 55.6, 12.6)

    def test_unknown_method_raises(self):
        with pytest.raises(ValueError, match="unknown interpolation method"):
            interpolate(CELL_LATS, CELL_LONS, CELL_VALUES, 55.6, 12.6, method="quadratic")
