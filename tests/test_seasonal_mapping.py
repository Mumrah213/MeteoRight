"""Tests for seasonal mapping correctness.

Verifies:
- Month-to-season mapping is correct
- Season ordering is correct
- Edge cases are handled
"""

from plots.seasonal import MONTH_ORDER, MONTH_SEASON_NAMES, SEASON_MAP, SEASON_ORDER


class TestSeasonMapping:
    """Test the season mapping dictionary."""

    def test_djf_months(self):
        """December, January, February should map to DJF."""
        assert SEASON_MAP[12] == "DJF"
        assert SEASON_MAP[1] == "DJF"
        assert SEASON_MAP[2] == "DJF"

    def test_mam_months(self):
        """March, April, May should map to MAM."""
        assert SEASON_MAP[3] == "MAM"
        assert SEASON_MAP[4] == "MAM"
        assert SEASON_MAP[5] == "MAM"

    def test_jja_months(self):
        """June, July, August should map to JJA."""
        assert SEASON_MAP[6] == "JJA"
        assert SEASON_MAP[7] == "JJA"
        assert SEASON_MAP[8] == "JJA"

    def test_son_months(self):
        """September, October, November should map to SON."""
        assert SEASON_MAP[9] == "SON"
        assert SEASON_MAP[10] == "SON"
        assert SEASON_MAP[11] == "SON"

    def test_all_months_mapped(self):
        """All 12 months should be mapped."""
        assert set(SEASON_MAP.keys()) == set(range(1, 13))

    def test_no_unmapped_months(self):
        """No month should map to None or empty string."""
        for _month, season in SEASON_MAP.items():
            assert season is not None
            assert season != ""

    def test_season_order(self):
        """Season order should be DJF, MAM, JJA, SON."""
        assert SEASON_ORDER == ["DJF", "MAM", "JJA", "SON"]

    def test_month_names(self):
        """Month name abbreviations should be correct."""
        assert MONTH_SEASON_NAMES[1] == "Jan"
        assert MONTH_SEASON_NAMES[6] == "Jun"
        assert MONTH_SEASON_NAMES[12] == "Dec"

    def test_month_order(self):
        """Month order should be 1-12."""
        assert list(range(1, 13)) == MONTH_ORDER

    def test_season_map_consistency(self):
        """Each month should map to exactly one season."""
        for month in range(1, 13):
            assert SEASON_MAP[month] in SEASON_ORDER
