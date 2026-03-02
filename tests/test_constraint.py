# ruff: noqa: D100,D101,D102,D103,PLR2004
"""Tests for dap/constraint.py — constraint expression parser."""

import pytest

from xpublish_opendap.dap.constraint import (
    Constraint,
    HyperSlab,
    parse_dap2_constraint,
    parse_dap4_constraint,
)
from xpublish_opendap.errors import ConstraintNotSupportedError, ConstraintSyntaxError


class TestParseEmpty:
    def test_empty_string(self):
        c = parse_dap2_constraint("")
        assert c.projections == []
        assert c.selections == []

    def test_no_projection_no_selection(self):
        c = parse_dap2_constraint("")
        assert isinstance(c, Constraint)


class TestProjectionParsing:
    def test_single_variable(self):
        c = parse_dap2_constraint("air")
        assert len(c.projections) == 1
        assert c.projections[0].name == "air"
        assert c.projections[0].slices is None

    def test_multiple_variables(self):
        c = parse_dap2_constraint("air,lat,lon")
        assert len(c.projections) == 3
        assert c.projections[0].name == "air"
        assert c.projections[1].name == "lat"
        assert c.projections[2].name == "lon"

    def test_variable_with_single_index(self):
        c = parse_dap2_constraint("air[5]")
        assert len(c.projections) == 1
        item = c.projections[0]
        assert item.name == "air"
        assert item.slices == (HyperSlab(start=5, stop=5, stride=1),)

    def test_variable_with_range(self):
        c = parse_dap2_constraint("air[0:10]")
        item = c.projections[0]
        assert item.slices == (HyperSlab(start=0, stop=10, stride=1),)

    def test_variable_with_stride(self):
        c = parse_dap2_constraint("air[0:2:10]")
        item = c.projections[0]
        assert item.slices == (HyperSlab(start=0, stop=10, stride=2),)

    def test_multidimensional_slicing(self):
        c = parse_dap2_constraint("air[0:10][0:5][0:5]")
        item = c.projections[0]
        assert item.slices is not None
        assert len(item.slices) == 3
        assert item.slices[0] == HyperSlab(start=0, stop=10, stride=1)
        assert item.slices[1] == HyperSlab(start=0, stop=5, stride=1)
        assert item.slices[2] == HyperSlab(start=0, stop=5, stride=1)

    def test_mixed_projections(self):
        c = parse_dap2_constraint("air[0:10][0:5],lat")
        assert len(c.projections) == 2
        assert c.projections[0].name == "air"
        assert c.projections[0].slices is not None
        assert c.projections[1].name == "lat"
        assert c.projections[1].slices is None

    def test_percent_encoded(self):
        c = parse_dap2_constraint("air%5B0%3A10%5D")
        item = c.projections[0]
        assert item.name == "air"
        assert item.slices == (HyperSlab(start=0, stop=10, stride=1),)


class TestSelectionParsing:
    def test_selection_raises_not_supported(self):
        with pytest.raises(ConstraintNotSupportedError, match="not yet supported"):
            parse_dap2_constraint("air&time>100")


class TestErrorHandling:
    def test_invalid_hyperslab_non_numeric(self):
        with pytest.raises(ConstraintSyntaxError, match="Invalid hyperslab"):
            parse_dap2_constraint("air[abc]")

    def test_zero_stride_raises(self):
        with pytest.raises(ConstraintSyntaxError, match="Stride must be positive"):
            parse_dap2_constraint("air[0:0:10]")

    def test_function_call_raises(self):
        with pytest.raises(ConstraintNotSupportedError, match="Server-side functions"):
            parse_dap2_constraint("geogrid(air,10,20,30,40)")


class TestHyperSlab:
    def test_frozen(self):
        slab = HyperSlab(start=0, stop=10, stride=1)
        with pytest.raises(AttributeError):
            slab.start = 5  # type: ignore[misc]

    def test_equality(self):
        assert HyperSlab(0, 10, 1) == HyperSlab(0, 10, 1)
        assert HyperSlab(0, 10, 1) != HyperSlab(0, 10, 2)


class TestDAP4ConstraintParsing:
    def test_empty_string(self):
        c = parse_dap4_constraint("")
        assert c.projections == []
        assert c.selections == []

    def test_single_variable_with_slash(self):
        c = parse_dap4_constraint("/air")
        assert len(c.projections) == 1
        assert c.projections[0].name == "air"

    def test_single_variable_without_slash(self):
        c = parse_dap4_constraint("air")
        assert len(c.projections) == 1
        assert c.projections[0].name == "air"

    def test_semicolon_separated(self):
        c = parse_dap4_constraint("/air;/lat;/lon")
        assert len(c.projections) == 3
        assert c.projections[0].name == "air"
        assert c.projections[1].name == "lat"
        assert c.projections[2].name == "lon"

    def test_with_hyperslab(self):
        c = parse_dap4_constraint("/air[0:10]")
        assert len(c.projections) == 1
        item = c.projections[0]
        assert item.name == "air"
        assert item.slices == (HyperSlab(start=0, stop=10, stride=1),)

    def test_multidim_hyperslab(self):
        c = parse_dap4_constraint("/air[0:2:10][0:5]")
        item = c.projections[0]
        assert item.slices is not None
        assert len(item.slices) == 2
        assert item.slices[0] == HyperSlab(start=0, stop=10, stride=2)
        assert item.slices[1] == HyperSlab(start=0, stop=5, stride=1)

    def test_mixed_projections(self):
        c = parse_dap4_constraint("/air[0:10][0:5];/lat")
        assert len(c.projections) == 2
        assert c.projections[0].name == "air"
        assert c.projections[0].slices is not None
        assert c.projections[1].name == "lat"
        assert c.projections[1].slices is None

    def test_filter_raises_not_supported(self):
        with pytest.raises(ConstraintNotSupportedError, match="filter"):
            parse_dap4_constraint("/air|/air.temp>100")

    def test_percent_encoded(self):
        c = parse_dap4_constraint("%2Fair%5B0%3A10%5D")
        item = c.projections[0]
        assert item.name == "air"
        assert item.slices == (HyperSlab(start=0, stop=10, stride=1),)

    def test_whitespace_only(self):
        c = parse_dap4_constraint("   ")
        assert c.projections == []


class TestParserEdgeCases:
    def test_empty_variable_name_raises(self):
        with pytest.raises(ConstraintSyntaxError, match="Empty variable name"):
            parse_dap2_constraint("[0:10]")

    def test_four_part_hyperslab_raises(self):
        with pytest.raises(ConstraintSyntaxError, match="Invalid hyperslab"):
            parse_dap2_constraint("var[1:2:3:4]")

    def test_quoted_string_in_constraint(self):
        with pytest.raises(ConstraintNotSupportedError, match="not yet supported"):
            parse_dap2_constraint('var&field>"hello"')

    def test_space_between_hyperslabs(self):
        c = parse_dap2_constraint("var[0:1] [0:1]")
        item = c.projections[0]
        assert item.name == "var"
        assert item.slices is not None
        assert len(item.slices) == 2
        assert item.slices[0] == HyperSlab(start=0, stop=1, stride=1)
        assert item.slices[1] == HyperSlab(start=0, stop=1, stride=1)
