# ruff: noqa: D100,D101,D102,D103,PLR2004
"""Tests for dap/xarray_adapter.py — constraint to xarray subsetting."""

import numpy as np
import pytest
import xarray as xr

from xpublish_opendap.dap.constraint import Constraint, HyperSlab, ProjectionItem
from xpublish_opendap.dap.xarray_adapter import apply_plan, plan_subsetting
from xpublish_opendap.errors import IndexOutOfRangeError, VariableNotFoundError


@pytest.fixture
def sample_ds():
    """Small dataset for subsetting tests."""
    return xr.Dataset(
        {
            "temp": xr.DataArray(
                np.random.rand(10, 5),
                dims=["time", "x"],
            ),
            "pressure": xr.DataArray(
                np.random.rand(10, 5),
                dims=["time", "x"],
            ),
        },
        coords={
            "time": np.arange(10),
            "x": np.arange(5),
        },
    )


class TestPlanSubsetting:
    def test_empty_constraint_selects_all(self, sample_ds):
        constraint = Constraint()
        plan = plan_subsetting(sample_ds, constraint)
        assert plan.variables is None
        assert plan.isel_args == {}
        assert plan.estimated_bytes > 0

    def test_single_variable_projection(self, sample_ds):
        constraint = Constraint(projections=[ProjectionItem(name="temp")])
        plan = plan_subsetting(sample_ds, constraint)
        assert "temp" in plan.variables
        # Coordinate variables should be auto-included
        assert "time" in plan.variables
        assert "x" in plan.variables

    def test_hyperslab_creates_isel(self, sample_ds):
        constraint = Constraint(
            projections=[
                ProjectionItem(
                    name="temp",
                    slices=(
                        HyperSlab(start=0, stop=4, stride=1),
                        HyperSlab(start=0, stop=2, stride=1),
                    ),
                ),
            ],
        )
        plan = plan_subsetting(sample_ds, constraint)
        # DAP inclusive stop → Python exclusive stop
        assert plan.isel_args["time"] == slice(0, 5, 1)
        assert plan.isel_args["x"] == slice(0, 3, 1)

    def test_stride_preserved(self, sample_ds):
        constraint = Constraint(
            projections=[
                ProjectionItem(
                    name="temp",
                    slices=(
                        HyperSlab(start=0, stop=8, stride=2),
                        HyperSlab(start=0, stop=4, stride=1),
                    ),
                ),
            ],
        )
        plan = plan_subsetting(sample_ds, constraint)
        assert plan.isel_args["time"] == slice(0, 9, 2)

    def test_variable_not_found(self, sample_ds):
        constraint = Constraint(projections=[ProjectionItem(name="nonexistent")])
        with pytest.raises(VariableNotFoundError):
            plan_subsetting(sample_ds, constraint)

    def test_wrong_number_of_slices(self, sample_ds):
        constraint = Constraint(
            projections=[
                ProjectionItem(
                    name="temp",
                    slices=(HyperSlab(start=0, stop=4, stride=1),),
                ),
            ],
        )
        with pytest.raises(IndexOutOfRangeError, match="2 dimensions"):
            plan_subsetting(sample_ds, constraint)

    def test_index_out_of_range(self, sample_ds):
        constraint = Constraint(
            projections=[
                ProjectionItem(
                    name="temp",
                    slices=(
                        HyperSlab(start=0, stop=100, stride=1),
                        HyperSlab(start=0, stop=4, stride=1),
                    ),
                ),
            ],
        )
        with pytest.raises(IndexOutOfRangeError, match="exceeds"):
            plan_subsetting(sample_ds, constraint)

    def test_negative_index_raises(self, sample_ds):
        constraint = Constraint(
            projections=[
                ProjectionItem(
                    name="temp",
                    slices=(
                        HyperSlab(start=-1, stop=4, stride=1),
                        HyperSlab(start=0, stop=4, stride=1),
                    ),
                ),
            ],
        )
        with pytest.raises(IndexOutOfRangeError, match="Negative"):
            plan_subsetting(sample_ds, constraint)

    def test_memory_estimation_positive(self, sample_ds):
        constraint = Constraint()
        plan = plan_subsetting(sample_ds, constraint)
        assert plan.estimated_bytes > 0

    def test_grid_array_member_path(self, sample_ds):
        """Grid-qualified path temp.temp resolves to temp."""
        constraint = Constraint(projections=[ProjectionItem(name="temp.temp")])
        plan = plan_subsetting(sample_ds, constraint)
        assert "temp" in plan.variables

    def test_grid_array_member_with_slices(self, sample_ds):
        """Grid-qualified path with hyperslabs resolves and applies slices."""
        constraint = Constraint(
            projections=[
                ProjectionItem(
                    name="temp.temp",
                    slices=(
                        HyperSlab(start=0, stop=4, stride=1),
                        HyperSlab(start=0, stop=2, stride=1),
                    ),
                ),
            ],
        )
        plan = plan_subsetting(sample_ds, constraint)
        assert "temp" in plan.variables
        assert plan.isel_args["time"] == slice(0, 5, 1)
        assert plan.isel_args["x"] == slice(0, 3, 1)

    def test_grid_map_member_path(self, sample_ds):
        """Grid map member path temp.time resolves to coordinate time."""
        constraint = Constraint(projections=[ProjectionItem(name="temp.time")])
        plan = plan_subsetting(sample_ds, constraint)
        assert "time" in plan.variables

    def test_invalid_grid_member_raises(self, sample_ds):
        """Grid member path temp.nonexistent raises VariableNotFoundError."""
        constraint = Constraint(projections=[ProjectionItem(name="temp.nonexistent")])
        with pytest.raises(VariableNotFoundError):
            plan_subsetting(sample_ds, constraint)

    def test_dotted_name_not_grid_raises(self, sample_ds):
        """Dotted path x.x where x is a 1-d coord (not a Grid) raises error."""
        constraint = Constraint(projections=[ProjectionItem(name="x.x")])
        with pytest.raises(VariableNotFoundError):
            plan_subsetting(sample_ds, constraint)


class TestApplyPlan:
    def test_no_subsetting(self, sample_ds):
        from xpublish_opendap.dap.xarray_adapter import SubsettingPlan

        plan = SubsettingPlan()
        result = apply_plan(sample_ds, plan)
        assert set(result.data_vars) == set(sample_ds.data_vars)
        assert result.sizes == sample_ds.sizes

    def test_variable_selection(self, sample_ds):
        from xpublish_opendap.dap.xarray_adapter import SubsettingPlan

        plan = SubsettingPlan(variables=["temp", "time", "x"])
        result = apply_plan(sample_ds, plan)
        assert "temp" in result
        assert "pressure" not in result

    def test_dimension_slicing(self, sample_ds):
        from xpublish_opendap.dap.xarray_adapter import SubsettingPlan

        plan = SubsettingPlan(isel_args={"time": slice(0, 5)})
        result = apply_plan(sample_ds, plan)
        assert result.sizes["time"] == 5
        assert result.sizes["x"] == 5

    def test_roundtrip_with_constraint(self, sample_ds):
        constraint = Constraint(
            projections=[
                ProjectionItem(
                    name="temp",
                    slices=(
                        HyperSlab(start=2, stop=7, stride=1),
                        HyperSlab(start=1, stop=3, stride=1),
                    ),
                ),
            ],
        )
        plan = plan_subsetting(sample_ds, constraint)
        result = apply_plan(sample_ds, plan)
        assert result.sizes["time"] == 6  # indices 2-7 inclusive
        assert result.sizes["x"] == 3  # indices 1-3 inclusive
        assert "temp" in result
