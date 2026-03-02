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
        assert plan.variables is not None
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
        assert plan.variables is not None
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
        assert plan.variables is not None
        assert "temp" in plan.variables
        assert plan.isel_args["time"] == slice(0, 5, 1)
        assert plan.isel_args["x"] == slice(0, 3, 1)

    def test_grid_map_member_path(self, sample_ds):
        """Grid map member path temp.time resolves to coordinate time."""
        constraint = Constraint(projections=[ProjectionItem(name="temp.time")])
        plan = plan_subsetting(sample_ds, constraint)
        assert plan.variables is not None
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


class TestEdgeCaseSubsetting:
    """Edge-case subsetting plans."""

    def test_single_element_hyperslab(self, sample_ds):
        """[0:0] should select exactly 1 element."""
        constraint = Constraint(
            projections=[
                ProjectionItem(
                    name="temp",
                    slices=(
                        HyperSlab(start=0, stop=0, stride=1),
                        HyperSlab(start=0, stop=0, stride=1),
                    ),
                ),
            ],
        )
        plan = plan_subsetting(sample_ds, constraint)
        result = apply_plan(sample_ds, plan)
        assert result.sizes["time"] == 1
        assert result.sizes["x"] == 1

    def test_large_stride_single_output(self, sample_ds):
        """Stride larger than range should yield 1 element."""
        constraint = Constraint(
            projections=[
                ProjectionItem(
                    name="temp",
                    slices=(
                        HyperSlab(start=0, stop=3, stride=100),
                        HyperSlab(start=0, stop=4, stride=1),
                    ),
                ),
            ],
        )
        plan = plan_subsetting(sample_ds, constraint)
        result = apply_plan(sample_ds, plan)
        assert result.sizes["time"] == 1
        assert result.sizes["x"] == 5

    def test_start_equals_stop(self, sample_ds):
        """[3:3] is valid and selects index 3."""
        constraint = Constraint(
            projections=[
                ProjectionItem(
                    name="temp",
                    slices=(
                        HyperSlab(start=3, stop=3, stride=1),
                        HyperSlab(start=0, stop=4, stride=1),
                    ),
                ),
            ],
        )
        plan = plan_subsetting(sample_ds, constraint)
        result = apply_plan(sample_ds, plan)
        assert result.sizes["time"] == 1

    def test_mixed_constrained_unconstrained(self, sample_ds):
        """One variable sliced, one variable bare."""
        constraint = Constraint(
            projections=[
                ProjectionItem(
                    name="temp",
                    slices=(
                        HyperSlab(start=0, stop=4, stride=1),
                        HyperSlab(start=0, stop=2, stride=1),
                    ),
                ),
                ProjectionItem(name="time"),
            ],
        )
        plan = plan_subsetting(sample_ds, constraint)
        assert plan.variables is not None
        assert "temp" in plan.variables
        assert "time" in plan.variables
        # Slices should still be applied
        assert "time" in plan.isel_args
        result = apply_plan(sample_ds, plan)
        assert "temp" in result

    def test_coord_auto_inclusion_for_data_var(self, sample_ds):
        """Projecting only 'temp' auto-adds 'time' and 'x' coords."""
        constraint = Constraint(projections=[ProjectionItem(name="temp")])
        plan = plan_subsetting(sample_ds, constraint)
        assert plan.variables is not None
        assert "temp" in plan.variables
        assert "time" in plan.variables
        assert "x" in plan.variables

    def test_single_dim_dataset(self):
        """Dataset with a single dimension, hyperslab [0:0]."""
        ds = xr.Dataset(
            {"v": xr.DataArray(np.array([42.0], dtype="float64"), dims=["t"])},
            coords={"t": np.array([0])},
        )
        constraint = Constraint(
            projections=[
                ProjectionItem(
                    name="v",
                    slices=(HyperSlab(start=0, stop=0, stride=1),),
                ),
            ],
        )
        plan = plan_subsetting(ds, constraint)
        result = apply_plan(ds, plan)
        assert result.sizes["t"] == 1
        assert float(result["v"].values[0]) == 42.0


class TestConflictingSlices:
    """Tests for conflicting/duplicate projections on the same variable."""

    def test_conflicting_slices_last_wins(self, sample_ds):
        """Two ProjectionItems for same var with different slices — last wins."""
        constraint = Constraint(
            projections=[
                ProjectionItem(
                    name="temp",
                    slices=(
                        HyperSlab(start=0, stop=4, stride=1),
                        HyperSlab(start=0, stop=2, stride=1),
                    ),
                ),
                ProjectionItem(
                    name="temp",
                    slices=(
                        HyperSlab(start=0, stop=1, stride=1),
                        HyperSlab(start=0, stop=1, stride=1),
                    ),
                ),
            ],
        )
        plan = plan_subsetting(sample_ds, constraint)
        # Last-wins behavior: second projection's slices should be used
        assert plan.isel_args["time"] == slice(0, 2, 1)
        assert plan.isel_args["x"] == slice(0, 2, 1)

    def test_duplicate_variable_projection(self, sample_ds):
        """Same var projected twice without slices — appears in variables list."""
        constraint = Constraint(
            projections=[
                ProjectionItem(name="temp"),
                ProjectionItem(name="temp"),
            ],
        )
        plan = plan_subsetting(sample_ds, constraint)
        result = apply_plan(sample_ds, plan)
        # Regardless of duplicates in plan.variables, the result should have temp
        assert "temp" in result


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


class TestValidationBranches:
    def test_negative_stop_raises(self, sample_ds):
        constraint = Constraint(
            projections=[
                ProjectionItem(
                    name="temp",
                    slices=(
                        HyperSlab(start=0, stop=-1, stride=1),
                        HyperSlab(start=0, stop=4, stride=1),
                    ),
                ),
            ],
        )
        with pytest.raises(IndexOutOfRangeError, match="Negative"):
            plan_subsetting(sample_ds, constraint)

    def test_start_exceeds_dim_size_raises(self, sample_ds):
        constraint = Constraint(
            projections=[
                ProjectionItem(
                    name="temp",
                    slices=(
                        HyperSlab(start=100, stop=100, stride=1),
                        HyperSlab(start=0, stop=4, stride=1),
                    ),
                ),
            ],
        )
        with pytest.raises(IndexOutOfRangeError, match="exceeds"):
            plan_subsetting(sample_ds, constraint)

    def test_start_greater_than_stop_raises(self, sample_ds):
        constraint = Constraint(
            projections=[
                ProjectionItem(
                    name="temp",
                    slices=(
                        HyperSlab(start=5, stop=2, stride=1),
                        HyperSlab(start=0, stop=4, stride=1),
                    ),
                ),
            ],
        )
        with pytest.raises(IndexOutOfRangeError, match="Start index 5 > stop index 2"):
            plan_subsetting(sample_ds, constraint)


class TestEstimateMemoryEdgeCases:
    def test_plan_with_nonexistent_variable(self, sample_ds):
        from xpublish_opendap.dap.xarray_adapter import SubsettingPlan, _estimate_memory

        plan = SubsettingPlan(variables=["nonexistent", "temp", "time", "x"])
        mem = _estimate_memory(sample_ds, plan)
        assert mem > 0

    def test_estimate_memory_unsupported_dtype(self):
        from xpublish_opendap.dap.xarray_adapter import SubsettingPlan, _estimate_memory

        ds = xr.Dataset(
            {
                "cvar": xr.DataArray(
                    np.array([1 + 2j, 3 + 4j], dtype="complex128"), dims=["x"]
                )
            },
            coords={"x": np.arange(2)},
        )
        plan = SubsettingPlan(variables=["cvar"])
        mem = _estimate_memory(ds, plan)
        # Falls back to var.dtype.itemsize (16 for complex128)
        assert mem == 2 * 16
