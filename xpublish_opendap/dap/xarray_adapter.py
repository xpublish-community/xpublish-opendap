"""Constraint expression to xarray subsetting adapter."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import xarray as xr

from xpublish_opendap.dap.constraint import Constraint, HyperSlab
from xpublish_opendap.dap.types import cf_encode_variable, resolve_dap_type
from xpublish_opendap.errors import IndexOutOfRangeError, VariableNotFoundError


@dataclass
class SubsettingPlan:
    """An executable plan for subsetting an xr.Dataset."""

    variables: list[str] | None = None
    isel_args: dict[str, slice] = field(default_factory=dict)
    estimated_bytes: int = 0


def plan_subsetting(ds: xr.Dataset, constraint: Constraint) -> SubsettingPlan:
    """Build a subsetting plan from a parsed constraint.

    Converts DAP inclusive-stop hyperslabs to Python exclusive-stop slices,
    validates variable names and index bounds, and estimates the memory
    footprint of the resulting subset.

    Args:
        ds: The source dataset.
        constraint: The parsed constraint expression.

    Returns:
        A SubsettingPlan ready for apply_plan().

    Raises:
        VariableNotFoundError: If a projected variable doesn't exist.
        IndexOutOfRangeError: If a hyperslab index exceeds dimension bounds.
    """
    plan = SubsettingPlan()

    all_var_names = set(ds.data_vars) | set(ds.coords)

    if not constraint.projections:
        # No projection = all variables
        plan.variables = None
    else:
        plan.variables = []
        for proj in constraint.projections:
            if proj.name not in all_var_names:
                raise VariableNotFoundError(proj.name)
            plan.variables.append(proj.name)

            # Collect dimension slices from hyperslab specifications
            if proj.slices is not None:
                var = ds[proj.name]
                if len(proj.slices) != len(var.dims):
                    raise IndexOutOfRangeError(
                        f'Variable {proj.name!r} has {len(var.dims)} dimensions '
                        f'but {len(proj.slices)} hyperslab(s) given'
                    )
                for dim, slab in zip(var.dims, proj.slices):
                    dim_size = ds.sizes[dim]
                    _validate_hyperslab(slab, dim, dim_size)
                    new_slice = _hyperslab_to_slice(slab)

                    # If this dimension already has a slice, take the more
                    # restrictive one (intersection would be complex; for now
                    # last write wins, which matches DAP2 behavior)
                    plan.isel_args[dim] = new_slice

        # Ensure coordinate variables for selected data variables are included
        coords_to_add = set()
        for var_name in plan.variables:
            if var_name in ds.data_vars:
                for dim in ds[var_name].dims:
                    if dim in ds.coords and dim not in plan.variables:
                        coords_to_add.add(dim)
        plan.variables.extend(sorted(coords_to_add))

    # Estimate memory
    plan.estimated_bytes = _estimate_memory(ds, plan)

    return plan


def apply_plan(ds: xr.Dataset, plan: SubsettingPlan) -> xr.Dataset:
    """Apply a SubsettingPlan to a lazy xr.Dataset.

    Returns a new Dataset with only selected variables and sliced dimensions.
    No data is loaded - this operates on the dask/numpy computation graph.

    Args:
        ds: The source (lazy) dataset.
        plan: The subsetting plan from plan_subsetting().

    Returns:
        A new subsetted Dataset.
    """
    result = ds

    # Apply dimension slicing first
    if plan.isel_args:
        result = result.isel(plan.isel_args)

    # Select variables
    if plan.variables is not None:
        # Keep only the requested variables and their coordinate dependencies
        keep_vars = set(plan.variables)
        # Always include dimension coordinates that the selected variables depend on
        for var_name in plan.variables:
            if var_name in result:
                for dim in result[var_name].dims:
                    if dim in result.coords:
                        keep_vars.add(dim)

        all_vars = set(result.data_vars) | set(result.coords)
        drop_vars = all_vars - keep_vars
        # Only drop variables that exist and aren't index coordinates we need
        drop_vars = {v for v in drop_vars if v in result}
        if drop_vars:
            result = result.drop_vars(drop_vars)

    return result


def _validate_hyperslab(slab: HyperSlab, dim: str, dim_size: int) -> None:
    """Validate that hyperslab indices are within bounds."""
    if slab.start < 0:
        raise IndexOutOfRangeError(
            f'Negative start index {slab.start} for dimension {dim!r}'
        )
    if slab.stop < 0:
        raise IndexOutOfRangeError(
            f'Negative stop index {slab.stop} for dimension {dim!r}'
        )
    if slab.start >= dim_size:
        raise IndexOutOfRangeError(
            f'Start index {slab.start} exceeds dimension {dim!r} size {dim_size}'
        )
    if slab.stop >= dim_size:
        raise IndexOutOfRangeError(
            f'Stop index {slab.stop} exceeds dimension {dim!r} size {dim_size}'
        )
    if slab.start > slab.stop:
        raise IndexOutOfRangeError(
            f'Start index {slab.start} > stop index {slab.stop} '
            f'for dimension {dim!r}'
        )


def _hyperslab_to_slice(slab: HyperSlab) -> slice:
    """Convert a DAP hyperslab (inclusive stop) to a Python slice (exclusive stop)."""
    return slice(slab.start, slab.stop + 1, slab.stride)


def _estimate_memory(ds: xr.Dataset, plan: SubsettingPlan) -> int:
    """Estimate memory footprint of the subsetted result in bytes."""
    total = 0

    if plan.variables is not None:
        var_names = plan.variables
    else:
        var_names = list(ds.data_vars) + list(ds.coords)

    for var_name in var_names:
        if var_name not in ds:
            continue
        var = ds[var_name]

        # Compute subsetted shape
        shape = []
        for dim in var.dims:
            dim_size = ds.sizes[dim]
            if dim in plan.isel_args:
                s = plan.isel_args[dim]
                start = s.start or 0
                stop = min(s.stop or dim_size, dim_size)
                stride = s.step or 1
                shape.append(math.ceil((stop - start) / stride))
            else:
                shape.append(dim_size)

        # CF-encode to get the wire dtype
        try:
            encoded = cf_encode_variable(var.variable)
            dap_type = resolve_dap_type(encoded.dtype)
            item_size = dap_type.xdr_wire_size if dap_type.xdr_wire_size > 0 else 64
        except (ValueError, AttributeError):
            item_size = var.dtype.itemsize or 8

        n_elements = math.prod(shape) if shape else 1
        total += n_elements * item_size

    return total
