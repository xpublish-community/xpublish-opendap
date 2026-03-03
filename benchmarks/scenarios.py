"""Benchmark scenario definitions for xpublish OpenDAP endpoints."""

from __future__ import annotations

from dataclasses import dataclass

import xarray as xr


# Expected content types for each DAP endpoint suffix.
EXPECTED_CONTENT_TYPES: dict[str, str] = {
    '.dds': 'text/plain',
    '.das': 'text/plain',
    '.dods': 'application/octet-stream',
    '.ver': 'text/plain',
    '.dmr': 'application/vnd.opendap.dap4.dataset-metadata+xml',
    '.dap': 'application/vnd.opendap.dap4.data',
    '.dsr': 'application/vnd.opendap.dap4.dataset-services+xml',
}

# DAP-specific response headers that prove the endpoint is real.
# Only DAP4 endpoints need header validation — older releases serve valid
# DAP2 responses without the xdods-server header, so requiring it would
# cause false negatives when benchmarking the official release.
EXPECTED_HEADERS: dict[str, str] = {
    '.dmr': 'xdap',
    '.dap': 'xdap',
    '.dsr': 'xdap',
}


@dataclass
class Scenario:
    """A single benchmark scenario targeting one endpoint."""

    name: str
    path: str  # URL path relative to base, e.g. /datasets/ds/opendap.dds
    category: str  # For grouping in output tables
    description: str
    expected_content_type: str | None = None  # Validate response Content-Type
    expected_header: str | None = None  # Validate presence of this response header


# Target total element count for spatial dimensions in data scenarios.
# Keeps request sizes reasonable for both local and remote-backed datasets.
# With float32 data, 10K elements ≈ 40KB — fast even over remote stores.
_SPATIAL_ELEMENT_BUDGET = 10_000


def _clamp(desired: int, dim_size: int) -> int:
    """Clamp an index to be within [0, dim_size - 1]."""
    return min(desired, dim_size - 1)


def _spatial_ends(spatial_shape: tuple[int, ...]) -> list[int]:
    """Compute per-dimension end indices that stay within the element budget.

    Distributes the budget evenly across spatial dimensions (Nth root),
    then clamps to actual dimension sizes. Returns 0-based end indices
    suitable for use in DAP hyperslab syntax ``[0:1:end]``.
    """
    n = len(spatial_shape)
    if n == 0:
        return []
    per_dim = int(_SPATIAL_ELEMENT_BUDGET ** (1.0 / n))
    per_dim = max(per_dim, 1)
    return [min(per_dim, s) - 1 for s in spatial_shape]


def _find_data_var(ds: xr.Dataset, min_ndim: int = 3) -> str | None:
    """Find the first data variable with at least min_ndim dimensions."""
    for name in ds.data_vars:
        if ds[name].ndim >= min_ndim:
            return str(name)
    return None


def build_scenarios(ds: xr.Dataset, dataset_id: str) -> list[Scenario]:
    """Build benchmark scenarios by inspecting the dataset.

    Args:
        ds: The xarray Dataset to benchmark against.
        dataset_id: The dataset ID used in URL paths.

    Returns:
        List of Scenario objects covering metadata, data, and parsing endpoints.
    """
    prefix = f'/datasets/{dataset_id}/opendap'
    scenarios: list[Scenario] = []

    # Find a suitable data variable (3D+) for spatial scenarios
    data_var = _find_data_var(ds, min_ndim=3)

    # Get list of all variable names for projection scenarios
    all_vars = list(ds.data_vars)
    first_var = str(all_vars[0]) if all_vars else None
    second_var = str(all_vars[1]) if len(all_vars) > 1 else first_var

    # --- Metadata scenarios (6) ---
    scenarios.append(Scenario(
        name='metadata-dds-full',
        path=f'{prefix}.dds',
        category='Metadata',
        description='DDS with no constraint (full dataset structure)',
    ))
    scenarios.append(Scenario(
        name='metadata-das-full',
        path=f'{prefix}.das',
        category='Metadata',
        description='DAS with no constraint (full attribute structure)',
    ))
    scenarios.append(Scenario(
        name='metadata-ver',
        path=f'{prefix}.ver',
        category='Metadata',
        description='DAP2 version response',
    ))
    scenarios.append(Scenario(
        name='metadata-dmr-full',
        path=f'{prefix}.dmr',
        category='Metadata',
        description='DAP4 DMR with no constraint (full metadata)',
    ))
    scenarios.append(Scenario(
        name='metadata-dsr',
        path=f'{prefix}.dsr',
        category='Metadata',
        description='DAP4 Dataset Services Response',
    ))

    if first_var:
        scenarios.append(Scenario(
            name='metadata-dds-projected',
            path=f'{prefix}.dds?{first_var}',
            category='Metadata',
            description=f'DDS with single variable projection ({first_var})',
        ))

    # --- Data scenarios (DAP2 DODS) ---
    if data_var:
        dims = ds[data_var].dims
        shape = ds[data_var].shape
        spatial_shape = shape[1:]
        sp_ends = _spatial_ends(spatial_shape)

        # Scalar point: first element of every dimension
        indices = ''.join('[0]' for _ in dims)
        scenarios.append(Scenario(
            name='data-dap2-scalar',
            path=f'{prefix}.dods?{data_var}{indices}',
            category='Data (DAP2)',
            description=f'Single scalar point from {data_var}',
        ))

        # Single timestep (first dim = 0, spatial dims budget-capped)
        slices = '[0]' + ''.join(f'[0:1:{e}]' for e in sp_ends)
        scenarios.append(Scenario(
            name='data-dap2-single-timestep',
            path=f'{prefix}.dods?{data_var}{slices}',
            category='Data (DAP2)',
            description=f'Single timestep, spatial region for {data_var}',
        ))

        # 10-timestep range
        t_end = _clamp(9, shape[0])
        slices_10t = f'[0:1:{t_end}]' + ''.join(f'[0:1:{e}]' for e in sp_ends)
        scenarios.append(Scenario(
            name='data-dap2-10-timesteps',
            path=f'{prefix}.dods?{data_var}{slices_10t}',
            category='Data (DAP2)',
            description=f'10 timesteps, spatial region for {data_var}',
        ))

        # Zonal transect (first dim = 0, second dim = 0, remaining dims capped)
        if len(dims) >= 3:
            trailing_ends = _spatial_ends(shape[2:])
            zonal = '[0][0]' + ''.join(f'[0:1:{e}]' for e in trailing_ends)
            scenarios.append(Scenario(
                name='data-dap2-zonal-transect',
                path=f'{prefix}.dods?{data_var}{zonal}',
                category='Data (DAP2)',
                description=f'Zonal transect (1D slice) for {data_var}',
            ))

        # Two-variable projection (both constrained to scalar points)
        if second_var and second_var != data_var:
            second_indices = ''.join('[0]' for _ in ds[second_var].dims)
            scenarios.append(Scenario(
                name='data-dap2-two-var',
                path=f'{prefix}.dods?{data_var}{indices},{second_var}{second_indices}',
                category='Data (DAP2)',
                description=f'Two variables: {data_var} (point) + {second_var} (point)',
            ))

        # Strided subsample (stride of 2, budget-capped extent)
        strided = f'[0:2:{_clamp(9, shape[0])}]' + ''.join(
            f'[0:2:{e}]' for e in sp_ends
        )
        scenarios.append(Scenario(
            name='data-dap2-strided',
            path=f'{prefix}.dods?{data_var}{strided}',
            category='Data (DAP2)',
            description=f'Strided subsample (stride=2) for {data_var}',
        ))

    # --- Data scenarios (DAP4) ---
    if data_var:
        dims = ds[data_var].dims
        shape = ds[data_var].shape
        spatial_shape = shape[1:]
        sp_ends = _spatial_ends(spatial_shape)

        # DAP4 scalar
        d4_indices = ''.join('[0]' for _ in dims)
        scenarios.append(Scenario(
            name='data-dap4-scalar',
            path=f'{prefix}.dap?dap4.ce=/{data_var}{d4_indices}',
            category='Data (DAP4)',
            description=f'DAP4 single scalar point from {data_var}',
        ))

        # DAP4 single timestep
        d4_slices = '[0]' + ''.join(f'[0:1:{e}]' for e in sp_ends)
        scenarios.append(Scenario(
            name='data-dap4-single-timestep',
            path=f'{prefix}.dap?dap4.ce=/{data_var}{d4_slices}',
            category='Data (DAP4)',
            description=f'DAP4 single timestep, spatial region for {data_var}',
        ))

        # DAP4 10-timestep range
        t_end = _clamp(9, shape[0])
        d4_slices_10t = f'[0:1:{t_end}]' + ''.join(f'[0:1:{e}]' for e in sp_ends)
        scenarios.append(Scenario(
            name='data-dap4-10-timesteps',
            path=f'{prefix}.dap?dap4.ce=/{data_var}{d4_slices_10t}',
            category='Data (DAP4)',
            description=f'DAP4 10 timesteps, spatial region for {data_var}',
        ))

        # DAP4 multi-variable (both constrained to scalar points)
        if second_var and second_var != data_var:
            d4_second_indices = ''.join('[0]' for _ in ds[second_var].dims)
            scenarios.append(Scenario(
                name='data-dap4-multi-var',
                path=f'{prefix}.dap?dap4.ce=/{data_var}{d4_indices};/{second_var}{d4_second_indices}',
                category='Data (DAP4)',
                description=f'DAP4 multi-variable: {data_var} (point) + {second_var} (point)',
            ))

    # --- Constraint parsing scenarios (all use .dds to isolate parsing overhead) ---
    scenarios.append(Scenario(
        name='parsing-no-constraint',
        path=f'{prefix}.dds',
        category='Constraint Parsing',
        description='DDS with no constraint expression',
    ))

    if first_var:
        scenarios.append(Scenario(
            name='parsing-projection-only',
            path=f'{prefix}.dds?{first_var}',
            category='Constraint Parsing',
            description=f'DDS with projection-only constraint ({first_var})',
        ))

    if data_var:
        dims = ds[data_var].dims
        shape = ds[data_var].shape

        # Single index
        single_idx = ''.join(f'[0]' for _ in dims)
        scenarios.append(Scenario(
            name='parsing-single-index',
            path=f'{prefix}.dds?{data_var}{single_idx}',
            category='Constraint Parsing',
            description=f'DDS with single-index constraint on {data_var}',
        ))

        # Range
        range_slices = ''.join(f'[0:1:{_clamp(s - 1, s)}]' for s in shape)
        scenarios.append(Scenario(
            name='parsing-range',
            path=f'{prefix}.dds?{data_var}{range_slices}',
            category='Constraint Parsing',
            description=f'DDS with range constraint on {data_var}',
        ))

        # Multi-variable with hyperslabs
        if second_var and second_var != data_var:
            second_shape = ds[second_var].shape
            second_slices = ''.join(
                f'[0:1:{_clamp(s - 1, s)}]' for s in second_shape
            ) if second_shape else ''
            scenarios.append(Scenario(
                name='parsing-multi-var-hyperslab',
                path=f'{prefix}.dds?{data_var}{range_slices},{second_var}{second_slices}',
                category='Constraint Parsing',
                description=f'DDS with multi-var hyperslabs ({data_var} + {second_var})',
            ))

    # Auto-fill expected_content_type and expected_header from the path suffix
    for s in scenarios:
        # Extract suffix like ".dds", ".dap" from path (before query string)
        path_part = s.path.split('?')[0]
        for suffix in EXPECTED_CONTENT_TYPES:
            if path_part.endswith(suffix):
                s.expected_content_type = EXPECTED_CONTENT_TYPES[suffix]
                s.expected_header = EXPECTED_HEADERS.get(suffix)
                break

    return scenarios
