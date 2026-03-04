#!/usr/bin/env python
"""Standalone IO performance benchmark for xpublish-opendap.

Isolates the data loading + encoding pipeline without any HTTP server,
FastAPI, or xpublish overhead. Benchmarks different streaming strategies
to identify where time is spent.

Strategies:
    eager           Load entire subsetted variable at once (dask parallelizes
                    all chunk fetches), then encode.
    slab-prefetch-1 Current slab streaming: load slab N+1 while encoding slab N.
    slab-prefetch-2 Prefetch 2 slabs ahead instead of 1.
    slab-batch-N    Group N outermost chunks into larger slabs, reducing
                    sequential round-trips while bounding memory.

Usage:
    python benchmarks/io_bench.py                              # default: air, all strategies
    python benchmarks/io_bench.py --dataset air -n 2           # quick local test
    python benchmarks/io_bench.py --dataset ifs -n 3           # full IFS benchmark
    python benchmarks/io_bench.py --scenario 10-timesteps      # specific scenario
    python benchmarks/io_bench.py --strategy eager,slab-prefetch-1
    python benchmarks/io_bench.py --encoding dap4              # DAP4 encoding
    python benchmarks/io_bench.py --var 2t                     # different variable
    python benchmarks/io_bench.py --strategy slab-batch --batch-sizes 1,5,10
    python benchmarks/io_bench.py --warmup 2 -n 3             # 2 warmup rounds + 3 timed
    python benchmarks/io_bench.py --isolate -n 2              # subprocess isolation
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import subprocess
import sys
import time
from collections import defaultdict
from collections.abc import AsyncIterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import xarray as xr

# Ensure the project root is on sys.path so benchmarks/ and xpublish_opendap/
# are importable when run as `python benchmarks/io_bench.py`.
_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import dask

from benchmarks.run import load_dataset
from xpublish_opendap.dap.dap2.dods import _xdr_encode_array, _xdr_encode_slab_data
from xpublish_opendap.dap.dap4.data import _dap4_encode_variable
from xpublish_opendap.dap.types import cf_encode_variable, resolve_dap_type
from xpublish_opendap.io import get_slab_boundaries, load_variable, run_in_executor

# Default rechunk byte budget for dask-graph strategy (matches old opendap-protocol)
_RECHUNK_BYTES = 20_000_000


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class Scenario:
    """A subset pattern to benchmark."""

    name: str
    isel: dict
    description: str


@dataclass
class TrialResult:
    """Result of a single trial run."""

    strategy: str
    scenario: str
    ttfb_s: float
    total_s: float
    encoded_bytes: int
    n_slabs: int
    iteration: int = 0
    is_warmup: bool = False

    @property
    def throughput_mbps(self) -> float:
        if self.total_s <= 0:
            return 0.0
        return (self.encoded_bytes / (1024 * 1024)) / self.total_s


# ---------------------------------------------------------------------------
# Sync helpers for thread-pool execution
# ---------------------------------------------------------------------------


def _load_and_encode_full(
    da: xr.DataArray,
    protocol: str,
    dask_num_workers: int,
) -> bytes:
    """Load full DataArray and encode (runs in thread pool)."""
    load_variable(da, dask_num_workers)
    encoded_var = cf_encode_variable(da.variable)
    data = np.asarray(encoded_var.data)
    dap_type = resolve_dap_type(encoded_var.dtype, protocol=protocol)
    if protocol == "dap4":
        return b"".join(_dap4_encode_variable(data, dap_type))
    return b"".join(_xdr_encode_array(data, dap_type))


def _load_and_encode_slab(
    da: xr.DataArray,
    dim: str,
    start: int,
    stop: int,
    protocol: str,
    dask_num_workers: int,
) -> bytes:
    """Load a slab and encode data bytes (no length prefix, runs in thread pool)."""
    slab = da.isel({dim: slice(start, stop)})
    load_variable(slab, dask_num_workers)
    encoded_var = cf_encode_variable(slab.variable)
    data = np.asarray(encoded_var.data)
    dap_type = resolve_dap_type(encoded_var.dtype, protocol=protocol)
    if protocol == "dap4":
        if data.ndim == 0:
            data = data.reshape(1)
        target_dtype = data.dtype.newbyteorder("=")
        return data.astype(target_dtype, copy=False).tobytes()
    return b"".join(_xdr_encode_slab_data(data, dap_type))


def _dask_graph_encode_full(
    da: xr.DataArray,
    protocol: str,
    dask_num_workers: int,
    rechunk_bytes: int = _RECHUNK_BYTES,
) -> bytes:
    """Old opendap-protocol pattern: ravel, rechunk, fused astype, block-by-block compute.

    Mirrors dods_encode() from the opendap-protocol library:
    - Keeps data lazy until the last moment
    - Fuses type conversion (big-endian byteswap) into the dask graph
    - Computes block-by-block, letting dask parallelize chunk fetches within each block
    - No cf_encode_variable, no xarray .load()
    """
    dap_type = resolve_dap_type(da.dtype, protocol=protocol)

    if protocol == "dap4":
        wire_dtype = da.dtype.newbyteorder("=")
    elif dap_type.xdr_format is not None:
        wire_dtype = np.dtype(dap_type.xdr_format)
    else:
        wire_dtype = da.dtype

    if hasattr(da.data, "chunks") and da.data.chunks is not None:
        arr = da.data
        chunk_elems = max(1, int(rechunk_bytes / max(arr.dtype.itemsize, 1)))
        flat = arr.ravel().rechunk(chunk_elems)

        parts = []
        with dask.config.set(scheduler="threads", num_workers=dask_num_workers):
            for block in flat.blocks:
                parts.append(block.astype(wire_dtype).compute().tobytes())
        return b"".join(parts)
    else:
        data = np.asarray(da.data)
        return data.astype(wire_dtype).tobytes()


def _load_and_encode_nocf(
    da: xr.DataArray,
    protocol: str,
    dask_num_workers: int,
) -> bytes:
    """Like eager but skip cf_encode_variable for non-datetime types."""
    load_variable(da, dask_num_workers)

    if da.dtype.kind in ("M", "m"):
        encoded_var = cf_encode_variable(da.variable)
        data = np.asarray(encoded_var.data)
        dap_type = resolve_dap_type(encoded_var.dtype, protocol=protocol)
    else:
        data = np.asarray(da.data)
        dap_type = resolve_dap_type(da.dtype, protocol=protocol)

    if protocol == "dap4":
        return b"".join(_dap4_encode_variable(data, dap_type))
    return b"".join(_xdr_encode_array(data, dap_type))


# ---------------------------------------------------------------------------
# Strategy async generators — each yields encoded byte chunks
# ---------------------------------------------------------------------------


async def eager_load_encode(
    da: xr.DataArray,
    protocol: str,
    dask_num_workers: int,
) -> AsyncIterator[bytes]:
    """Load entire subsetted variable at once, then encode.

    Dask parallelizes all chunk fetches. Single yield of all bytes.
    """
    result = await run_in_executor(
        _load_and_encode_full, da, protocol, dask_num_workers
    )
    yield result


async def dask_graph_encode(
    da: xr.DataArray,
    protocol: str,
    dask_num_workers: int,
) -> AsyncIterator[bytes]:
    """Old opendap-protocol pattern: fused dask graph, rechunk, block-by-block compute.

    Tests hypothesis: is the dask ravel/rechunk/block approach fundamentally
    faster than load-then-encode?
    """
    result = await run_in_executor(
        _dask_graph_encode_full, da, protocol, dask_num_workers
    )
    yield result


async def eager_nocf_load_encode(
    da: xr.DataArray,
    protocol: str,
    dask_num_workers: int,
) -> AsyncIterator[bytes]:
    """Like eager but skip CF encoding for non-datetime types.

    Tests hypothesis: does cf_encode_variable add significant overhead?
    """
    result = await run_in_executor(
        _load_and_encode_nocf, da, protocol, dask_num_workers
    )
    yield result


async def eager_sync_load_encode(
    da: xr.DataArray,
    protocol: str,
    dask_num_workers: int,
) -> AsyncIterator[bytes]:
    """Like eager but runs synchronously — no executor, no semaphore.

    Tests hypothesis: does the async executor machinery add significant overhead?
    Blocks the event loop, but fine for benchmarking.
    """
    result = _load_and_encode_full(da, protocol, dask_num_workers)
    yield result


async def slab_stream(
    da: xr.DataArray,
    protocol: str,
    dask_num_workers: int,
    prefetch: int = 1,
    batch_size: int = 1,
) -> AsyncIterator[bytes]:
    """Stream data slab-by-slab along the outermost dimension.

    Args:
        da: The (lazy) DataArray.
        protocol: 'dap2' or 'dap4'.
        dask_num_workers: Dask thread count for chunk loading.
        prefetch: Number of slabs to prefetch ahead.
        batch_size: Number of outermost chunks per slab.
    """
    slab_info = get_slab_boundaries(da, threshold_bytes=0)
    if slab_info is None:
        # No slab streaming possible (in-memory or single outer chunk) — fall back to eager
        result = await run_in_executor(
            _load_and_encode_full, da, protocol, dask_num_workers
        )
        yield result
        return

    dim, boundaries = slab_info
    # Merge adjacent boundaries into larger batches
    batched = _batch_boundaries(boundaries, batch_size)

    # Fill the prefetch buffer
    pending: list[asyncio.Task] = []
    for i in range(min(prefetch, len(batched))):
        start, stop = batched[i]
        task = asyncio.ensure_future(
            run_in_executor(
                _load_and_encode_slab,
                da,
                dim,
                start,
                stop,
                protocol,
                dask_num_workers,
            ),
        )
        pending.append(task)

    next_to_launch = prefetch

    for _i in range(len(batched)):
        slab_bytes = await pending.pop(0)

        # Launch next prefetch
        if next_to_launch < len(batched):
            start, stop = batched[next_to_launch]
            task = asyncio.ensure_future(
                run_in_executor(
                    _load_and_encode_slab,
                    da,
                    dim,
                    start,
                    stop,
                    protocol,
                    dask_num_workers,
                ),
            )
            pending.append(task)
            next_to_launch += 1

        yield slab_bytes


def _batch_boundaries(
    boundaries: list[tuple[int, int]],
    batch_size: int,
) -> list[tuple[int, int]]:
    """Merge adjacent slab boundaries into larger batches."""
    if batch_size <= 1:
        return boundaries
    batched = []
    for i in range(0, len(boundaries), batch_size):
        group = boundaries[i : i + batch_size]
        batched.append((group[0][0], group[-1][1]))
    return batched


# ---------------------------------------------------------------------------
# Subprocess isolation helpers
# ---------------------------------------------------------------------------


def _serialize_isel(isel: dict) -> str:
    """Serialize an isel dict (may contain slices) to JSON."""

    def _encode(v):
        if isinstance(v, slice):
            return {"__slice__": True, "start": v.start, "stop": v.stop, "step": v.step}
        return v

    return json.dumps({k: _encode(v) for k, v in isel.items()})


def _deserialize_isel(s: str) -> dict:
    """Deserialize a JSON isel dict back (restoring slices)."""
    raw = json.loads(s)

    def _decode(v):
        if isinstance(v, dict) and v.get("__slice__"):
            return slice(v.get("start"), v.get("stop"), v.get("step"))
        return v

    return {k: _decode(v) for k, v in raw.items()}


def _run_isolated_trial(
    *,
    dataset: str,
    var_name: str,
    isel_json: str,
    strategy: str,
    encoding: str,
    batch_size: int,
    dask_workers: int,
) -> TrialResult:
    """Run a single trial in a subprocess and return the result."""
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--_trial",
        "--dataset",
        dataset,
        "--var",
        var_name,
        "--_isel",
        isel_json,
        "--strategy",
        strategy,
        "--encoding",
        encoding,
        "--batch-sizes",
        str(batch_size),
        "--dask-workers",
        str(dask_workers),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    if result.returncode != 0:
        raise RuntimeError(
            f"Subprocess trial failed (exit {result.returncode}):\n{result.stderr}",
        )
    # Find the JSON line in stdout (skip any other output)
    for line in result.stdout.strip().splitlines():
        line = line.strip()
        if line.startswith("{"):
            data = json.loads(line)
            return TrialResult(
                strategy=data["strategy"],
                scenario=data["scenario"],
                ttfb_s=data["ttfb_s"],
                total_s=data["total_s"],
                encoded_bytes=data["encoded_bytes"],
                n_slabs=data["n_slabs"],
            )
    raise RuntimeError(f"No JSON result in subprocess output:\n{result.stdout}")


async def _trial_subprocess_main(args: argparse.Namespace) -> None:
    """Entry point for --_trial subprocess mode: run one trial, print JSON, exit."""
    protocol = "dap4" if args.encoding == "dap4" else "dap2"
    ds, _dataset_id = load_dataset(args.dataset)
    var_name = args.var
    isel = _deserialize_isel(args._isel)
    da = ds[var_name].isel(isel)

    strategy = args.strategy
    batch_sizes = [int(x) for x in args.batch_sizes.split(",")]
    batch_size = batch_sizes[0] if batch_sizes else 1

    trial = await run_trial(
        da, strategy, protocol, args.dask_workers, batch_size=batch_size
    )
    trial.scenario = ""

    result = {
        "strategy": trial.strategy,
        "scenario": trial.scenario,
        "ttfb_s": trial.ttfb_s,
        "total_s": trial.total_s,
        "encoded_bytes": trial.encoded_bytes,
        "n_slabs": trial.n_slabs,
    }
    print(json.dumps(result), flush=True)


# ---------------------------------------------------------------------------
# Harness
# ---------------------------------------------------------------------------


async def run_trial(
    da: xr.DataArray,
    strategy: str,
    protocol: str,
    dask_num_workers: int,
    batch_size: int = 1,
) -> TrialResult:
    """Time a single strategy run and return metrics."""
    slab_info = get_slab_boundaries(da, threshold_bytes=0)
    boundaries = slab_info[1] if slab_info is not None else None

    if strategy == "eager":
        gen = eager_load_encode(da, protocol, dask_num_workers)
        n_slabs = 0
    elif strategy == "dask-graph":
        gen = dask_graph_encode(da, protocol, dask_num_workers)
        n_slabs = 0
    elif strategy == "eager-nocf":
        gen = eager_nocf_load_encode(da, protocol, dask_num_workers)
        n_slabs = 0
    elif strategy == "eager-sync":
        gen = eager_sync_load_encode(da, protocol, dask_num_workers)
        n_slabs = 0
    elif strategy.startswith("slab-prefetch-"):
        prefetch = int(strategy.split("-")[-1])
        gen = slab_stream(
            da, protocol, dask_num_workers, prefetch=prefetch, batch_size=1
        )
        n_slabs = len(boundaries) if boundaries else 0
    elif strategy == "slab-batch":
        gen = slab_stream(
            da, protocol, dask_num_workers, prefetch=1, batch_size=batch_size
        )
        if boundaries:
            n_slabs = -(-len(boundaries) // batch_size)  # ceiling division
        else:
            n_slabs = 0
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    display_name = f"slab-batch-{batch_size}" if strategy == "slab-batch" else strategy

    total_bytes = 0
    ttfb = None
    t0 = time.perf_counter()

    async for chunk in gen:
        if ttfb is None:
            ttfb = time.perf_counter() - t0
        total_bytes += len(chunk)

    total = time.perf_counter() - t0

    return TrialResult(
        strategy=display_name,
        scenario="",
        ttfb_s=ttfb if ttfb is not None else total,
        total_s=total,
        encoded_bytes=total_bytes,
        n_slabs=n_slabs,
    )


# ---------------------------------------------------------------------------
# Scenarios
# ---------------------------------------------------------------------------

_SPATIAL_ELEMENT_BUDGET = 10_000


def _spatial_sizes(spatial_shape: tuple[int, ...]) -> list[int]:
    """Compute per-dim sizes that fit within spatial element budget."""
    n = len(spatial_shape)
    if n == 0:
        return []
    per_dim = int(_SPATIAL_ELEMENT_BUDGET ** (1.0 / n))
    per_dim = max(per_dim, 1)
    return [min(per_dim, s) for s in spatial_shape]


def build_scenarios(ds: xr.Dataset, var_name: str) -> list[Scenario]:
    """Build benchmark scenarios for a given variable."""
    da = ds[var_name]
    dims = da.dims
    shape = da.shape

    if da.ndim < 2:
        return [Scenario(name="full", isel={}, description=f"Full variable {var_name}")]

    spatial_shape = shape[1:]
    sp_sizes = _spatial_sizes(spatial_shape)

    scenarios = []

    # 1-timestep: 1 outer step, spatial dims budget-capped
    isel_1t: dict = {dims[0]: slice(0, 1)}
    for i, dim in enumerate(dims[1:]):
        isel_1t[dim] = slice(0, sp_sizes[i])
    scenarios.append(
        Scenario(
            name="1-timestep",
            isel=isel_1t,
            description="1 outer step, spatial budget-capped",
        ),
    )

    # 10-timesteps: 10 outer steps, spatial dims budget-capped
    t_end = min(10, shape[0])
    isel_10t: dict = {dims[0]: slice(0, t_end)}
    for i, dim in enumerate(dims[1:]):
        isel_10t[dim] = slice(0, sp_sizes[i])
    scenarios.append(
        Scenario(
            name="10-timesteps",
            isel=isel_10t,
            description=f"{t_end} outer steps, spatial budget-capped",
        ),
    )

    # full-spatial-1t: 1 outer step, full spatial extent
    scenarios.append(
        Scenario(
            name="full-spatial-1t",
            isel={dims[0]: slice(0, 1)},
            description="1 outer step, full spatial extent",
        ),
    )

    return scenarios


def _find_data_var(ds: xr.Dataset, min_ndim: int = 3) -> str | None:
    """Find the first data variable with at least min_ndim dimensions."""
    for name in ds.data_vars:
        if ds[name].ndim >= min_ndim:
            return str(name)
    return None


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------


def _fmt_time(seconds: float) -> str:
    if seconds < 0.001:
        return f"{seconds * 1_000_000:.0f}us"
    if seconds < 1.0:
        return f"{seconds * 1000:.0f}ms"
    return f"{seconds:.2f}s"


def _fmt_throughput(mbps: float) -> str:
    if mbps >= 1.0:
        return f"{mbps:.1f} MB/s"
    return f"{mbps * 1024:.0f} KB/s"


def _print_round_detail(round_results: list[TrialResult], round_label: str) -> None:
    """Print a single line showing each strategy's timing for one round."""
    parts = [f"{r.strategy} {_fmt_time(r.total_s)}" for r in round_results]
    print(f"  {round_label:>10}:  {' | '.join(parts)}")


def print_summary(results: list[TrialResult]) -> None:
    """Print a median-based summary table (warmup results excluded by caller)."""
    grouped: dict[str, list[TrialResult]] = defaultdict(list)
    for r in results:
        grouped[r.strategy].append(r)

    header = f"{'Strategy':<22} {'TTFB(med)':>10} {'Total(med)':>10} {'Throughput':>12} {'Slabs':>6}"
    sep = "\u2500" * len(header)
    print(header)
    print(sep)

    for strategy, trials in grouped.items():
        med_ttfb = statistics.median(t.ttfb_s for t in trials)
        med_total = statistics.median(t.total_s for t in trials)
        med_throughput = statistics.median(t.throughput_mbps for t in trials)
        n_slabs = trials[0].n_slabs
        slabs_str = str(n_slabs) if n_slabs > 0 else "-"
        print(
            f"{strategy:<22} "
            f"{_fmt_time(med_ttfb):>10} "
            f"{_fmt_time(med_total):>10} "
            f"{_fmt_throughput(med_throughput):>12} "
            f"{slabs_str:>6}",
        )

    print(sep)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

ALL_STRATEGIES = [
    "eager",
    "dask-graph",
    "eager-nocf",
    "eager-sync",
    "slab-prefetch-1",
    "slab-prefetch-2",
    "slab-batch",
]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Standalone IO performance benchmark for xpublish-opendap",
    )
    parser.add_argument(
        "--dataset",
        choices=["ifs", "air"],
        default="air",
        help="Dataset to benchmark (default: air)",
    )
    parser.add_argument(
        "-n",
        "--iterations",
        type=int,
        default=3,
        help="Iterations per strategy (default: 3)",
    )
    parser.add_argument(
        "--scenario",
        help='Run only this scenario (e.g. "10-timesteps")',
    )
    parser.add_argument(
        "--strategy",
        help="Comma-separated strategies (default: all)",
    )
    parser.add_argument(
        "--encoding",
        choices=["xdr", "dap4"],
        default="xdr",
        help="Encoding format (default: xdr)",
    )
    parser.add_argument(
        "--var",
        help="Variable name to benchmark (default: auto-detect)",
    )
    parser.add_argument(
        "--batch-sizes",
        default="5",
        help="Comma-separated batch sizes for slab-batch strategy (default: 5)",
    )
    parser.add_argument(
        "--dask-workers",
        type=int,
        default=4,
        help="Dask thread count for chunk loading (default: 4)",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=None,
        help="Warmup rounds before timed iterations (default: 1, or 0 with --isolate)",
    )
    parser.add_argument(
        "--isolate",
        action="store_true",
        help="Run each trial in a subprocess for cache isolation",
    )
    # Hidden arguments for subprocess trial mode
    parser.add_argument("--_trial", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--_isel", type=str, help=argparse.SUPPRESS)
    return parser.parse_args(argv)


async def async_main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    # Handle subprocess trial mode
    if args._trial:
        await _trial_subprocess_main(args)
        return

    protocol = "dap4" if args.encoding == "dap4" else "dap2"

    # Resolve warmup default: 0 for --isolate, 1 otherwise
    warmup = args.warmup if args.warmup is not None else (0 if args.isolate else 1)

    # Load dataset
    ds, dataset_id = load_dataset(args.dataset)

    # Pick variable
    var_name = args.var
    if var_name is None:
        var_name = _find_data_var(ds)
        if var_name is None:
            var_name = str(next(iter(ds.data_vars)))

    da_full = ds[var_name]
    print(f"{dataset_id} variable: {var_name}  shape: {da_full.shape}")
    if hasattr(da_full.data, "chunks") and da_full.data.chunks is not None:
        print(f"  chunks: {da_full.data.chunks}")
    else:
        print("  chunks: none (in-memory)")
    mode_str = "isolated" if args.isolate else "interleaved"
    print(
        f"  encoding: {args.encoding}  |  iterations: {args.iterations}  |  warmup: {warmup}  |  mode: {mode_str}"
    )

    # Build scenarios
    scenarios = build_scenarios(ds, var_name)
    if args.scenario:
        scenarios = [s for s in scenarios if s.name == args.scenario]
        if not scenarios:
            print(f'Error: no scenario named "{args.scenario}"')
            sys.exit(1)

    # Parse strategies
    batch_sizes = [int(x) for x in args.batch_sizes.split(",")]
    if args.strategy:
        strategies = [s.strip() for s in args.strategy.split(",")]
    else:
        strategies = list(ALL_STRATEGIES)

    # Expand slab-batch into per-batch-size entries
    expanded: list[tuple[str, int]] = []
    for s in strategies:
        if s == "slab-batch":
            for bs in batch_sizes:
                expanded.append(("slab-batch", bs))
        else:
            expanded.append((s, 1))

    for scenario in scenarios:
        da = ds[var_name].isel(scenario.isel)

        chunks_str = ""
        if hasattr(da.data, "chunks") and da.data.chunks is not None:
            chunks_str = f"  chunks: {da.data.chunks}"
        else:
            chunks_str = "  chunks: none (in-memory)"

        print()
        print(f"--- {scenario.name}: {scenario.description} ---")
        print(f"  shape: {da.shape}{chunks_str}")

        slab_info = get_slab_boundaries(da, threshold_bytes=0)
        if slab_info is None and any(s != "eager" for s, _ in expanded):
            print("  (no dask chunks — slab strategies will fall back to eager)")

        print()

        isel_json = _serialize_isel(scenario.isel)
        all_results: list[TrialResult] = []

        # Phases: warmup rounds, then timed rounds — each round iterates all strategies
        phases: list[tuple[str, int, bool]] = []
        if warmup > 0:
            phases.append(("Warmup", warmup, True))
        phases.append(("Round", args.iterations, False))

        if args.isolate:
            for phase_label, n_rounds, is_warmup in phases:
                for r in range(n_rounds):
                    round_results: list[TrialResult] = []
                    for strategy_name, batch_size in expanded:
                        trial = _run_isolated_trial(
                            dataset=args.dataset,
                            var_name=var_name,
                            isel_json=isel_json,
                            strategy=strategy_name,
                            encoding=args.encoding,
                            batch_size=batch_size,
                            dask_workers=args.dask_workers,
                        )
                        trial.scenario = scenario.name
                        trial.iteration = r
                        trial.is_warmup = is_warmup
                        round_results.append(trial)
                    _print_round_detail(round_results, f"{phase_label} {r + 1}")
                    all_results.extend(round_results)
        else:
            for phase_label, n_rounds, is_warmup in phases:
                for r in range(n_rounds):
                    round_results = []
                    for strategy_name, batch_size in expanded:
                        trial = await run_trial(
                            da,
                            strategy_name,
                            protocol,
                            args.dask_workers,
                            batch_size=batch_size,
                        )
                        trial.scenario = scenario.name
                        trial.iteration = r
                        trial.is_warmup = is_warmup
                        round_results.append(trial)
                    _print_round_detail(round_results, f"{phase_label} {r + 1}")
                    all_results.extend(round_results)

        timed = [r for r in all_results if not r.is_warmup]
        print()
        print_summary(timed)


def main(argv: list[str] | None = None) -> None:
    asyncio.run(async_main(argv))


if __name__ == "__main__":
    main()
