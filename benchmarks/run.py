#!/usr/bin/env python
"""Xpublish OpenDAP benchmark runner.

Starts an in-process xpublish server and measures request latency
across a suite of DAP2/DAP4 scenarios.

Usage:
    python benchmarks/run.py --dataset air -n 5          # Quick local test
    python benchmarks/run.py -o results.json              # Save JSON
    python benchmarks/run.py -c baseline.json             # Compare vs baseline
    python benchmarks/run.py --scenarios dap4             # Filter scenarios
"""

from __future__ import annotations

import argparse
import math
import sys
import threading
import time
from pathlib import Path

# Ensure the project root is on sys.path so `benchmarks` is importable
# when run as `python benchmarks/run.py` from the project root.
_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import httpx
import uvicorn
import xarray as xr

import xpublish
from xpublish_opendap import OpenDapPlugin

from benchmarks.results import (
    RequestResult,
    ScenarioResult,
    print_comparison,
    print_table,
    write_csv,
    write_json,
)
from benchmarks.scenarios import Scenario, build_scenarios


def load_dataset(name: str) -> tuple[xr.Dataset, str]:
    """Load a dataset by name.

    Args:
        name: Either 'ifs' (IFS Realtime from Arraylake) or 'air' (xarray tutorial).

    Returns:
        Tuple of (dataset, dataset_id).
    """
    if name == 'ifs':
        try:
            from arraylake import Client
        except ImportError:
            print('arraylake is required for --dataset ifs')
            print('Install with: pip install arraylake')
            sys.exit(1)

        print('Loading IFS Realtime dataset from Arraylake...')
        client = Client()
        repo = client.get_repo('earthmover-demos/ifs-demo')
        session = repo.readonly_session(branch='main')
        ds = xr.open_zarr(session.store, zarr_format=3)
        return ds, 'ifs'

    # Default: air_temperature tutorial dataset
    print('Loading air_temperature tutorial dataset...')
    ds = xr.tutorial.open_dataset('air_temperature')
    return ds, 'air'


def start_server(
    ds: xr.Dataset,
    dataset_id: str,
    host: str,
    port: int,
) -> uvicorn.Server:
    """Start xpublish server in a daemon thread.

    Returns the uvicorn.Server instance for shutdown control.
    """
    rest = xpublish.Rest(
        {dataset_id: ds},
        plugins={'opendap': OpenDapPlugin()},
    )
    config = uvicorn.Config(rest.app, host=host, port=port, log_level='warning')
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    return server


def check_port_free(host: str, port: int) -> None:
    """Check that the port is not already in use."""
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind((host, port))
        except OSError:
            print(f'Port {port} is already in use. Use --port to pick another.')
            sys.exit(1)


def wait_for_server(base_url: str, dataset_id: str, timeout: float = 30.0) -> None:
    """Poll the datasets endpoint until the expected dataset is listed."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            resp = httpx.get(f'{base_url}/datasets', timeout=2.0)
            if resp.status_code == 200 and dataset_id in resp.text:
                return
        except httpx.ConnectError:
            pass
        time.sleep(0.5)
    print(f'Server did not start within {timeout}s')
    sys.exit(1)


def run_scenario(
    client: httpx.Client,
    base_url: str,
    scenario: Scenario,
    iterations: int,
    warmup: int,
    timeout: float,
    verbose: bool,
) -> ScenarioResult:
    """Run a single scenario: warmup then timed iterations."""
    url = f'{base_url}{scenario.path}'
    result = ScenarioResult(
        name=scenario.name,
        category=scenario.category,
        description=scenario.description,
        path=scenario.path,
    )

    # Warmup
    for _ in range(warmup):
        try:
            client.get(url, timeout=timeout)
        except Exception:
            pass

    # Timed iterations
    for i in range(iterations):
        error = None
        status_code = 0
        response_bytes = 0

        t0 = time.perf_counter()
        try:
            resp = client.get(url, timeout=timeout)
            status_code = resp.status_code
            response_bytes = len(resp.content)

            # Validate response authenticity via DAP-specific header
            if status_code == 200 and scenario.expected_header:
                if scenario.expected_header not in resp.headers:
                    error = f'missing {scenario.expected_header} header (endpoint not implemented?)'
                    status_code = 0  # Mark as failure
        except httpx.TimeoutException:
            error = 'timeout'
        except Exception as e:
            error = str(e)
        t1 = time.perf_counter()

        req = RequestResult(
            latency_s=t1 - t0,
            status_code=status_code,
            response_bytes=response_bytes,
            error=error,
        )
        result.requests.append(req)

        if verbose and (error or status_code != 200):
            print(f'  [{scenario.name}] iter {i + 1}: status={status_code} error={error}')

    # Log first failure details on first occurrence (even without -v)
    first_fail = next((r for r in result.requests if not r.success), None)
    if first_fail and not verbose:
        print(f' [!] status={first_fail.status_code}', end='')
        if first_fail.error:
            print(f' error={first_fail.error}', end='')

    return result


def dataset_info(ds: xr.Dataset, dataset_id: str) -> dict:
    """Build dataset metadata for JSON output."""
    return {
        'dataset_id': dataset_id,
        'dims': {str(k): int(v) for k, v in ds.sizes.items()},
        'data_vars': list(str(v) for v in ds.data_vars),
        'coords': list(str(c) for c in ds.coords),
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Xpublish OpenDAP benchmark suite',
    )
    parser.add_argument(
        '-n', '--iterations',
        type=int,
        default=10,
        help='Number of timed iterations per scenario (default: 10)',
    )
    parser.add_argument(
        '--warmup',
        type=int,
        default=2,
        help='Number of warmup iterations per scenario (default: 2)',
    )
    parser.add_argument(
        '--port',
        type=int,
        default=9876,
        help='Server port (default: 9876)',
    )
    parser.add_argument(
        '--host',
        default='127.0.0.1',
        help='Server host (default: 127.0.0.1)',
    )
    parser.add_argument(
        '-o', '--output',
        help='JSON output path (default: benchmarks/results/<auto>.json)',
    )
    parser.add_argument(
        '--csv',
        help='Also write summary CSV to this path',
    )
    parser.add_argument(
        '--no-save',
        action='store_true',
        help='Skip writing JSON results to disk',
    )
    parser.add_argument(
        '-c', '--compare',
        help='Compare against baseline JSON file',
    )
    parser.add_argument(
        '--dataset',
        choices=['ifs', 'air'],
        default='ifs',
        help='Dataset to benchmark (default: ifs)',
    )
    parser.add_argument(
        '--scenarios',
        help='Filter scenarios by name prefix (e.g. "dap4", "metadata")',
    )
    parser.add_argument(
        '--timeout',
        type=float,
        default=60.0,
        help='Request timeout in seconds (default: 60)',
    )
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Print per-request errors',
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    # Load dataset
    ds, dataset_id = load_dataset(args.dataset)
    print(f'Dataset: {dataset_id}')
    print(f'  Dims: {dict(ds.sizes)}')
    print(f'  Data vars: {list(ds.data_vars)}')
    print()

    # Start server
    base_url = f'http://{args.host}:{args.port}'
    check_port_free(args.host, args.port)
    print(f'Starting server on {base_url}...')
    server = start_server(ds, dataset_id, args.host, args.port)
    wait_for_server(base_url, dataset_id)
    print('Server ready.')
    print()

    # Build scenarios
    scenarios = build_scenarios(ds, dataset_id)
    if args.scenarios:
        prefix = args.scenarios.lower()
        scenarios = [s for s in scenarios if s.name.lower().startswith(prefix)]
    print(f'Running {len(scenarios)} scenarios, {args.iterations} iterations each '
          f'({args.warmup} warmup)')
    print()

    # Run benchmarks
    results: list[ScenarioResult] = []
    with httpx.Client() as client:
        for i, scenario in enumerate(scenarios, 1):
            print(f'  [{i}/{len(scenarios)}] {scenario.name}...', end='', flush=True)
            result = run_scenario(
                client, base_url, scenario,
                args.iterations, args.warmup, args.timeout, args.verbose,
            )
            results.append(result)
            ok = sum(1 for r in result.requests if r.success)
            avg_bytes = result.total_bytes / max(len(result.successes), 1)
            if avg_bytes >= 1024 * 1024:
                size_str = f'{avg_bytes / (1024 * 1024):.1f}MB'
            elif avg_bytes >= 1024:
                size_str = f'{avg_bytes / 1024:.1f}KB'
            else:
                size_str = f'{avg_bytes:.0f}B'
            if math.isnan(result.mean_latency):
                print(f' N/A, {size_str} ({ok}/{len(result.requests)} ok)')
            else:
                print(f' {result.mean_latency * 1000:.1f}ms avg, {size_str} ({ok}/{len(result.requests)} ok)')

    # Output
    print_table(results)

    info = dataset_info(ds, dataset_id)

    if args.output:
        # Explicit -o always writes
        write_json(results, Path(args.output), dataset_info=info)
    elif not args.no_save:
        # Auto-save to benchmarks/results/ unless --no-save
        results_dir = Path(__file__).resolve().parent / 'results'
        results_dir.mkdir(exist_ok=True)
        ts = time.strftime('%Y%m%d-%H%M%S')
        json_path = results_dir / f'{ts}_{dataset_id}.json'
        write_json(results, json_path, dataset_info=info)

    if args.csv:
        write_csv(results, args.csv)

    if args.compare:
        print_comparison(results, args.compare)

    # Shutdown
    server.should_exit = True
    print('Done.')


if __name__ == '__main__':
    main()
