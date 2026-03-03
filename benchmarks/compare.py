#!/usr/bin/env python
"""Compare multiple xpublish benchmark runs.

Reads two or more JSON result files and produces a side-by-side comparison
table showing latency values and % change relative to the first file (baseline).

Usage:
    python benchmarks/compare.py baseline.json feature.json
    python benchmarks/compare.py run1.json run2.json run3.json --metric p95_latency_s
    python benchmarks/compare.py run1.json run2.json --format csv > comparison.csv
    python benchmarks/compare.py run1.json run2.json --format json > comparison.json
    python benchmarks/compare.py benchmarks/results/*.json --sort delta
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure the project root is on sys.path so `benchmarks` is importable
# when run as `python benchmarks/compare.py` from the project root.
_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from benchmarks.results import STAT_KEYS, BenchmarkRun, compare_runs, load_run


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Compare multiple xpublish benchmark runs',
    )
    parser.add_argument(
        'files',
        nargs='+',
        help='JSON result files to compare (first is treated as baseline)',
    )
    parser.add_argument(
        '--metric',
        default='mean_latency_s',
        choices=STAT_KEYS,
        help='Statistic to compare (default: mean_latency_s)',
    )
    parser.add_argument(
        '--format',
        dest='output_format',
        default='table',
        choices=['table', 'csv', 'json'],
        help='Output format (default: table)',
    )
    parser.add_argument(
        '--scenarios',
        help='Filter scenarios by name prefix',
    )
    parser.add_argument(
        '--sort',
        choices=['name', 'delta', 'value'],
        default='name',
        help='Sort scenarios: name (default), delta (%% change of last run vs baseline), '
             'value (metric value in last run)',
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    if len(args.files) < 2:
        print('Need at least 2 result files to compare.', file=sys.stderr)
        sys.exit(1)

    # Load runs
    runs: list[BenchmarkRun] = []
    for f in args.files:
        path = Path(f)
        if not path.exists():
            print(f'File not found: {path}', file=sys.stderr)
            sys.exit(1)
        runs.append(load_run(path))

    # Filter scenarios if requested
    if args.scenarios:
        prefix = args.scenarios.lower()
        for run in runs:
            to_remove = [
                name for name in run.scenarios
                if not name.lower().startswith(prefix)
            ]
            for name in to_remove:
                del run.scenarios[name]

    # Sort scenarios if requested (applied by reordering the baseline's scenario dict)
    if args.sort != 'name':
        baseline = runs[0]
        last_run = runs[-1]
        all_names = list(baseline.scenarios.keys())

        if args.sort == 'delta':
            def delta_key(name: str) -> float:
                b = baseline.stat(name, args.metric)
                c = last_run.stat(name, args.metric)
                if b is None or c is None or b == 0:
                    return 0.0
                return ((c - b) / b) * 100

            all_names.sort(key=delta_key)
        elif args.sort == 'value':
            def value_key(name: str) -> float:
                v = last_run.stat(name, args.metric)
                return v if v is not None else 0.0

            all_names.sort(key=value_key, reverse=True)

        # Reorder all runs to match
        for run in runs:
            reordered = {}
            for name in all_names:
                if name in run.scenarios:
                    reordered[name] = run.scenarios[name]
            # Append any scenarios only in this run
            for name in run.scenarios:
                if name not in reordered:
                    reordered[name] = run.scenarios[name]
            run.scenarios = reordered

    output = compare_runs(runs, metric=args.metric, output_format=args.output_format)
    print(output)


if __name__ == '__main__':
    main()
