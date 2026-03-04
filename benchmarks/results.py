"""Benchmark result collection, statistics, and output formatting."""

from __future__ import annotations

import csv
import io
import json
import math
import statistics
import subprocess
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

_NAN = float("nan")


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class RequestResult:
    """Result of a single HTTP request."""

    latency_s: float
    status_code: int
    response_bytes: int
    error: str | None = None

    @property
    def success(self) -> bool:
        return self.status_code == 200 and self.error is None


@dataclass
class ScenarioResult:
    """Aggregated results for one scenario."""

    name: str
    category: str
    description: str
    path: str
    requests: list[RequestResult] = field(default_factory=list)

    @property
    def successes(self) -> list[RequestResult]:
        return [r for r in self.requests if r.success]

    @property
    def success_rate(self) -> float:
        if not self.requests:
            return 0.0
        return len(self.successes) / len(self.requests)

    @property
    def latencies(self) -> list[float]:
        return [r.latency_s for r in self.successes]

    @property
    def mean_latency(self) -> float:
        lats = self.latencies
        return statistics.mean(lats) if lats else _NAN

    @property
    def median_latency(self) -> float:
        lats = self.latencies
        return statistics.median(lats) if lats else _NAN

    @property
    def p95_latency(self) -> float:
        lats = sorted(self.latencies)
        if not lats:
            return _NAN
        idx = int(len(lats) * 0.95)
        return lats[min(idx, len(lats) - 1)]

    @property
    def min_latency(self) -> float:
        lats = self.latencies
        return min(lats) if lats else _NAN

    @property
    def max_latency(self) -> float:
        lats = self.latencies
        return max(lats) if lats else _NAN

    @property
    def total_bytes(self) -> int:
        return sum(r.response_bytes for r in self.successes)

    @property
    def throughput_mbps(self) -> float:
        """Throughput in MB/s based on total bytes and total latency."""
        lats = self.latencies
        if not lats:
            return _NAN
        total_time = sum(lats)
        if total_time == 0:
            return _NAN
        return (self.total_bytes / (1024 * 1024)) / total_time

    def stats_dict(self) -> dict:
        """Return a flat dictionary of computed statistics."""
        return {
            "mean_latency_s": self.mean_latency,
            "median_latency_s": self.median_latency,
            "p95_latency_s": self.p95_latency,
            "min_latency_s": self.min_latency,
            "max_latency_s": self.max_latency,
            "throughput_mbps": self.throughput_mbps,
            "success_rate": self.success_rate,
            "total_requests": len(self.requests),
            "total_bytes": self.total_bytes,
        }


@dataclass
class BenchmarkRun:
    """A complete benchmark run loaded from JSON."""

    path: Path
    metadata: dict
    scenarios: dict[str, dict]  # name -> stats dict

    @property
    def label(self) -> str:
        """Short human-readable label for this run."""
        branch = self.metadata.get("git_branch", "?")
        sha = self.metadata.get("git_sha", "?")
        return f"{branch}@{sha}"

    @property
    def timestamp(self) -> str:
        return self.metadata.get("timestamp", "unknown")

    def stat(self, scenario_name: str, metric: str) -> float | None:
        """Get a single stat value, or None if the scenario is missing."""
        entry = self.scenarios.get(scenario_name)
        if entry is None:
            return None
        return entry.get(metric)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _git_info() -> dict[str, str]:
    """Get current git SHA and branch."""
    info: dict[str, str] = {}
    try:
        info["sha"] = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        info["sha"] = "unknown"
    try:
        info["branch"] = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        info["branch"] = "unknown"
    return info


def _xpublish_version() -> str:
    """Get installed xpublish version."""
    try:
        import xpublish

        return getattr(xpublish, "__version__", "unknown")
    except ImportError:
        return "unknown"


STAT_KEYS = [
    "mean_latency_s",
    "median_latency_s",
    "p95_latency_s",
    "min_latency_s",
    "max_latency_s",
    "throughput_mbps",
    "success_rate",
    "total_requests",
    "total_bytes",
]


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def load_run(path: str | Path) -> BenchmarkRun:
    """Load a benchmark run from a JSON file."""
    path = Path(path)
    data = json.loads(path.read_text())
    scenarios: dict[str, dict] = {}
    for entry in data.get("results", []):
        scenarios[entry["name"]] = entry.get("stats", {})
    return BenchmarkRun(
        path=path,
        metadata=data.get("metadata", {}),
        scenarios=scenarios,
    )


# ---------------------------------------------------------------------------
# Console table
# ---------------------------------------------------------------------------


def print_table(results: list[ScenarioResult]) -> None:
    """Print a grouped console table of benchmark results."""
    name_w = max(len(r.name) for r in results)
    name_w = max(name_w, 8)

    header = (
        f"{'Scenario':<{name_w}}  "
        f"{'Mean':>8}  {'Median':>8}  {'P95':>8}  "
        f"{'Min':>8}  {'Max':>8}  "
        f"{'MB/s':>8}  {'OK%':>5}  {'N':>3}"
    )
    separator = "-" * len(header)

    print()
    current_category = None
    for r in results:
        if r.category != current_category:
            current_category = r.category
            print(separator)
            print(f"  {current_category}")
            print(separator)
            print(header)
            print(separator)

        ok_pct = f"{r.success_rate * 100:.0f}%"
        if math.isnan(r.mean_latency):
            print(
                f"{r.name:<{name_w}}  "
                f"{'N/A':>8}  {'N/A':>8}  {'N/A':>8}  "
                f"{'N/A':>8}  {'N/A':>8}  "
                f"{'N/A':>8}  "
                f"{ok_pct:>5}  "
                f"{len(r.requests):>3}",
            )
        else:
            print(
                f"{r.name:<{name_w}}  "
                f"{r.mean_latency * 1000:>7.1f}ms  "
                f"{r.median_latency * 1000:>7.1f}ms  "
                f"{r.p95_latency * 1000:>7.1f}ms  "
                f"{r.min_latency * 1000:>7.1f}ms  "
                f"{r.max_latency * 1000:>7.1f}ms  "
                f"{r.throughput_mbps:>8.2f}  "
                f"{ok_pct:>5}  "
                f"{len(r.requests):>3}",
            )
    print(separator)
    print()


# ---------------------------------------------------------------------------
# JSON output
# ---------------------------------------------------------------------------


def _build_json_data(
    results: list[ScenarioResult],
    dataset_info: dict | None = None,
) -> dict:
    """Build the full JSON-serializable dict for a benchmark run."""
    git = _git_info()
    data: dict = {
        "metadata": {
            "timestamp": datetime.now(UTC).isoformat(),
            "git_sha": git["sha"],
            "git_branch": git["branch"],
            "xpublish_version": _xpublish_version(),
            "dataset": dataset_info or {},
        },
        "results": [],
    }

    for r in results:
        stats = {
            k: (None if isinstance(v, float) and math.isnan(v) else v)
            for k, v in r.stats_dict().items()
        }
        entry = {
            "name": r.name,
            "category": r.category,
            "description": r.description,
            "path": r.path,
            "stats": stats,
            "requests": [
                {
                    "latency_s": req.latency_s,
                    "status_code": req.status_code,
                    "response_bytes": req.response_bytes,
                    "error": req.error,
                }
                for req in r.requests
            ],
        }
        data["results"].append(entry)

    return data


def write_json(
    results: list[ScenarioResult],
    path: str | Path,
    dataset_info: dict | None = None,
) -> None:
    """Write full results + run metadata to a JSON file."""
    data = _build_json_data(results, dataset_info)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n")
    print(f"Results written to {path}")


# ---------------------------------------------------------------------------
# CSV output
# ---------------------------------------------------------------------------


def write_csv(
    results: list[ScenarioResult],
    path: str | Path,
) -> None:
    """Write per-scenario summary stats to a CSV file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = ["name", "category"] + STAT_KEYS
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            row: dict[str, object] = {"name": r.name, "category": r.category}
            row.update(
                {
                    k: ("" if isinstance(v, float) and math.isnan(v) else v)
                    for k, v in r.stats_dict().items()
                },
            )
            writer.writerow(row)

    print(f"CSV written to {path}")


# ---------------------------------------------------------------------------
# Two-run comparison (used by run.py --compare)
# ---------------------------------------------------------------------------


def print_comparison(results: list[ScenarioResult], baseline_path: str | Path) -> None:
    """Load baseline JSON and print a comparison table with % delta."""
    baseline = load_run(baseline_path)
    if not baseline.scenarios:
        print(f"No results in baseline file: {baseline_path}")
        return

    print()
    print("Comparing against baseline:")
    print(f"  SHA: {baseline.metadata.get('git_sha', 'unknown')}")
    print(f"  Branch: {baseline.metadata.get('git_branch', 'unknown')}")
    print(f"  Timestamp: {baseline.timestamp}")

    name_w = max(len(r.name) for r in results)
    name_w = max(name_w, 8)

    header = (
        f"{'Scenario':<{name_w}}  "
        f"{'Current':>10}  {'Baseline':>10}  {'Delta':>8}  {'Change':>8}"
    )
    separator = "-" * len(header)

    print()
    print(header)
    print(separator)

    for r in results:
        b_mean = baseline.stat(r.name, "mean_latency_s")
        c_mean = r.mean_latency

        if b_mean is None:
            print(
                f"{r.name:<{name_w}}  "
                f"{c_mean * 1000:>9.1f}ms  "
                f"{'N/A':>10}  {'':>8}  {'new':>8}",
            )
            continue

        c_str = _fmt_ms(c_mean)
        b_str = _fmt_ms(b_mean)

        if not _isnan(b_mean) and not _isnan(c_mean) and b_mean > 0:
            delta_pct = ((c_mean - b_mean) / b_mean) * 100
            delta_str = f"{delta_pct:+.1f}%"
            if delta_pct < -5:
                change = "faster"
            elif delta_pct > 5:
                change = "slower"
            else:
                change = "~same"
        else:
            delta_str = ""
            change = ""

        print(
            f"{r.name:<{name_w}}  "
            f"{c_str:>10}  "
            f"{b_str:>10}  "
            f"{delta_str:>8}  "
            f"{change:>8}",
        )

    print(separator)
    print()


# ---------------------------------------------------------------------------
# Multi-run comparison (used by compare.py)
# ---------------------------------------------------------------------------


def _fmt_size(avg_bytes: float) -> str:
    """Format an average response size as a human-readable string."""
    if avg_bytes >= 1024 * 1024:
        return f"{avg_bytes / (1024 * 1024):.1f}MB"
    if avg_bytes >= 1024:
        return f"{avg_bytes / 1024:.1f}KB"
    return f"{avg_bytes:.0f}B"


def _fmt_ms(val: float | None) -> str:
    """Format a latency value in seconds as milliseconds string."""
    if val is None or (isinstance(val, float) and math.isnan(val)):
        return "N/A"
    return f"{val * 1000:.1f}ms"


def _isnan(v: float | None) -> bool:
    return v is None or (isinstance(v, float) and math.isnan(v))


def _fmt_delta(current: float | None, baseline: float | None) -> tuple[str, str]:
    """Return (delta_str, change_label) comparing current vs baseline."""
    if _isnan(current) or _isnan(baseline) or baseline == 0:
        return ("", "")
    delta_pct = ((current - baseline) / baseline) * 100
    delta_str = f"{delta_pct:+.1f}%"
    if delta_pct < -5:
        return (delta_str, "faster")
    elif delta_pct > 5:
        return (delta_str, "slower")
    return (delta_str, "~same")


def _unique_labels(runs: list[BenchmarkRun]) -> list[str]:
    """Build unique display labels for a list of runs.

    Uses the short label (branch@sha) when unique, otherwise appends
    the filename stem to disambiguate.
    """
    raw = [r.label for r in runs]
    counts: dict[str, int] = {}
    for label in raw:
        counts[label] = counts.get(label, 0) + 1

    labels: list[str] = []
    for run, label in zip(runs, raw):
        if counts[label] > 1:
            labels.append(f"{label} ({run.path.stem})")
        else:
            labels.append(label)
    return labels


def compare_runs(
    runs: list[BenchmarkRun],
    metric: str = "mean_latency_s",
    output_format: str = "table",
) -> str:
    """Compare multiple benchmark runs and return formatted output.

    Args:
        runs: List of BenchmarkRun objects (first is treated as baseline).
        metric: Stat key to compare (default: mean_latency_s).
        output_format: One of 'table', 'csv', 'json'.

    Returns:
        Formatted string (printed by caller).
    """
    if not runs:
        return "No runs to compare."

    # Collect all scenario names in order (union across runs, stable order from first run)
    seen: dict[str, None] = {}
    for run in runs:
        for name in run.scenarios:
            if name not in seen:
                seen[name] = None
    all_scenarios = list(seen)

    baseline = runs[0]
    labels = _unique_labels(runs)

    if output_format == "json":
        return _compare_json(runs, all_scenarios, metric, baseline, labels)
    elif output_format == "csv":
        return _compare_csv(runs, all_scenarios, metric, baseline, labels)
    else:
        return _compare_table(runs, all_scenarios, metric, baseline, labels)


def _compare_table(
    runs: list[BenchmarkRun],
    scenarios: list[str],
    metric: str,
    baseline: BenchmarkRun,
    labels: list[str],
) -> str:
    """Render a multi-run comparison as a console table."""
    lines: list[str] = []

    # Header: run metadata
    lines.append("")
    lines.append(f"Metric: {metric}")
    lines.append(f"Baseline: {labels[0]} ({baseline.path.name})")
    lines.append("")
    for i, (run, label) in enumerate(zip(runs, labels)):
        tag = " (baseline)" if i == 0 else ""
        lines.append(f"  [{i + 1}] {label}{tag}  {run.timestamp}")
    lines.append("")

    # Column layout
    name_w = max((len(s) for s in scenarios), default=8)
    name_w = max(name_w, 8)

    is_latency = "latency" in metric
    fmt_val = _fmt_ms if is_latency else (lambda v: "N/A" if v is None else f"{v:.2f}")

    # Build header row
    col_w = 10
    parts = [f"{'Scenario':<{name_w}}"]
    for i, run in enumerate(runs):
        tag = f"[{i + 1}]"
        parts.append(f"{tag:>{col_w}}")
        if i > 0:
            parts.append(f"{'delta':>{col_w}}")
    header = "  ".join(parts)
    sep = "-" * len(header)

    lines.append(header)
    lines.append(sep)

    for scenario_name in scenarios:
        parts = [f"{scenario_name:<{name_w}}"]
        b_val = baseline.stat(scenario_name, metric)

        for i, run in enumerate(runs):
            val = run.stat(scenario_name, metric)
            parts.append(f"{fmt_val(val):>{col_w}}")
            if i > 0:
                delta_str, change = _fmt_delta(val, b_val)
                label = f"{delta_str} {change}".strip() if delta_str else ""
                parts.append(f"{label:>{col_w}}")

        # Append response size for each run
        size_parts = []
        for run in runs:
            total_bytes = run.stat(scenario_name, "total_bytes")
            n_requests = run.stat(scenario_name, "total_requests")
            if total_bytes is not None and n_requests and n_requests > 0:
                ok_rate = run.stat(scenario_name, "success_rate") or 0
                n_ok = int(n_requests * ok_rate)
                avg = total_bytes / max(n_ok, 1) if n_ok > 0 else 0
                size_parts.append(_fmt_size(avg))
            else:
                size_parts.append("N/A")
        parts.append(f"  [{'/'.join(size_parts)}]")

        lines.append("  ".join(parts))

    lines.append(sep)
    lines.append("")
    return "\n".join(lines)


def _compare_csv(
    runs: list[BenchmarkRun],
    scenarios: list[str],
    metric: str,
    baseline: BenchmarkRun,
    labels: list[str],
) -> str:
    """Render a multi-run comparison as CSV."""
    buf = io.StringIO()
    fieldnames = ["scenario"]
    for i, label in enumerate(labels):
        fieldnames.append(label)
        if i > 0:
            fieldnames.append(f"{label}_delta_pct")

    writer = csv.DictWriter(buf, fieldnames=fieldnames)
    writer.writeheader()

    for scenario_name in scenarios:
        row: dict[str, str] = {"scenario": scenario_name}
        b_val = baseline.stat(scenario_name, metric)

        for i, (run, label) in enumerate(zip(runs, labels)):
            val = run.stat(scenario_name, metric)
            row[label] = "" if _isnan(val) else f"{val}"
            if i > 0:
                if not _isnan(val) and not _isnan(b_val) and b_val > 0:
                    delta = ((val - b_val) / b_val) * 100
                    row[f"{label}_delta_pct"] = f"{delta:.1f}"
                else:
                    row[f"{label}_delta_pct"] = ""

        writer.writerow(row)

    return buf.getvalue()


def _compare_json(
    runs: list[BenchmarkRun],
    scenarios: list[str],
    metric: str,
    baseline: BenchmarkRun,
    labels: list[str],
) -> str:
    """Render a multi-run comparison as JSON."""
    data: dict = {
        "metric": metric,
        "runs": [],
        "comparisons": [],
    }

    for i, (run, label) in enumerate(zip(runs, labels)):
        data["runs"].append(
            {
                "index": i + 1,
                "label": label,
                "file": str(run.path),
                "timestamp": run.timestamp,
                "is_baseline": i == 0,
                "metadata": run.metadata,
            },
        )

    for scenario_name in scenarios:
        entry: dict = {"scenario": scenario_name, "values": {}}
        b_val = baseline.stat(scenario_name, metric)

        for i, (run, label) in enumerate(zip(runs, labels)):
            val = run.stat(scenario_name, metric)
            run_entry: dict = {"value": None if _isnan(val) else val}
            if i > 0 and not _isnan(val) and not _isnan(b_val) and b_val > 0:
                run_entry["delta_pct"] = round(((val - b_val) / b_val) * 100, 2)
            entry["values"][label] = run_entry

        data["comparisons"].append(entry)

    return json.dumps(data, indent=2)
