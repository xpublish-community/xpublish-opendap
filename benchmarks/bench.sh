#!/usr/bin/env bash
# bench.sh — Run xpublish-opendap benchmarks against the PyPI release and
#             the feature/dap4 branch, then compare results.
#
# Uses uv to create isolated temporary venvs. No conda/mamba required.
#
# Usage:
#   ./benchmarks/bench.sh                     # full suite (IFS + air)
#   ./benchmarks/bench.sh --dataset air       # air only (no Arraylake needed)
#   ./benchmarks/bench.sh -n 5 --warmup 1     # fewer iterations
#
# Any extra arguments are forwarded to benchmarks/run.py for both runs.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
RESULTS_DIR="$SCRIPT_DIR/results"
mkdir -p "$RESULTS_DIR"

# Shared dependencies needed by the benchmark runner
COMMON_DEPS=(
    "xpublish>=0.4.2"
    "httpx"
    "uvicorn"
    "xarray"
    "pooch"          # xr.tutorial.open_dataset
    "netcdf4"        # xr.tutorial backend
    "arraylake"      # --dataset ifs
    "setuptools<81"  # pkg_resources for opendap_protocol (removed in 81+)
)

# Ports for the two runs (avoid collisions if a daemon lingers)
PORT_RELEASE=9876
PORT_DAP4=9877

# ──────────────────────────────────────────────────────────────────────
# Parse flags
# ──────────────────────────────────────────────────────────────────────
DATASETS=()
FORWARD_ARGS=("$@")

# Pull --dataset out of FORWARD_ARGS so we control it per-run
for i in "${!FORWARD_ARGS[@]}"; do
    if [[ "${FORWARD_ARGS[$i]}" == "--dataset" ]]; then
        next=$((i + 1))
        if [[ $next -lt ${#FORWARD_ARGS[@]} ]]; then
            DATASETS=("${FORWARD_ARGS[$next]}")
            unset 'FORWARD_ARGS[$i]'
            unset "FORWARD_ARGS[$next]"
            FORWARD_ARGS=("${FORWARD_ARGS[@]+"${FORWARD_ARGS[@]}"}")
        fi
        break
    fi
done

# Default: both datasets
if [[ ${#DATASETS[@]} -eq 0 ]]; then
    DATASETS=("air" "ifs")
fi

# ──────────────────────────────────────────────────────────────────────
# Helper: run benchmarks inside a uv venv
# ──────────────────────────────────────────────────────────────────────
run_bench() {
    local label="$1"
    local venv_dir="$2"
    local port="$3"
    local output_prefix="$4"
    shift 4
    local extra_deps=("$@")

    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  [$label] Creating isolated environment"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    uv venv "$venv_dir" --python 3.13 --quiet
    # shellcheck disable=SC1091
    source "$venv_dir/bin/activate"

    echo "  Installing dependencies..."
    uv pip install --quiet "${COMMON_DEPS[@]}" "${extra_deps[@]}"

    echo "  Installed xpublish-opendap:"
    uv pip show xpublish-opendap 2>/dev/null | grep -E "^(Version):" | sed 's/^/    /'

    echo ""
    echo "  [$label] Running benchmarks (port $port)..."

    for ds in "${DATASETS[@]}"; do
        local ds_output="${output_prefix}_${ds}.json"
        echo ""
        echo "  --- Dataset: $ds ---"
        python "$PROJECT_DIR/benchmarks/run.py" \
            --dataset "$ds" \
            --port "$port" \
            -o "$ds_output" \
            --no-save \
            ${FORWARD_ARGS[@]+"${FORWARD_ARGS[@]}"} || {
                echo "  [!] $label/$ds benchmark failed (exit $?), continuing..."
            }
    done

    deactivate
}

# ──────────────────────────────────────────────────────────────────────
# Setup
# ──────────────────────────────────────────────────────────────────────
TS="$(date +%Y%m%d-%H%M%S)"
RELEASE_PREFIX="$RESULTS_DIR/${TS}_release"
DAP4_PREFIX="$RESULTS_DIR/${TS}_dap4"

TMPDIR_BASE="$(mktemp -d)"
trap 'rm -rf "$TMPDIR_BASE"' EXIT

echo "xpublish-opendap benchmark suite"
echo "================================"
echo "  Datasets: ${DATASETS[*]}"
echo "  Results:  $RESULTS_DIR/"
echo "  Temp:     $TMPDIR_BASE/"

# ──────────────────────────────────────────────────────────────────────
# Run 1: PyPI release (xpublish-opendap==0.2.0)
# ──────────────────────────────────────────────────────────────────────
run_bench "release" \
    "$TMPDIR_BASE/venv-release" \
    "$PORT_RELEASE" \
    "$RELEASE_PREFIX" \
    "xpublish-opendap==0.2.0"

# ──────────────────────────────────────────────────────────────────────
# Run 2: feature/dap4 branch from GitHub (installed via git+https)
# ──────────────────────────────────────────────────────────────────────
run_bench "dap4" \
    "$TMPDIR_BASE/venv-dap4" \
    "$PORT_DAP4" \
    "$DAP4_PREFIX" \
    "xpublish-opendap @ git+https://github.com/xpublish-community/xpublish-opendap.git@feature/dap4"

# ──────────────────────────────────────────────────────────────────────
# Compare results
# ──────────────────────────────────────────────────────────────────────
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Comparison"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

for ds in "${DATASETS[@]}"; do
    release_file="${RELEASE_PREFIX}_${ds}.json"
    dap4_file="${DAP4_PREFIX}_${ds}.json"

    if [[ -f "$release_file" && -f "$dap4_file" ]]; then
        echo ""
        echo "  === $ds dataset ==="
        python "$PROJECT_DIR/benchmarks/compare.py" "$release_file" "$dap4_file"
    else
        echo "  Skipping $ds comparison (missing result files)"
    fi
done

echo ""
echo "Result files:"
for f in "$RESULTS_DIR/${TS}_"*.json; do
    [[ -f "$f" ]] && echo "  $f"
done
echo ""
echo "Done."
