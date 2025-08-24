#!/usr/bin/env bash
set -euo pipefail

# Example run script for WSL/Linux. Requires: cmake, g++, perf (optional).
# Pin CPU & local NUMA if available for stability.
PIN=${PIN:-"taskset -c 0"}
NUMA=${NUMA:-"numactl --localalloc"}
SIZES=${SIZES:-"1024,4096,16384,65536,262144,1048576"}
TRIALS=${TRIALS:-"20"}
DIST=${DIST:-"uniform"}

build_dir="${BUILD_DIR:-build}"
mkdir -p "$build_dir"
cmake -S . -B "$build_dir" -DCMAKE_BUILD_TYPE=Release
cmake --build "$build_dir" -j

out="data/results/bench_${DIST}.csv"
mkdir -p data/results

echo "Running bench → $out"
$PIN $NUMA "$build_dir/bench" --sizes "$SIZES" --trials "$TRIALS" --dist "$DIST" > "$out"

# Optional: perf counters (example). Produces a parallel CSV with perf summaries.
if command -v perf >/dev/null 2>&1; then
  perf_out="${out%.csv}_perf.txt"
  echo "Running perf (summary) → $perf_out"
  $PIN $NUMA perf stat -x, -e cycles,instructions,cache-misses,branches,branch-misses \
    "$build_dir/bench" --sizes "$SIZES" --trials 5 --dist "$DIST" 1> /dev/null 2> "$perf_out" || true
fi

echo "Done."
