<!-- ProjectCadence: repo-specific instructions for AI coding agents -->
# ProjectCadence — Copilot Instructions (concise)

This file gives focused, discoverable guidance so an AI coding agent can be productive in this repository immediately.

- Big picture
  - Purpose: weekly, self-contained research experiments. Each week lives under `weeks/` (example: `weeks/W02_20250811_dsa-benchmarks`).
  - Structure: each week is a small project scaffolded with `tooling/cadence.py` and contains `src/`, `benchmarks/`, `scripts/`, `data/`, `docs/`, `notebooks/`.
  - Invariants: keep environment fixed when benchmarking; data/results contain CSV rows where each row = one trial (see `weeks/*/README.md`).

- Key files & commands
  - Activate local env: `source ./activate` (top-level README quick-start).
  - Week tooling: `python3 tooling/cadence.py new <n> <slug>` to scaffold, `... info <n>` and `... current --set <n>` to manage `weeks/current`.
  - C++ benchmarks (week examples): build with `cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j` from the week folder containing `CMakeLists.txt` (see `weeks/W02_20250811_dsa-benchmarks/README.md`).
  - Run bench: `./build/bench --sizes ... --trials ... --dist uniform > data/results/bench_uniform.csv` or the helper `bash scripts/run_bench.sh`.
  - Analysis: `weeks/current/scripts/analyze.py` reads `data/results/*.csv` and writes `summary_stats.csv` and `speedup_table.csv`.

- Project-specific conventions
  - Week folder naming: `W<nn>_YYYYMMDD_<slug>` produced by `cadence.py` (see `tooling/cadence.py:slugify`, `week_bounds`).
  - Git commit prefixes: use `feat(Wnn): ...` for week work (documented in top-level README).
  - Results: bench outputs are CSV rows per trial; the repo emphasizes distributions (median, p10/p90) not single runs.
  - Environment control: CPU pinning (`taskset`), local NUMA (`numactl`), and `perf` usage are expected for meaningful results (examples in week README).

- Common code patterns to follow / examples
  - Small, readable C++ data-structure implementations in `weeks/W02_*/src/` (e.g., `vector.hpp`, `hash.hpp`) — prefer keeping changes explicit and minimal when modifying benchmarks.
  - Plotting & analysis live in `weeks/*/scripts/` (see `plot.py`, `analyze.py`) — preserve CSV schema when adding fields.
  - Tooling scripts are deliberately simple and shell/python-based (see `tooling/new-week.sh`, `tooling/cadence.py`) — avoid heavy new dependencies unless necessary.

- How to make safe edits
  - Small focused PRs: change one week at a time. Keep `data/` out of PRs unless necessary; prefer generated outputs in CI artifacts.
  - When modifying benchmarks, maintain the CSV schema columns: `ds,impl,workload,N,dist,params,trial,seed,ns,checksum`.
  - Preserve measurement controls: if you add runs, include a note in the week README about pinning/flags used (taskset/numactl/perf).

- Integration & dependencies
  - Native builds: CMake + system toolchain (g++/clang). No heavy package manager required for the C++ parts.
  - Python: used for tooling/analysis (`tooling/cadence.py`, `scripts/plot.py`, `scripts/analyze.py`) — assume Python 3.8+.

- When in doubt (agent guidance)
  - Read the target week's `README.md` first — it contains the build/run recipe and experiment invariants.
  - Match existing naming and CSV fields exactly when adding baselines (e.g., `impl=stl` vs `impl=custom`).
  - Do not commit large binary blobs or long-running data files into `weeks/*/data/`; prefer adding a small sample or instructions to reproduce.

If anything in this file is unclear or you want more examples (CI recipes, preferred PR size, or test harnesses), ask and I'll expand.
