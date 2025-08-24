# Week 02 — Data Structures & Algorithms Benchmarks

**Week** 02 – 2025-08-11 → 2025-08-17

Benchmarks are experiments. Control the environment, define workloads, and report **distributions**—not single runs.

> **One-line invariant:** Fix the environment, fix the workload, then measure multiple signals.

---

## 0) Project Map

```sh
.
├── CMakeLists.txt
├── benchmarks/
│   ├── main.cpp            # emits CSV rows
│   ├── datasets.hpp        # uniform & zipf generators
│   └── workloads.hpp       # per-DS workloads
├── src/
│   ├── arena.hpp           # optional monotonic arena
│   ├── vector.hpp          # DynArray<T>
│   ├── list.hpp            # SinglyList<T>
│   ├── heap.hpp            # BinaryHeap<T> (peek workload wired)
│   └── hash.hpp            # HashMap (open addressing)
├── scripts/
│   ├── run\_bench.sh        # build + run + (optional) perf summary
│   └── plot.py             # quick matplotlib plots
├── data/
│   └── results/            # CSV outputs land here
└── README.md
```

---

## 1) Goals

- Implement small, readable DS variants from scratch to expose real bottlenecks (allocation, pointer chasing, cache misses, probe sequences).
- Compare their steady-state behavior under **defined** workloads and key distributions.
- Explain results using **more than wall-time**: use counters (cycles, instructions, cache-misses, branches) to tell the *why*.

---

## 2) What’s Implemented (scaffold)

**Data structures (custom)**

- `DynArray<T>` — growth policy (default ×2), optional arena-backed reallocation.
- `SinglyList<T>` — pointer-chasing baseline.
- `BinaryHeap<T>` — array-backed max-heap (push + top wired in workload; `pop()` path is left simple to keep the scaffold small—see “Open TODOs”).
- `HashMap` — open addressing with linear probing, load factor cap 0.7, tombstones and rehash.

**Workloads (first pass)**

- `vector`: bulk append then scan (prevents dead-code elimination via checksumming).
- `list`: push_front N then scan.
- `heap`: push N then **peek** top (pop loop can be enabled once `DynArray::pop_back()` is added).
- `hash`: insert N unique keys, then 50/50 mix of successful and unsuccessful finds.

**Distributions**

- `uniform` 64-bit keys.
- `zipf(s=1.2)` sampler (basic but good enough to show hot-spot behavior).

---

## 3) Build & Run

### Requirements
- **Compiler:** g++ 11+ or clang++ 13+ (C++20). MSVC works; perf is Linux/WSL only.
- **Tools:** cmake, python3 (optional for plots), perf (optional), taskset/numactl (optional).

### Build
```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
````

### Run (default sizes & trials)

```bash
# Simple run
./build/bench --sizes 1024,4096,16384,65536,262144 --trials 20 --dist uniform > data/results/bench_uniform.csv
```

Or use the helper:

```bash
bash scripts/run_bench.sh      # writes data/results/bench_uniform.csv
```

### Plot (quick-n-dirty)

```bash
python3 scripts/plot.py data/results/bench_uniform.csv
# -> data/results/*.png
```

---

## 4) CSV Schema

Each row is **one trial** of one workload and size.

| Column   | Meaning                                       |
| -------- | --------------------------------------------- |
| ds       | `vector` \| `list` \| `heap` \| `hash`        |
| impl     | `custom` (space for `stl` later)              |
| workload | e.g., `bulk_append+scan`, `insert+mixed_find` |
| N        | problem size                                  |
| dist     | `uniform` \| `zipf`                           |
| params   | knobs noted (e.g., `growth=2.0`, `load<=0.7`) |
| trial    | 0..                                           |
| seed     | RNG seed used                                 |
| ns       | wall time in nanoseconds                      |
| checksum | sink to defeat dead-code elimination          |

---

## 5) Environment Control (write this in your paper/blog too)

* **CPU pinning:** `taskset -c 0` (Linux/WSL). Keep other cores idle.
* **NUMA locality:** `numactl --localalloc` to avoid cross-node noise.
* **Governor/Turbo:** Prefer a fixed frequency (`performance` governor). If you can’t, keep runs short and interleave A/B/A/B.
* **Build flags:** Release with `-O3 -march=native -DNDEBUG` (Linux). MSVC: `/O2 /DNDEBUG`.
* **Thermals:** Don’t run 10-minute marathons without cooldown; throttle flips winners.
* **Background noise:** Close browsers, sync tools, and anything scanning files.

---

## 6) Perf Counters (optional but recommended)

Example (summary mode):

```bash
taskset -c 0 numactl --localalloc \
perf stat -x, -e cycles,instructions,cache-misses,branches,branch-misses \
./build/bench --sizes 1024,4096,16384 --trials 5 --dist zipf \
1> /dev/null 2> data/results/bench_zipf_perf.txt
```

Use counters to compute **CPI** (cycles/instruction) and **miss rates**. They explain *why* a curve bends.

---

## 7) Fairness Notes

* **Consume results:** We checksum traversals to stop the compiler from deleting loops.
* **Allocator dominance:** If allocation noise dominates, switch workloads to “build once, then operate,” or use `Arena` for *build* and `new/delete` for *operate*.
* **Apples-to-apples:** When you later add STL baselines, match policies (e.g., `reserve()` behavior for `std::vector`, load factors for hash tables).

---

## 8) Interpreting Curves (what to look for)

* **Vector bulk append:** With growth ×2 you should see \~O(1) amortized; if ns/op spikes periodically, that’s reallocation. Add `reserve(N)` as an ablation.
* **List scan:** ns/op will be flat but *much* higher than vector due to cache misses and pointer chasing.
* **Heap push/peek:** Mostly log-like; hotspots show up if key distribution triggers long sift-up chains.
* **Hash insert/find:** Watch how `zipf` worsens probe clustering; miss rate climbs as load factor approaches 0.7.

---

## 9) Open TODOs (intended exercises)

* Add `DynArray::pop_back()` and wire a proper `BinaryHeap::pop()` loop.
* Expose knobs:

  * Vector growth factor 1.5/2.0 and a variant with `reserve(N)`.
  * Hash load factor targets 0.5/0.7/0.9 and quadratic probing vs linear vs robin-hood.
* Add **STL baselines** (`std::vector`, `std::list`, `std::priority_queue`, `std::unordered_map`) under the same workloads and emit rows with `impl=stl`.
* Optional: integrate Google Benchmark as a second harness to cross-check numbers.

---

## 10) Known Limitations (on purpose)

* This harness prints wall-time only. Perf counters are run separately (simple, transparent).
* Zipf sampler is basic. Good enough to stress caches; you can swap in a sharper sampler if needed.
* No cross-platform pinning helpers; we keep scripts plain for WSL/Linux.

---

## 11) Repro Recipe (pasteable)

```bash
# 1) Build
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j

# 2) Run two distributions
taskset -c 0 numactl --localalloc ./build/bench \
  --sizes 1024,4096,16384,65536,262144,1048576 \
  --trials 20 --dist uniform > data/results/bench_uniform.csv

taskset -c 0 numactl --localalloc ./build/bench \
  --sizes 1024,4096,16384,65536,262144,1048576 \
  --trials 20 --dist zipf > data/results/bench_zipf.csv

# 3) Optional perf summary
taskset -c 0 numactl --localalloc perf stat -x, -e cycles,instructions,cache-misses,branches,branch-misses \
  ./build/bench --sizes 16384,65536,262144 --trials 5 --dist uniform \
  1>/dev/null 2> data/results/bench_uniform_perf.txt

# 4) Quick plots
python3 scripts/plot.py data/results/bench_uniform.csv
```

---

## 12) License / Attribution

This scaffold is intentionally minimal and unopinionated. Use, modify, and extend for your Week-2 DSA benchmarks. Add your name and license of choice in `METADATA.yaml`.
