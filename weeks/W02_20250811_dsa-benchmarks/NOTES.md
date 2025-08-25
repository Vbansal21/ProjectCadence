Here’s my Week-2 journal note.

---

# Week 2 — Data Structures & Algorithms Benchmarks (Lab Journal)

**Intent.** I set out to build a small but honest lab for comparing hand-rolled data structures against their standard-library counterparts. The goal wasn’t “beat STL,” it was to sharpen my performance intuition by running controlled experiments and explaining results with signals beyond wall-time.

**Core invariant I kept pinned:** fix the environment, fix the workload, then measure multiple signals and distributions—not single hero runs.

---

## What I built

**Custom data structures.**

* `DynArray<T>`: minimal dynamic array with a tunable growth factor and optional monotonic `Arena` to separate allocator noise from algorithmic work; added `push_back`, `pop_back`, `back`, `reserve`, clear construction/destruction paths.
* `SinglyList<T>`: intentionally cache-unfriendly pointer-chasing baseline to make locality costs visible.
* `BinaryHeap<T>`: array-backed max-heap on top of `DynArray`; sift-up/down; real `pop()` wired to let me do push-then-pop-all workloads.
* `HashMap`: open addressing with linear probing, tombstones, power-of-two capacity, and a solid 64-bit mixer; rehash triggers at a target load factor (default 0.7).

**Bench harness.**

* Standalone C++ harness that prints **CSV** rows: `{ds, impl, workload, N, dist, params, trial, seed, ns, checksum}`.
* Workloads per DS:

  * vector: bulk append → scan
  * list: push\_front → scan
  * heap: push N → pop all
  * hash: insert N → 50/50 successful+unsuccessful finds
* Input distributions: uniform 64-bit and a basic Zipf(s=1.2) to surface hot-spot/probe clustering effects.
* **STL baselines** for the same workloads: `std::vector` (with/without `reserve`), `std::list`, `std::priority_queue`, `std::unordered_map` (with/without `reserve`).
* Guarded against dead-code elimination with a volatile checksum sink.

**Reproducibility + tooling.**

* Release-mode defaults with `-O3 -march=native -DNDEBUG`.
* Simple `run_bench.sh` for pinned runs (`taskset`/`numactl`) and optional `perf stat` counters.
* Quick plotting (`plot.py`) and an analyzer to emit medians and p10/p90 bands + speedup tables.
* (Planned/ready) Metadata sidecar printing compiler/build info to stderr for each run.

---

## Why I built it this way (origin story in brief)

Every “I beat STL by 3×” claim I’ve seen collapses under a different CPU governor, a missing `reserve`, or a cold cache. Modern CPUs hide dragons—caches, branch predictors, allocators, turbo/thermal drift. The harness design bakes in the lessons people learn the hard way: consume results to prevent DCE, isolate allocator costs, interleave trials, record distributions, and vary data distributions to avoid branch-predictor luck.

---

## What I ran (so far)

* **Smoke runs** with small size sweeps and a handful of trials to validate plumbing. CSVs look consistent; checksums differ across workloads as expected; no obvious signs of DCE (the checksum shifts when I perturb computation).
* Baselines compile and emit rows alongside custom implementations, so plotting relative gaps is straightforward.

I can already see the expected qualitative shapes:

* Vector bulk-append is near flat in ns/op after warm-up, with periodic spikes at growth boundaries (resolved by `reserve`).
* List scan time per element sits well above vector due to poor locality.
* Heap push/pop curves follow the log-like slope; constants differ but the story matches priority\_queue.
* Hash insert+find degrades as load climbs; Zipf makes probe clustering visible.

(Full variance bands + perf counters are queued; see “What’s left to do.”)

---

## What I learned / re-learned

1. **Experiment design > code cleverness.** The hardest part is defining fair workloads and pinning the environment. Code changes are cheap; experimental discipline isn’t.
2. **Locality is the kingmaker.** `std::list` exists for semantic stability, not speed. Watching it lose to contiguous arrays even on inserts is a visceral reminder.
3. **Allocators can dominate the plot.** Pooling or a monotonic arena turns some “speedups” into apples-to-apples comparisons instead of “I measured `new/delete`.”
4. **Distributions matter.** Uniform hides pain; Zipf exposes it. Hash tables that look fine at 0.7 load under uniform stumble when keys cluster.
5. **Distributions (plural) > datapoint (singular).** A median without p10/p90 is a story half-told. Small Ns lie; large Ns saturate memory bandwidth. Crossing L1/L2/LLC is where curves turn and insights live.

---

## Pitfall sweep (and how I addressed each)

* **Dead-code elimination:** checksums consume traversal outputs; results are printed so the compiler can’t throw away work.
* **Warm-up bias:** smoke runs already show first-iteration drag; final sweeps will discard initial iterations and interleave variants.
* **Allocator artifacts:** arena available; plus STL `reserve` variants for fairness.
* **Thermal/turbo drift:** scripts support CPU pinning; runs are chunked rather than marathon’d.
* **Branch predictor luck:** random vs Zipf workloads and different orderings; seeds logged.
* **Unfair baselines:** vector with and without `reserve`; `unordered_map` with and without `reserve` targeting load parity.

---

## What I actually achieved this week

* A **compile-clean, reproducible** benchmarking lab with:

  * four custom DS implementations,
  * mirrored STL baselines,
  * a CSV-first harness and plotting/analyze scripts,
  * environment controls and anti-DCE safeguards,
  * workloads that expose both algorithmic costs and hardware effects.
* A working path from **microbench** to **macro reasoning** (heap now supports pop-all, enabling end-to-end tasks like top-K or Dijkstra variants).
* A structure that is **extensible** for ablations (growth factors, load factors, probe schemes) without refactoring the core.

This is not a toy mockup anymore; it’s a small, honest lab that can make me wrong quickly and for the right reasons.

---

## What still feels unfinished (on purpose)

* Variance bands and **CIs** are not in the plots yet (the analyzer is ready; I need to run 20–50 trials per point).
* **Perf counters** (cycles, instructions, cache-misses, branch-misses) are gathered via a side command; I haven’t joined them to the CSV per-row yet.
* **Cache-knee probe** exists as a tiny program; I need to annotate vertical L1/L2/LLC lines on the DS plots for storytelling.
* Hash map only uses linear probing; robin-hood and quadratic are on deck for a fairer comparison with modern tables.

---

## Trade-offs I’m okay with

* I chose a **transparent, DIY harness** over Google Benchmark for a first pass. This makes the measurement model obvious and the CSV simple to post-process. If results drift, I can cross-check with GB later.
* I defaulted to **wall-time + checksum** in the main loop and run `perf` separately. It keeps the hot path clean and lets me scale trial counts without perf’s overhead when I just need medians.

---

## “One-liner trim” for future me

> Benchmarks are experiments: lock the environment and workload, then measure distributions and a couple of architectural signals to explain the curves.

(If space is tight, drop allocator discussions and keep locality + distributions + variance.)

---

## What’s next (short plan)

1. Run full sweeps (uniform + Zipf) with 20–50 trials per point; generate p10/median/p90 plots and a speedup table (custom vs STL).
2. Annotate cache knees using the probe; add CPI and miss-rate overlays from `perf`.
3. Add two ablations:

   * vector growth factor 1.5 vs 2.0 vs `reserve(N)`;
   * hash load 0.5/0.7/0.9 and linear vs quadratic probing.
4. One macro task: **top-K selection** using my heap vs priority\_queue on large arrays (vary K/N).

Expected résumé line once the ablations are in:

> Engineered a reproducible DS benchmark lab; explained performance with locality and counter signals. Matched `std::vector` within \~5–10% on bulk append (with `reserve` parity) and characterized hash-table probe behavior under Zipf clustering.

---

## Closing thought

The value here wasn’t squeezing a faster vector; it was turning vague “performance instincts” into testable expectations. If a plot bends, I want to know whether it’s L2, a growth spike, a branch predictor sulk, or my allocator screaming—then prove it.

(**P.S. :- Use of CHATGPT for learning, and formatting this note and readme file[s] - has been done extensively.**)
