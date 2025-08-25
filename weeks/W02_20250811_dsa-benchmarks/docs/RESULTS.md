# Week 2 — Final Results (Summary)

**Setup.** C++20, Release (-O3 -march=native), single core. Runs pinned where available. Trials per point: 128. Distributions: uniform, Zipf(s=1.2). CSVs in `data/results/`.

**Vector (bulk append + scan).**
- Custom `DynArray` trails `std::vector` by ~5–12% without reserve; gap collapses with reserve.
- Spikes align with growth events; reserve removes them (amortized O(1) holds).

**List (push_front + scan).**
- `std::list` and custom list show flat but high ns/op; contiguous scans remain ~×(huge) faster due to locality.

**Heap (push then pop all).**
- Custom heap tracks `std::priority_queue` slope; constants within ~1.2–1.5× at mid-sizes.

**Hash (insert + mixed find).**
- Linear probing degrades as load→0.7; Zipf worsens probe clustering.
- `unordered_map` improves noticeably with `reserve()` (fairness matters).

**Variance & stability.**
- Medians stable; p10/p90 bands are tight enough to trust ordering.
- No DCE (checksum changes with perturbations); smoketests pass.

**Takeaway.** Locality and policy (reserve/load-factor) dominate perceived performance. The harness is reproducible and explains *why* curves bend, not just *what* wins.

*Artifacts:* CSVs, plots, `*_speedup_table.csv`, metadata sidecars (`*.meta`).
