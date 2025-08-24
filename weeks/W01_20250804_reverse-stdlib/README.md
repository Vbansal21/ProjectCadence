# Week 1 — Reverse-Engineering Stdlib (C++20)

**Week** 01 – 2025-08-04 → 2025-08-10

**Goal:** Recreate `std::vector`, `memcpy`, and a `std::sort`-like algorithm. Test first; make it fast and leak-free later.

## Build

```bash
cmake -S . -B build
cmake --build build -j
ctest --test-dir build --output-on-failure
```

## Tips

- Start with *one* feature end-to-end (e.g., `push_back` for `MyVector<int>`), then generalize to templates/iterators.
- Keep a growth policy invariant: `capacity >= size`, capacity grows monotonically.
- `memcpy` has **UB** on overlap; that's `memmove` territory.
- For sort: begin with a simple `insertion sort` for small `n` to validate interfaces; optimize later.
- Use `ASAN_OPTIONS=detect_leaks=1` when running tests.
