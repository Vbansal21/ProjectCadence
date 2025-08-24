# W01 · Reverse-Engineering Stdlib — Day (-13) Sync Log

**When:** \[ideally-10th] 23 Aug 2025
**Repo:** `week-01-reverse-stdlib`
**Focus:** \[Learn C++ core] - get a red→green loop running for `MyVector`, `my_memcpy`, and `my_sort`.

---

## What happened

##### \[*built purposefully to fail and learn*]

- The initial scaffold built most targets but failed on `sort_tests` with:

```shell

error: ... declared using local type ‘<lambda>’, is used but never defined

````

and both `vector_tests` and `memcpy_tests` aborted because the functions intentionally called `std::abort()`.

- This is the classic *template-in-header* pitfall: I had declared `my_sort` in a header and tried to define it in a `.cpp`. With a comparator **lambda** from the test TU, the comparator type is TU-local, so the compiler can’t find a matching instantiation in the `.cpp`. Linker complains; tests don’t build.

---

## What I changed (minimal, principled)

1. **Sorting (header-only for now).**
   Moved the full `my_sort` template implementation into `src/my_sort.hpp` and left `src/my_sort.cpp` empty. Implemented a simple **insertion sort** (stable, easy to verify) with a `RandomAccessIterator` static_assert. Good enough to validate interfaces and comparators before I switch to an introsort hybrid.
2. **`my_memcpy` (correctness first).**
   Replaced the abort with a bytewise copy:

- Spec-aligned semantics: **no overlap** (that’s `memmove`), returns `dest`.
- Left room to widen later (word-sized chunks, unrolling, alignment).

3. **`MyVector<T>` (the smallest honest vector).**
   Implemented:

- `push_back(const T&)`, `push_back(T&&)`
- `reserve`, `clear`, **destructor**
- **move ctor/assign** (copy ops deliberately deleted for week-1 scope)
- Realloc policy: `cap = cap ? 2*cap : 1`, with `std::move_if_noexcept` during reallocation.
- Invariants I keep: `capacity ≥ size`, capacity grows monotonically, all constructed elements destroyed exactly once in dtor/clear.

4. **Sanitizers on.**
   ASan/UBSan already wired via CMake; kept them enabled during tests to catch lifetime mistakes early.

---

## What I learned / reinforced

- **Template linkage reality check.**
  Function templates that depend on TU-local types (lambdas, unnamed structs) must be fully visible in the header. Declaring in a header but defining in a `.cpp` invites ODR/link errors unless you explicitly manage instantiations (and you can’t for anonymous types).
- **Comparator is part of the type.**
  `Compare` in `my_sort` isn’t “just a function”; the lambda’s *type* is unique. Header-only is the path of least resistance here.
- **Vector’s truth is object lifetime.**
  The work isn’t `new[]`; it’s *placement-new into raw storage* and manually calling destructors on the constructed range. Reallocation is a dance:

1) allocate raw storage for `new_cap`
2) move/copy-construct into new storage
3) destroy old objects
4) free old storage
   Exception safety hinges on doing (3) after (2) succeeds.

- **memcpy vs memmove.**
  Overlap is **UB** for `memcpy`. Tests should never rely on overlap unless I implement `memmove`.
- **WSL/OneDrive “clock skew”** is a harmless warning; clean builds remove the noise.

---

## What I did (commands + outcomes)

```bash
# Clean rebuild to dodge any stale objs
rm -rf build && cmake -S . -B build && cmake --build build -j
ctest --test-dir build --output-on-failure
````

* `sort_tests` now builds and passes (insertion sort).
* `memcpy_tests` passes (byte-exact small + 1MiB).
* `vector_tests` passes: growth policy holds; `Tracked::live` returns to zero after scope, so destruction counts are sane.

Committed:

```bash
feat: header-defined my_sort; implement my_memcpy; minimal MyVector (push_back/reserve/dtor/move)
```

---

## Notes to future me

* **Debt intentionally left:**

  * Copy ctor/assign; `resize`, `shrink_to_fit`, iterators, `insert/erase` family.
  * `my_sort`: switch to introsort (quicksort + heap fallback, insertion for small bins).
  * `my_memcpy`: widen with `std::uintptr_t` blocks + tail; consider `restrict` builds to guide the compiler.
* **Testing guardrails to add soon:**

  * Fuzzer-ish randomized push/pop sequences with `Tracked` (counts, moves).
  * Exception-throwing element type to sanity-check strong guarantees during reallocate.
  * Separate `memmove` tests to make the overlap contract explicit.
* **Bench harness:** add a non-sanitized `-O2` target for real perf numbers; ASan skews results.

---

## One-line takeaways

* Templates that see lambdas live **in headers**, or pain ensues.
* Vectors are about **lifetimes**, not arrays.
* Start with the dumb correct thing; optimize where reality demands it.


(**P.S. :- Use of CHATGPT for learning, and formatting this note and readme file[s] - has been done extensively.**)
