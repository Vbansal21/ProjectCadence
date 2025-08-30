# Week 3 — Git-Flow + CI Foundations (Dev Notes)

**Window:** 2025-08-18 → 2025-08-24 (executing on 2025-08-30)
**Intent:** Use real Week 1 tests + Week 2 build to stand up CI that auto-lints, builds, and runs tests.

## Chosen shape

- Branches: `main` (release), `develop` (integration), short-lived `feature/*`.
- CI: GitHub Actions with jobs for pre-commit, Week 1 tests, Week 2 build.
- Guardrails: conditional steps so "missing week dir" doesn't fail CI.

## Today's log

- Confirmed `develop` branch exists and is tracking origin.
- Added CI workflow (pre-commit + W01 tests + W02 build).
- First PR to `main` will serve as the Week 3 proof.

## Pitfalls I'm watching

- CI toolchain vs. local flags (e.g., `-march=native` causing illegal instruction on runners).
- `CTest` not discovering tests if `enable_testing()` / `add_test()` misconfigured.
- Flaky network on dependency fetches (re-run job is fine).

## Decisions

- Start minimal; add coverage/CodeQL once green is consistent.
- Keep pre-commit as the first gate (fast feedback locally + in CI).

## After-action

- Note pipeline times, any flake causes, and tweaks carried into future weeks.
