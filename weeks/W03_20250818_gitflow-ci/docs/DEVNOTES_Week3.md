# Week 3 — Git-Flow + CI Foundations (Dev Notes)

**Window:** 2025-08-18 → 2025-08-24 (slipped; executing on 2025-08-30)
**Intent:** Stand up disciplined branching and a reusable CI that runs lint + build + tests across languages in this monorepo.
**Why this exists:** Future weeks should be reproducible without bespoke setup; CI enforces that through automation.

## Chosen shape

- Branching: Git-Flow lite — `main` (release), `develop` (integration), short-lived `feature/*`, occasional `hotfix/*`.
- CI: GitHub Actions with conditional jobs so empty stacks don't fail (Python, C++, Node, Docs).
- Pre-commit: formatting + hygiene to catch slop locally before CI.

## Today's log

- Created `develop` from `main` and pushed to origin.
- Generated Week-3 workspace via `cadence.py` and set `weeks/current`.
- Added `.pre-commit-config.yaml` and installed hooks.
- Dropped a smoke test and CI workflow to see one green pass.

## Pitfalls I'm watching

- "No tests found" as CI failure: guarded by conditional jobs.
- Pre-commit latency on first run (cache warms after).
- Line-ending churn on Windows; `.gitattributes` forces LF.
- Tooling mismatch: repo-level env vs per-project — keeping repo-level for speed.

## Decisions

- Keep Git-Flow for history clarity; can pivot to trunk on solo spikes.
- Keep CI minimal now; add coverage + CodeQL later when Week-3 closes.

## After-action (to write once green)

- What broke, what fixed it, average pipeline time, and changes I'll carry forward.
