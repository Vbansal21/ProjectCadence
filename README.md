# ProjectCadence

*One idea. One week. Fifty-two times.*

## Calendar-free cadence


| Week | Start (Mon) | End (Sun)  |
| ------ | ------------- | ------------ |
| 01   | 2025-08-04  | 2025-08-10 |
| 02   | 2025-08-11  | 2025-08-17 |
| …   | …          | …         |

Full table auto-computed via tooling.

## Quick start

```
# clone & enter
cd ProjectCadence

# activate env (creates .venv if absent)
source ./activate

# scaffold Week 1
python3 tooling/cadence.py new 1 vortex-viz

# commit
pre-commit run --all-files
git add .
git commit -m "feat(W01): scaffold vortex-viz"
```

## Core commands


| Purpose                | Command                                                    |
| ------------------------ | ------------------------------------------------------------ |
| **Create** week        | python3 tooling/cadence.py new<n> <slug>                   |
| **Info** for week/date | python3 tooling/cadence.py info 3   ·   --date 2025-09-05 |
| **Current** pointer    | python3 tooling/cadence.py current --set 4                 |

## Weekly workflow

1. `source ./activate`  – tools ready.
2. `cadence.py new …`  – scaffold folder.
3. Hack in `src/` or `notebooks/`.
4. Commit with prefix `feat(Wnn): …`.
5. End-of-week: add demo GIF, tag `git tag week-Wnn`.

## Promotion

```
# extract history into standalone repo
git subtree split --prefix weeks/W03_20250818_flow-mapper -b export/W03
```

---

© 2025 Vaibhav • Personal research licence
