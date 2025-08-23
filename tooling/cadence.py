#!/usr/bin/env python3
"""ProjectCadence CLI – create/info/current for custom weeks."""

import argparse
import datetime as dt
import sys
import textwrap
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
CFG = REPO / ".cadence.env"
WEEKS = REPO / "weeks"

DEF = {
    "SERIES_START_DATE": "2025-08-04",
    "WEEK_PREFIX": "W",
    "WEEKS_TOTAL": "52",
    "ALLOW_OVERFLOW": "false",
}


def getenv():
    """Load config, merge with defaults, strip quotes."""
    cfg = DEF.copy()
    if CFG.exists():
        for line in CFG.read_text().splitlines():
            if "=" not in line or line.lstrip().startswith("#"):
                continue
            key, val = line.split("=", 1)
            cfg[key.strip()] = val.strip().strip("'\"")
    return cfg


def as_bool(x):
    return str(x).lower() in {"1", "y", "yes", "true"}


def start_date(cfg):
    return dt.date.fromisoformat(cfg["SERIES_START_DATE"])


def week_bounds(series_start, week_no):
    start = series_start + dt.timedelta(days=7 * (week_no - 1))
    end = start + dt.timedelta(days=6)
    return start, end


def week_of_date(series_start, date_):
    return 1 + ((date_ - series_start).days // 7)


def slugify(text):
    cleaned = "".join(ch if ch.isalnum() else "-" for ch in text.lower())
    return "-".join(filter(None, cleaned.split("-")))


def mk_dirs(root):
    for d in [
        "src",
        "notebooks",
        "tests",
        "data",
        "docs",
        "scripts",
        "assets",
        "benchmarks",
        "examples",
    ]:
        (root / d).mkdir(parents=True, exist_ok=True)
    (root / "data" / ".gitkeep").touch()


def create(args):
    c = getenv()
    s = start_date(c)
    total = int(c["WEEKS_TOTAL"])
    allow = as_bool(c["ALLOW_OVERFLOW"])
    n = int(args.n)
    if n < 1 or (n > total and not allow):
        sys.exit(f"Week {n} outside 1..{total}")
    a, b = week_bounds(s, n)
    name = f"{c['WEEK_PREFIX']}{n:02d}_{a:%Y%m%d}_{slugify(args.slug)}"
    root = WEEKS / name
    if root.exists() and not args.force:
        sys.exit(f"Exists: {root}")
    root.mkdir(parents=True, exist_ok=True)
    mk_dirs(root)
    (root / "METADATA.yaml").write_text(
        textwrap.dedent(
            f"""
            week_no: {n}
            start_date: {a}
            end_date: {b}
            status: draft
            created_at: {dt.datetime.now().isoformat(timespec='seconds')}
            tags: []
        """
        )
    )
    (root / "README.md").write_text(
        f"# {name}\n\n**Week** {n:02d} – {a} → {b}\n\n", encoding="utf8"
    )
    cur = WEEKS / "current"
    if cur.exists() or cur.is_symlink():
        cur.unlink(missing_ok=True)
    try:
        cur.symlink_to(root, target_is_directory=True)
    except OSError:
        (WEEKS / "CURRENT.txt").write_text(str(root.resolve()))
    print(f"Created {root}")


def info(args):
    c = getenv()
    s = start_date(c)
    if args.date:
        d = dt.date.fromisoformat(args.date)
        n = week_of_date(s, d)
    else:
        n = int(args.week or week_of_date(s, dt.date.today()))
    a, b = week_bounds(s, n)
    pref = c["WEEK_PREFIX"]
    found = list(WEEKS.glob(f"{pref}{n:02d}_{a:%Y%m%d}_*"))
    print(f"Week {n:02d}: {a} .. {b}")
    print("folder:", found[0].name if found else "(not created)")


def cur(args):
    if args.set is not None:
        c = getenv()
        s = start_date(c)
        a, _ = week_bounds(s, int(args.set))
        pref = c["WEEK_PREFIX"]
        matches = list(WEEKS.glob(f"{pref}{int(args.set):02d}_{a:%Y%m%d}_*"))
        if not matches:
            sys.exit("folder not found")
        tgt = matches[0]
        link = WEEKS / "current"
        if link.exists():
            link.unlink()
        try:
            link.symlink_to(tgt, target_is_directory=True)
        except OSError:
            (WEEKS / "CURRENT.txt").write_text(str(tgt.resolve()))
        print("current →", tgt.name)
    else:
        link = WEEKS / "current"
        txt = WEEKS / "CURRENT.txt"
        if link.is_symlink():
            print("current →", link.readlink())
        elif txt.exists():
            print("CURRENT.txt →", txt.read_text().strip())
        else:
            print("No current set")


def main():
    a = argparse.ArgumentParser()
    sub = a.add_subparsers(dest="cmd", required=True)
    new = sub.add_parser("new")
    new.add_argument("n")
    new.add_argument("slug")
    new.add_argument("--force", action="store_true")
    new.set_defaults(f=create)
    inf = sub.add_parser("info")
    inf.add_argument("week", nargs="?")
    inf.add_argument("--date")
    inf.set_defaults(f=info)
    cur_p = sub.add_parser("current")
    cur_p.add_argument("--set", type=int)
    cur_p.set_defaults(f=cur)
    ns = a.parse_args()
    ns.f(ns)


if __name__ == "__main__":
    main()
