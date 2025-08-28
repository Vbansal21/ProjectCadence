#!/usr/bin/env python3
"""ProjectCadence CLI – weeks + alternate tracks (EXP/HYP/LAB/THY/MAJ/RES/INT).

Commands
--------
Weeks (unchanged):
  cadence.py new <week_no> <slug> [--force]
  cadence.py info [<week_no>] [--date YYYY-MM-DD] [--track weeks]
  cadence.py current [--set <week_no>] [--track weeks]

Alternate tracks:
  cadence.py newproj <track> <slug> [--id ID | --date YYYY-MM-DD] [--force]
  cadence.py current --track <track> [--name <slug-or-substring>]

Tracks
------
weeks  : weekly cadence rooted at ./weeks (existing behavior)
exp    : exploratory spikes     → ./exp       names: EXP_YYYYMMDD_<slug>
hyp    : hypothetical/design    → ./hyp       names: HYP_YYYYMMDD_<slug>
lab    : course labs            → ./labs      names: LAB_YYYYMMDD_<slug>  (use --id to inject course IDs)
thy    : theory-course builds   → ./theory    names: THY_YYYYMMDD_<slug>
maj    : major course projects  → ./majors    names: MAJ_YYYYMMDD_<slug>
res    : research artifacts     → ./research  names: RES_YYYYMMDD_<slug>
int    : internship work        → ./internships names: INT_YYYYMMDD_<slug>
"""

import argparse
import datetime as dt
import sys
import textwrap
from pathlib import Path

# --- Paths & config -----------------------------------------------------------

REPO = Path(__file__).resolve().parents[1]
CFG = REPO / ".cadence.env"

# Week track root (unchanged)
WEEKS = REPO / "weeks"

# Alternate tracks registry
TRACKS = {
    "weeks": {"dir": WEEKS, "prefix": None, "datefmt": "%Y%m%d"},
    "exp": {"dir": REPO / "exp", "prefix": "EXP", "datefmt": "%Y%m%d"},
    "hyp": {"dir": REPO / "hyp", "prefix": "HYP", "datefmt": "%Y%m%d"},
    "lab": {"dir": REPO / "labs", "prefix": "LAB", "datefmt": "%Y%m%d"},
    "thy": {"dir": REPO / "theory", "prefix": "THY", "datefmt": "%Y%m%d"},
    "maj": {"dir": REPO / "majors", "prefix": "MAJ", "datefmt": "%Y%m%d"},
    "res": {"dir": REPO / "research", "prefix": "RES", "datefmt": "%Y%m%d"},
    "int": {"dir": REPO / "internships", "prefix": "INT", "datefmt": "%Y%m%d"},
}

DEF = {
    "SERIES_START_DATE": "2025-08-04",
    "WEEK_PREFIX": "W",
    "WEEKS_TOTAL": "52",
    "ALLOW_OVERFLOW": "false",
}

# --- Utilities ----------------------------------------------------------------


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


def write_metadata_yaml(path, content: str):
    path.write_text(textwrap.dedent(content))


def set_current_symlink(track_root: Path, target: Path):
    link = track_root / "current"
    if link.exists() or link.is_symlink():
        link.unlink(missing_ok=True)
    try:
        link.symlink_to(target, target_is_directory=True)
    except OSError:
        (track_root / "CURRENT.txt").write_text(str(target.resolve()))


# --- Weeks: existing behavior --------------------------------------------------


def create_week(args):
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
    write_metadata_yaml(
        root / "METADATA.yaml",
        f"""
        track: weeks
        week_no: {n}
        start_date: {a}
        end_date: {b}
        status: draft
        created_at: {dt.datetime.now().isoformat(timespec="seconds")}
        tags: []
        """,
    )
    (root / "README.md").write_text(
        f"# {name}\n\n**Week** {n:02d} – {a} → {b}\n\n", encoding="utf8"
    )
    set_current_symlink(WEEKS, root)
    print(f"Created {root}")


def info(args):
    c = getenv()
    track = args.track or "weeks"
    if track == "weeks":
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
    else:
        tr = TRACKS[track]["dir"]
        if not tr.exists():
            print(f"{track} root does not exist yet: {tr}")
            return
        # Show latest or match by substring if provided via --name
        entries = [p for p in tr.iterdir() if p.is_dir()]
        if not entries:
            print(f"No entries under {tr}")
            return
        if args.name:
            key = slugify(args.name)
            matches = [p for p in entries if key in p.name.lower()]
            if not matches:
                print(f"No match for '{args.name}' under {tr}")
                return
            target = sorted(matches, key=lambda p: p.stat().st_mtime, reverse=True)[0]
        else:
            target = sorted(entries, key=lambda p: p.stat().st_mtime, reverse=True)[0]
        print(f"{track} latest/match:", target.name)


def cur(args):
    track = args.track or "weeks"
    if track == "weeks":
        if args.set is not None:
            c = getenv()
            s = start_date(c)
            a, _ = week_bounds(s, int(args.set))
            pref = c["WEEK_PREFIX"]
            matches = list(WEEKS.glob(f"{pref}{int(args.set):02d}_{a:%Y%m%d}_*"))
            if not matches:
                sys.exit("folder not found")
            tgt = matches[0]
            set_current_symlink(WEEKS, tgt)
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
    else:
        tr = TRACKS[track]["dir"]
        tr.mkdir(parents=True, exist_ok=True)
        if args.name:
            key = slugify(args.name)
            matches = [p for p in tr.iterdir() if p.is_dir() and key in p.name.lower()]
            if not matches:
                sys.exit(f"{track}: no folder matching '{args.name}'")
            tgt = sorted(matches, key=lambda p: p.stat().st_mtime, reverse=True)[0]
            set_current_symlink(tr, tgt)
            print(f"{track} current → {tgt.name}")
        else:
            link = tr / "current"
            txt = tr / "CURRENT.txt"
            if link.is_symlink():
                print(f"{track} current →", link.readlink())
            elif txt.exists():
                print(f"{track} CURRENT.txt →", txt.read_text().strip())
            else:
                print(f"{track}: No current set")


# --- Alternate tracks: creation ------------------------------------------------


def _date_id(date_str: str | None, fmt: str) -> str:
    if date_str:
        d = dt.date.fromisoformat(date_str)
    else:
        d = dt.date.today()
    return d.strftime(fmt)


def create_proj(args):
    track = args.track
    if track not in TRACKS or track == "weeks":
        sys.exit("Use 'new' for weeks; 'newproj' supports: exp/hyp/lab/thy/maj/res/int")
    reg = TRACKS[track]
    tr_root: Path = reg["dir"]
    tr_root.mkdir(parents=True, exist_ok=True)

    prefix = reg["prefix"]
    datefmt = reg["datefmt"]
    slug = slugify(args.slug)

    ident = args.id.strip() if args.id else _date_id(args.date, datefmt)
    name = f"{prefix}_{ident}_{slug}"
    root = tr_root / name
    if root.exists() and not args.force:
        sys.exit(f"Exists: {root}")

    root.mkdir(parents=True, exist_ok=True)
    mk_dirs(root)
    write_metadata_yaml(
        root / "METADATA.yaml",
        f"""
        track: {track}
        name: {slug}
        identifier: {ident}
        prefix: {prefix}
        status: draft
        created_at: {dt.datetime.now().isoformat(timespec="seconds")}
        tags: []
        """,
    )
    (root / "README.md").write_text(
        f"# {name}\n\n**Track** {track.upper()} — id: {ident}\n\n", encoding="utf8"
    )
    # Maintain per-track "current" pointer
    set_current_symlink(tr_root, root)
    print(f"Created {root}")


# --- CLI ----------------------------------------------------------------------


def main():
    a = argparse.ArgumentParser(prog="cadence.py")
    sub = a.add_subparsers(dest="cmd", required=True)

    # Weeks: new
    new = sub.add_parser("new", help="Create a weekly folder (weeks track).")
    new.add_argument("n", help="Week number (1-based).")
    new.add_argument("slug", help="Short slug for the week.")
    new.add_argument("--force", action="store_true", help="Overwrite if exists.")
    new.set_defaults(f=create_week)

    # Info (works for weeks; for other tracks shows latest or --name match)
    inf = sub.add_parser("info", help="Show week bounds or latest entry for a track.")
    inf.add_argument("week", nargs="?", help="Week number (weeks track only).")
    inf.add_argument("--date", help="YYYY-MM-DD (weeks track only).")
    inf.add_argument(
        "--track",
        choices=sorted(TRACKS.keys()),
        default="weeks",
        help="Track to query (default: weeks).",
    )
    inf.add_argument("--name", help="Substring/slug to match (non-weeks).")
    inf.set_defaults(f=info)

    # Current pointer (now per-track)
    cur_p = sub.add_parser("current", help="Get/set 'current' pointer per track.")
    cur_p.add_argument(
        "--track",
        choices=sorted(TRACKS.keys()),
        default="weeks",
        help="Track (default: weeks).",
    )
    cur_p.add_argument(
        "--set",
        type=int,
        help="For weeks: set to a given week number (1-based).",
    )
    cur_p.add_argument(
        "--name",
        help="For non-weeks: set by slug/substring (e.g., 'kv-cache' matches EXP_20250816_kv-cache-play).",
    )
    cur_p.set_defaults(f=cur)

    # New alternate project
    np = sub.add_parser(
        "newproj", help="Create an alternate project under a given track."
    )
    np.add_argument(
        "track",
        choices=[k for k in TRACKS.keys() if k != "weeks"],
        help="Project track (non-weeks).",
    )
    np.add_argument("slug", help="Project slug, e.g., 'kv-cache-play'.")
    np.add_argument(
        "--id",
        help="Identifier after prefix; defaults to date (YYYYMMDD). Example: CS301_Lab05 or 2025S7_MATH3.",
    )
    np.add_argument(
        "--date",
        help="YYYY-MM-DD to derive default ID (if --id not provided). Default: today.",
    )
    np.add_argument("--force", action="store_true", help="Overwrite if exists.")
    np.set_defaults(f=create_proj)

    ns = a.parse_args()
    ns.f(ns)


if __name__ == "__main__":
    main()
