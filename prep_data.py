#!/usr/bin/env python3
"""
prep_data.py - post-process GPS fixes produced by route_gps_sim.py.

Pipeline
--------
1. Read a fixes CSV (as produced by route_gps_sim.py) and keep only the
   t_s, lat, lon columns.
2. Remove N contiguous dropout windows (N given by --num-gaps), each
   spanning a random duration between --gap-min-minutes and
   --gap-max-minutes (default 10-15 min), to simulate a device losing its
   fix for a while.
3. Shuffle the remaining rows so timestamps are no longer in order, while
   keeping each (t_s, lat, lon) tuple intact.

Usage
-----
    python prep_data.py fixes.csv -n 2 -o prepped.csv
"""

import argparse
import csv
import random
import sys


def read_fixes(path):
    """Read t_s, lat, lon from a fixes CSV, sorted by t_s ascending.

    lat/lon are kept as the original strings (untouched); t_s is parsed to
    float so gap windows can be computed.
    """
    inp = sys.stdin if path == "-" else open(path, newline="")
    try:
        reader = csv.DictReader(inp)
        required = {"t_s", "lat", "lon"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise SystemExit(
                f"Input is missing required column(s): {', '.join(sorted(missing))}"
            )
        rows = [
            {"t_s": float(row["t_s"]), "lat": row["lat"], "lon": row["lon"]}
            for row in reader
        ]
    finally:
        if inp is not sys.stdin:
            inp.close()
    if not rows:
        raise SystemExit("Input has no data rows.")
    rows.sort(key=lambda r: r["t_s"])
    return rows


def choose_gaps(t_min, t_max, n, min_s, max_s, rng, max_attempts=2000):
    """
    Pick n non-overlapping [start, end) windows within [t_min, t_max], each
    with a duration drawn uniformly from [min_s, max_s] seconds.

    Uses rejection sampling: repeatedly draws random windows and retries if
    any overlap, up to max_attempts.
    """
    if n == 0:
        return []
    span = t_max - t_min
    if n * min_s > span:
        raise SystemExit(
            f"Route only spans {span:.0f}s; cannot fit {n} gap(s) of at "
            f"least {min_s:.0f}s each."
        )
    for _ in range(max_attempts):
        durations = [rng.uniform(min_s, max_s) for _ in range(n)]
        if any(d > span for d in durations):
            continue
        starts = [rng.uniform(t_min, t_max - d) for d in durations]
        windows = sorted(zip(starts, (s + d for s, d in zip(starts, durations))))
        if all(windows[i + 1][0] >= windows[i][1] for i in range(len(windows) - 1)):
            return windows
    raise SystemExit(
        f"Could not place {n} non-overlapping gap(s) of {min_s:.0f}-{max_s:.0f}s "
        "each after many attempts; try fewer/shorter gaps."
    )


def drop_gaps(rows, windows):
    """Remove rows whose t_s falls inside any [start, end) window."""

    def in_gap(t):
        return any(start <= t < end for start, end in windows)

    return [r for r in rows if not in_gap(r["t_s"])]


def write_csv(rows, path):
    out = sys.stdout if path == "-" else open(path, "w", newline="")
    try:
        w = csv.DictWriter(out, fieldnames=["t_s", "lat", "lon"])
        w.writeheader()
        for r in rows:
            w.writerow({"t_s": f"{r['t_s']:.1f}", "lat": r["lat"], "lon": r["lon"]})
    finally:
        if out is not sys.stdout:
            out.close()


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Strip a GPS fixes CSV down to t_s/lat/lon, drop N "
        "contiguous dropout windows, and shuffle the rows."
    )
    ap.add_argument(
        "input", help="input fixes CSV (from route_gps_sim.py), or '-' for stdin"
    )
    ap.add_argument(
        "-o", "--output", default="-",
        help="output CSV path, or '-' for stdout (default: stdout)",
    )
    ap.add_argument(
        "-n", "--num-gaps", type=int, default=1,
        help="number of contiguous dropout windows to remove (default: 1)",
    )
    ap.add_argument(
        "--gap-min-minutes", type=float, default=10.0,
        help="minimum duration of each dropout window, in minutes (default: 10)",
    )
    ap.add_argument(
        "--gap-max-minutes", type=float, default=15.0,
        help="maximum duration of each dropout window, in minutes (default: 15)",
    )
    ap.add_argument("--seed", type=int, default=None, help="random seed")
    args = ap.parse_args(argv)

    if args.num_gaps < 0:
        ap.error("--num-gaps must be >= 0")
    if args.gap_min_minutes <= 0 or args.gap_max_minutes <= 0:
        ap.error("--gap-min-minutes/--gap-max-minutes must be positive")
    if args.gap_min_minutes > args.gap_max_minutes:
        ap.error("--gap-min-minutes must be <= --gap-max-minutes")

    rng = random.Random(args.seed)

    rows = read_fixes(args.input)
    t_min, t_max = rows[0]["t_s"], rows[-1]["t_s"]

    windows = choose_gaps(
        t_min, t_max, args.num_gaps,
        args.gap_min_minutes * 60.0, args.gap_max_minutes * 60.0,
        rng,
    )
    kept = drop_gaps(rows, windows)
    rng.shuffle(kept)

    write_csv(kept, args.output)

    dropped = len(rows) - len(kept)
    for i, (start, end) in enumerate(windows, 1):
        print(
            f"Gap {i}: t={start:.1f}-{end:.1f}s ({(end - start) / 60:.1f} min)",
            file=sys.stderr,
        )
    print(
        f"Kept {len(kept)}/{len(rows)} fixes ({dropped} dropped across "
        f"{len(windows)} gap(s)), shuffled.",
        file=sys.stderr,
    )
    if args.output != "-":
        print(f"Wrote {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
