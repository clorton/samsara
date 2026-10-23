#!/usr/bin/env python3
"""
plot.py - plot a GPS track (lat/lon) from route_gps_sim.py / prep_data.py output.

Two use cases
-------------
1. Contiguous: the raw output of route_gps_sim.py - fixes are evenly spaced
   in time with no gaps, forming one continuous trip.
2. Multi-trip: output that has had one or more dropout windows removed
   (e.g. by prep_data.py) - the remaining fixes form several contiguous
   trips separated by large jumps in time. Each trip is drawn as its own
   segment/color so the dropouts are visually obvious.

Both cases are handled by the same function, ``plot_fixes``: it looks for
gaps in the timestamps larger than ``gap_threshold_s`` (auto-detected from
the data if not given) and starts a new trip/segment after each one. For
contiguous data this naturally yields a single segment.

In both cases, the fixes passed in must already be sorted by t_s ascending.
Note that prep_data.py shuffles its output rows, so that CSV must be
re-sorted by t_s before plotting (``read_fixes`` below does this for you).

Usage
-----
    python plot.py fixes.csv -o route.png
    python plot.py prepped.csv -o multi_trip.png
"""

import argparse
import csv
import sys

import matplotlib.pyplot as plt
import numpy as np


def read_fixes(path):
    """Read t_s, lat, lon (as floats) from a fixes CSV, sorted by t_s ascending.

    Extra columns (speed_mps, heading_deg, ...) are ignored if present, so
    this works on both route_gps_sim.py's raw output and prep_data.py's
    trimmed t_s/lat/lon output.
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
        rows = sorted(
            ((float(r["t_s"]), float(r["lat"]), float(r["lon"])) for r in reader),
            key=lambda r: r[0],
        )
    finally:
        if inp is not sys.stdin:
            inp.close()
    if not rows:
        raise SystemExit("Input has no data rows.")
    t_s, lat, lon = (np.array(col) for col in zip(*rows))
    return t_s, lat, lon


def split_trips(t_s, gap_threshold_s=None):
    """
    Split ascending timestamps into contiguous trip segments.

    A new trip starts wherever the gap between consecutive timestamps
    exceeds gap_threshold_s. If gap_threshold_s is not given, it is
    inferred as 5x the median sampling interval, which comfortably
    separates normal fix-to-fix spacing from multi-minute dropout gaps
    while still catching them (dropouts are 10-15 min; typical sampling
    intervals are seconds).

    Returns a list of (start_idx, end_idx) index pairs (end exclusive)
    into t_s, covering the whole array.
    """
    t_s = np.asarray(t_s)
    n = len(t_s)
    if n < 2:
        return [(0, n)]

    dt = np.diff(t_s)
    if gap_threshold_s is None:
        median_dt = np.median(dt)
        gap_threshold_s = max(median_dt * 5.0, median_dt + 1.0)

    breaks = np.flatnonzero(dt > gap_threshold_s) + 1
    bounds = [0, *breaks.tolist(), n]
    return [(bounds[i], bounds[i + 1]) for i in range(len(bounds) - 1)]


def plot_fixes(t_s, lat, lon, gap_threshold_s=None, ax=None, title=None):
    """
    Plot a GPS track (longitude on x, latitude on y), splitting it into
    separate colored trip segments wherever a time gap larger than
    gap_threshold_s occurs (auto-detected if None).

    Works for both contiguous single-trip data (one segment) and
    multi-trip data with dropout windows removed (one segment per
    surviving trip). Each segment's start/end fix is marked with a circle/
    square so direction of travel and dropout boundaries are visible.

    Parameters
    ----------
    t_s, lat, lon : array-like, shape (N,)
        Fixes sorted by t_s ascending.
    gap_threshold_s : float, optional
        Minimum time gap, in seconds, that starts a new trip segment.
        Auto-detected from the data if omitted.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on; a new figure/axes is created if omitted.
    title : str, optional
        Plot title; a sensible default is used if omitted.

    Returns
    -------
    matplotlib.axes.Axes
    """
    t_s = np.asarray(t_s)
    lat = np.asarray(lat)
    lon = np.asarray(lon)
    if not (len(t_s) == len(lat) == len(lon)):
        raise ValueError("t_s, lat, and lon must have the same length")
    if len(t_s) == 0:
        raise ValueError("no fixes to plot")

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 8))

    trips = split_trips(t_s, gap_threshold_s)

    cmap = plt.get_cmap("tab10")
    for i, (s, e) in enumerate(trips):
        color = cmap(i % 10)
        label = f"Trip {i + 1} ({e - s} fixes)" if len(trips) > 1 else "Route"
        ax.plot(lon[s:e], lat[s:e], "-", color=color, linewidth=1.5,
                label=label, zorder=2)
        ax.scatter(lon[s], lat[s], color=color, marker="o", s=40,
                   edgecolors="black", linewidths=0.5, zorder=3)
        ax.scatter(lon[e - 1], lat[e - 1], color=color, marker="s", s=40,
                   edgecolors="black", linewidths=0.5, zorder=3)

    # Approximate equirectangular correction: at latitude phi, one degree
    # of longitude spans cos(phi) as much ground distance as one degree of
    # latitude, so compress the x-axis accordingly for a visually
    # undistorted track.
    mean_lat_rad = np.radians(np.mean(lat))
    ax.set_aspect(1.0 / max(np.cos(mean_lat_rad), 1e-6), adjustable="datalim")

    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.grid(True, alpha=0.3)
    ax.set_title(title or (f"GPS track ({len(trips)} trips)" if len(trips) > 1
                            else "GPS track"))
    ax.legend(loc="best", fontsize="small")
    return ax


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Plot a GPS fixes CSV (t_s, lat, lon), splitting into "
                     "separate trip segments wherever there's a large gap "
                     "in time (e.g. from a dropped-data window)."
    )
    ap.add_argument(
        "input", help="input fixes CSV (t_s, lat, lon columns), or '-' for stdin"
    )
    ap.add_argument(
        "-o", "--output", default=None,
        help="output image path (e.g. route.png); if omitted, shows interactively",
    )
    ap.add_argument(
        "--gap-threshold-s", type=float, default=None,
        help="time gap in seconds above which a new trip segment starts "
             "(default: auto-detected from the data)",
    )
    ap.add_argument("--title", default=None, help="plot title")
    args = ap.parse_args(argv)

    t_s, lat, lon = read_fixes(args.input)
    plot_fixes(t_s, lat, lon, gap_threshold_s=args.gap_threshold_s,
               title=args.title)

    if args.output:
        plt.savefig(args.output, dpi=150, bbox_inches="tight")
        print(f"Wrote {args.output}", file=sys.stderr)
    else:
        plt.show()


if __name__ == "__main__":
    main()
