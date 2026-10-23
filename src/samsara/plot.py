#!/usr/bin/env python3
"""
samsara.plot - plot a GPS track (lat/lon) from samsara-get-data /
samsara-prep-data output, with an optional web-tile basemap underneath.
Installed as the ``samsara-plot`` console script.

Two use cases
-------------
1. Contiguous: the raw output of samsara-get-data - fixes are evenly spaced
   in time with no gaps, forming one continuous trip.
2. Multi-trip: output that has had one or more dropout windows removed
   (e.g. by samsara-prep-data) - the remaining fixes form several
   contiguous trips separated by large jumps in time. Each trip is drawn
   as its own segment/color so the dropouts are visually obvious.

Both cases are handled by the same function, ``plot_fixes``: it looks for
gaps in the timestamps larger than ``gap_threshold_s`` (auto-detected from
the data if not given) and starts a new trip/segment after each one. For
contiguous data this naturally yields a single segment.

In both cases, the fixes passed in must already be sorted by ts ascending.
Note that samsara-prep-data shuffles its output rows, so that CSV must be
re-sorted by ts before plotting (``read_fixes`` below does this for you).

Basemap
-------
By default, a web-tile basemap (Esri's World Street Map raster tiles, via
``contextily``) is drawn under the track. This requires network access the
first time a given tile is needed (tiles are cached locally by contextily
afterward) - pass ``--no-basemap`` to skip it for fully offline plotting.
Since tile providers serve Web Mercator (EPSG:3857), lat/lon fixes are
reprojected to that CRS with ``pyproj`` before plotting; this also makes the
plot's aspect ratio exactly correct with no latitude-dependent approximation.

Usage
-----
    samsara-plot fixes.csv -o route.png
    samsara-plot prepped.csv -o multi_trip.png
    samsara-plot fixes.csv -o route.png --no-basemap
"""

import argparse
import csv
import sys

import contextily as cx
import matplotlib.pyplot as plt
import numpy as np
from pyproj import Transformer

# OpenStreetMap's own tile server (OpenStreetMap.Mapnik) enforces a strict
# usage policy (registered User-Agent, low volume) and will return 403s for
# casual/automated use; Esri's World Street Map tiles are free, require no
# API key, and don't impose that restriction, so they're used as the default.
DEFAULT_BASEMAP_SOURCE = cx.providers.Esri.WorldStreetMap

_WGS84_TO_WEBMERC = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)


def read_fixes(path):
    """Read ts, lat, lon (as floats) from a fixes CSV, sorted by ts ascending.

    Extra columns (speed_mps, heading_deg, ...) are ignored if present, so
    this works on both route_gps_sim.py's raw output and prep_data.py's
    trimmed ts/lat/lon output.
    """
    inp = sys.stdin if path == "-" else open(path, newline="")
    try:
        reader = csv.DictReader(inp)
        required = {"ts", "lat", "lon"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise SystemExit(
                f"Input is missing required column(s): {', '.join(sorted(missing))}"
            )
        rows = sorted(
            ((float(r["ts"]), float(r["lat"]), float(r["lon"])) for r in reader),
            key=lambda r: r[0],
        )
    finally:
        if inp is not sys.stdin:
            inp.close()
    if not rows:
        raise SystemExit("Input has no data rows.")
    ts, lat, lon = (np.array(col) for col in zip(*rows))
    return ts, lat, lon


def split_trips(ts, gap_threshold_s=None):
    """
    Split ascending timestamps into contiguous trip segments.

    A new trip starts wherever the gap between consecutive timestamps
    exceeds gap_threshold_s. If gap_threshold_s is not given, it is
    inferred as 5x the median sampling interval, which comfortably
    separates normal fix-to-fix spacing from multi-minute dropout gaps
    while still catching them (dropouts are 10-15 min; typical sampling
    intervals are seconds).

    Returns a list of (start_idx, end_idx) index pairs (end exclusive)
    into ts, covering the whole array.
    """
    ts = np.asarray(ts)
    n = len(ts)
    if n < 2:
        return [(0, n)]

    dt = np.diff(ts)
    if gap_threshold_s is None:
        median_dt = np.median(dt)
        gap_threshold_s = max(median_dt * 5.0, median_dt + 1.0)

    breaks = np.flatnonzero(dt > gap_threshold_s) + 1
    bounds = [0, *breaks.tolist(), n]
    return [(bounds[i], bounds[i + 1]) for i in range(len(bounds) - 1)]


def to_web_mercator(lat, lon):
    """Reproject WGS84 (lat, lon) degrees to Web Mercator (x, y) metres."""
    lat = np.asarray(lat, dtype=float)
    lon = np.asarray(lon, dtype=float)
    x, y = _WGS84_TO_WEBMERC.transform(lon, lat)
    return np.asarray(x), np.asarray(y)


def plot_fixes(ts, lat, lon, gap_threshold_s=None, ax=None, title=None,
               basemap=True, basemap_source=None, basemap_zoom: int | str = "auto"):
    """
    Plot a GPS track, splitting it into separate colored trip segments
    wherever a time gap larger than gap_threshold_s occurs (auto-detected
    if None), optionally drawing a web-tile basemap underneath.

    Works for both contiguous single-trip data (one segment) and
    multi-trip data with dropout windows removed (one segment per
    surviving trip). Each segment's start/end fix is marked with a circle/
    square so direction of travel and dropout boundaries are visible.

    Parameters
    ----------
    ts, lat, lon : array-like, shape (N,)
        Fixes sorted by ts ascending.
    gap_threshold_s : float, optional
        Minimum time gap, in seconds, that starts a new trip segment.
        Auto-detected from the data if omitted.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on; a new figure/axes is created if omitted.
    title : str, optional
        Plot title; a sensible default is used if omitted.
    basemap : bool, optional
        If True (default), fetch and draw a web-tile basemap underneath the
        track via contextily. Requires network access for any tiles not
        already cached locally. Set to False for fully offline plotting.
    basemap_source : xyzservices.TileProvider, optional
        Tile provider to use; defaults to OpenStreetMap's standard style.
    basemap_zoom : int or "auto", optional
        Tile zoom level; "auto" (default) picks one based on the plotted
        extent.

    Returns
    -------
    matplotlib.axes.Axes
    """
    ts = np.asarray(ts)
    lat = np.asarray(lat)
    lon = np.asarray(lon)
    if not (len(ts) == len(lat) == len(lon)):
        raise ValueError("ts, lat, and lon must have the same length")
    if len(ts) == 0:
        raise ValueError("no fixes to plot")

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 8))

    # Web Mercator is what basemap tiles are served in, and it's conformal
    # (shapes/angles preserved locally), so plotting in these units avoids
    # the need for any latitude-dependent aspect-ratio correction.
    x, y = to_web_mercator(lat, lon)

    trips = split_trips(ts, gap_threshold_s)

    cmap = plt.get_cmap("tab10")
    for i, (s, e) in enumerate(trips):
        color = cmap(i % 10)
        label = f"Trip {i + 1} ({e - s} fixes)" if len(trips) > 1 else "Route"
        ax.plot(x[s:e], y[s:e], "-", color=color, linewidth=1.5,
                label=label, zorder=2)
        ax.scatter(x[s], y[s], color=color, marker="o", s=40,
                   edgecolors="black", linewidths=0.5, zorder=3)
        ax.scatter(x[e - 1], y[e - 1], color=color, marker="s", s=40,
                   edgecolors="black", linewidths=0.5, zorder=3)

    ax.set_aspect("equal", adjustable="datalim")

    if basemap:
        cx.add_basemap(
            ax,
            crs="EPSG:3857",
            source=basemap_source or DEFAULT_BASEMAP_SOURCE,
            zoom=basemap_zoom,
        )

    ax.set_xlabel("Easting (Web Mercator, m)")
    ax.set_ylabel("Northing (Web Mercator, m)")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title or (f"GPS track ({len(trips)} trips)" if len(trips) > 1
                            else "GPS track"))
    ax.legend(loc="best", fontsize="small")
    return ax


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Plot a GPS fixes CSV (ts, lat, lon) over a web-tile "
                     "basemap, splitting into separate trip segments "
                     "wherever there's a large gap in time (e.g. from a "
                     "dropped-data window)."
    )
    ap.add_argument(
        "input", help="input fixes CSV (ts, lat, lon columns), or '-' for stdin"
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
    ap.add_argument(
        "--no-basemap", dest="basemap", action="store_false",
        help="don't fetch/draw a basemap (useful for fully offline plotting)",
    )
    ap.add_argument(
        "--basemap-zoom", default="auto",
        help="tile zoom level (integer), or 'auto' (default) to pick one "
             "from the plotted extent",
    )
    args = ap.parse_args(argv)

    zoom = args.basemap_zoom
    if zoom != "auto":
        try:
            zoom = int(zoom)
        except ValueError:
            ap.error("--basemap-zoom must be an integer or 'auto'")

    ts, lat, lon = read_fixes(args.input)
    plot_fixes(ts, lat, lon, gap_threshold_s=args.gap_threshold_s,
               title=args.title, basemap=args.basemap, basemap_zoom=zoom)

    if args.output:
        plt.savefig(args.output, dpi=150, bbox_inches="tight")
        print(f"Wrote {args.output}", file=sys.stderr)
    else:
        plt.show()


if __name__ == "__main__":
    main()
