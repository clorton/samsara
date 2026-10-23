#!/usr/bin/env python3
"""
route_gps_sim.py - simulate GPS fixes along a driving route between two addresses.

Pipeline
--------
1. Geocode both addresses with OpenStreetMap Nominatim.
2. Request a driving route from an OSRM server, with per-segment distance/
   duration annotations.
3. Assign each route segment a travel speed: either OSRM's own speed
   estimate for the segment (the default) or a fixed speed you give with
   --default-speed-mph.
4. Walk the route at those speeds and emit a fix every --interval seconds
   (default 10 s), starting at t = 0 at the origin.

Output CSV columns
------------------
ts          seconds since departure
lat, lon     WGS84 degrees (with optional Gaussian noise; see --noise-m)
speed_mps    travel speed on the current segment, m/s
heading_deg  course over ground, degrees clockwise from true north
dist_m       distance travelled along the route, metres (noise-free)
speed_source "osrm_estimate" or "default"

Note: OSRM's HTTP API does not expose the posted OSM ``maxspeed`` tag via
route annotations (only ``distance``, ``duration``, ``nodes``, ``datasources``,
``weight`` and ``speed`` are supported), so segment speeds are always either
OSRM's own estimate or the fixed --default-speed-mph value.

Requirements: Python 3.8+, ``requests``.

Usage
-----
    python route_gps_sim.py "1600 Amphitheatre Pkwy, Mountain View, CA" \
        "1 Infinite Loop, Cupertino, CA" -o fixes.csv

Notes
-----
* Nominatim's usage policy requires a descriptive User-Agent and allows at most
  one request per second. Set --user-agent to something that identifies you.
* The public OSRM demo server (router.project-osrm.org) is for light use only.
  Use --osrm-url to point at your own instance for bulk work.
* Vehicles move at constant segment speed with instantaneous speed changes at
  segment boundaries: there is no acceleration, stopping at intersections,
  or traffic.
"""

import argparse
import csv
import math
import random
import sys
import time

import requests

from utils import haversine_distance

NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"
DEFAULT_OSRM_URL = "https://router.project-osrm.org"
DEFAULT_USER_AGENT = "route-gps-sim/1.0 (set --user-agent to your contact info)"

EARTH_RADIUS_M = 6_371_008.8
MPH_TO_MPS = 0.44704
KPH_TO_MPS = 1.0 / 3.6


# --------------------------------------------------------------------------- #
# Geometry helpers
# --------------------------------------------------------------------------- #
def haversine_m(lat1, lon1, lat2, lon2):
    """Great-circle distance in metres between a single pair of points."""
    return float(haversine_distance([[lat1, lon1]], [[lat2, lon2]])[0])


def bearing_deg(lat1, lon1, lat2, lon2):
    """Initial bearing from point 1 to point 2, degrees clockwise from north."""
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dl = math.radians(lon2 - lon1)
    x = math.sin(dl) * math.cos(p2)
    y = math.cos(p1) * math.sin(p2) - math.sin(p1) * math.cos(p2) * math.cos(dl)
    return (math.degrees(math.atan2(x, y)) + 360.0) % 360.0


# --------------------------------------------------------------------------- #
# Network services
# --------------------------------------------------------------------------- #
def geocode(address, session):
    """Return (lat, lon) for an address using Nominatim."""
    resp = session.get(
        NOMINATIM_URL,
        params={"q": address, "format": "jsonv2", "limit": 1},
        timeout=30,
    )
    resp.raise_for_status()
    hits = resp.json()
    if not hits:
        raise SystemExit(f"Could not geocode address: {address!r}")
    return float(hits[0]["lat"]), float(hits[0]["lon"])


def fetch_route(start, end, session, osrm_url):
    """
    Fetch a driving route from OSRM.

    Returns (coords, osrm_speeds):
      coords       list of (lat, lon) vertices, length N
      osrm_speeds  list of OSRM-estimated segment speeds in m/s, length N-1

    Note: OSRM's route annotations do not include the posted OSM ``maxspeed``
    tag (valid values are distance, duration, nodes, datasources, weight and
    speed), so this only fetches distance/duration and derives OSRM's own
    speed estimate per segment.
    """
    url = (
        f"{osrm_url.rstrip('/')}/route/v1/driving/"
        f"{start[1]},{start[0]};{end[1]},{end[0]}"
    )
    params = {
        "overview": "full",
        "geometries": "geojson",
        "annotations": "distance,duration",
        "steps": "false",
    }
    resp = session.get(url, params=params, timeout=60)
    resp.raise_for_status()
    data = resp.json()
    if data.get("code") != "Ok" or not data.get("routes"):
        raise SystemExit(f"OSRM routing failed: {data.get('code')} {data.get('message', '')}")

    route = data["routes"][0]
    coords = [(lat, lon) for lon, lat in route["geometry"]["coordinates"]]

    osrm_speeds = []
    for leg in route["legs"]:
        ann = leg.get("annotation", {})
        dists = ann.get("distance", [])
        durs = ann.get("duration", [])
        for d, t in zip(dists, durs):
            osrm_speeds.append(d / t if t > 0 else 0.0)

    if len(osrm_speeds) != len(coords) - 1:
        raise SystemExit(
            f"Route geometry has {len(coords)} vertices but {len(osrm_speeds)} "
            "annotated segments; cannot align speeds to geometry."
        )
    return coords, osrm_speeds


# --------------------------------------------------------------------------- #
# Speed assignment
# --------------------------------------------------------------------------- #
def assign_speeds(osrm_speeds, default_mps):
    """Choose a travel speed (m/s) and its source label for each segment."""
    speeds, sources = [], []
    for est in osrm_speeds:
        if default_mps is not None:
            speeds.append(default_mps)
            sources.append("default")
        elif est > 0:
            speeds.append(est)
            sources.append("osrm_estimate")
        else:
            # Zero-length or zero-duration segment; any positive speed works
            # because it contributes no time.
            speeds.append(1.0)
            sources.append("osrm_estimate")
    return speeds, sources


# --------------------------------------------------------------------------- #
# Sampling
# --------------------------------------------------------------------------- #
def sample_route(coords, speeds, sources, interval_s, include_arrival=False):
    """
    Walk the route at the given segment speeds and return a fix every
    interval_s seconds as a list of dicts (noise-free).
    """
    seg_len = [
        haversine_m(*coords[i], *coords[i + 1]) for i in range(len(coords) - 1)
    ]

    # Cumulative time and distance at each vertex.
    t_vert, d_vert = [0.0], [0.0]
    for length, v in zip(seg_len, speeds):
        t_vert.append(t_vert[-1] + length / v)
        d_vert.append(d_vert[-1] + length)
    total_t = t_vert[-1]

    times = [k * interval_s for k in range(int(total_t // interval_s) + 1)]
    if include_arrival and total_t - times[-1] > 1e-6:
        times.append(total_t)

    fixes = []
    seg = 0
    nseg = len(seg_len)
    for t in times:
        # Advance to the segment containing t, skipping zero-length segments.
        while seg < nseg - 1 and (t_vert[seg + 1] <= t or seg_len[seg] == 0):
            seg += 1
        t0, t1 = t_vert[seg], t_vert[seg + 1]
        f = 0.0 if t1 <= t0 else min(max((t - t0) / (t1 - t0), 0.0), 1.0)
        (lat0, lon0), (lat1, lon1) = coords[seg], coords[seg + 1]
        fixes.append(
            {
                "ts": t,
                "lat": lat0 + f * (lat1 - lat0),
                "lon": lon0 + f * (lon1 - lon0),
                "speed_mps": speeds[seg],
                "heading_deg": bearing_deg(lat0, lon0, lat1, lon1),
                "dist_m": d_vert[seg] + f * seg_len[seg],
                "speed_source": sources[seg],
            }
        )
    return fixes, total_t, d_vert[-1]


def add_noise(fixes, sigma_m, rng):
    """Add independent Gaussian horizontal error (sigma per axis, metres)."""
    for fx in fixes:
        dn = rng.gauss(0.0, sigma_m)
        de = rng.gauss(0.0, sigma_m)
        fx["lat"] += math.degrees(dn / EARTH_RADIUS_M)
        fx["lon"] += math.degrees(de / (EARTH_RADIUS_M * math.cos(math.radians(fx["lat"]))))


def write_csv(fixes, path):
    fields = ["ts", "lat", "lon", "speed_mps", "heading_deg", "dist_m", "speed_source"]
    out = sys.stdout if path == "-" else open(path, "w", newline="")
    try:
        w = csv.DictWriter(out, fieldnames=fields)
        w.writeheader()
        for fx in fixes:
            w.writerow(
                {
                    "ts": f"{fx['ts']:.1f}",
                    "lat": f"{fx['lat']:.7f}",
                    "lon": f"{fx['lon']:.7f}",
                    "speed_mps": f"{fx['speed_mps']:.2f}",
                    "heading_deg": f"{fx['heading_deg']:.1f}",
                    "dist_m": f"{fx['dist_m']:.1f}",
                    "speed_source": fx["speed_source"],
                }
            )
    finally:
        if out is not sys.stdout:
            out.close()


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Simulate GPS fixes along a driving route using OSRM's "
                     "segment speed estimates."
    )
    ap.add_argument("origin", help="starting address")
    ap.add_argument("destination", help="destination address")
    ap.add_argument("-o", "--output", default="gps_fixes.csv",
                    help="output CSV path, or '-' for stdout (default: gps_fixes.csv)")
    ap.add_argument("--interval", type=float, default=10.0,
                    help="seconds between fixes (default: 10)")
    ap.add_argument("--default-speed-mph", type=float, default=None,
                    help="fixed speed for every segment; if omitted, OSRM's "
                         "own speed estimate is used")
    ap.add_argument("--include-arrival", action="store_true",
                    help="append a final fix at the exact arrival time")
    ap.add_argument("--noise-m", type=float, default=0.0,
                    help="Gaussian position noise, sigma in metres per axis (default: 0)")
    ap.add_argument("--seed", type=int, default=None, help="random seed for noise")
    ap.add_argument("--osrm-url", default=DEFAULT_OSRM_URL,
                    help=f"OSRM server base URL (default: {DEFAULT_OSRM_URL})")
    ap.add_argument("--user-agent", default=DEFAULT_USER_AGENT,
                    help="User-Agent for Nominatim requests")
    args = ap.parse_args(argv)

    if args.interval <= 0:
        ap.error("--interval must be positive")

    session = requests.Session()
    session.headers["User-Agent"] = args.user_agent

    start = geocode(args.origin, session)
    time.sleep(1.0)  # Nominatim: max 1 request/second
    end = geocode(args.destination, session)
    print(f"Origin      {start[0]:.6f}, {start[1]:.6f}", file=sys.stderr)
    print(f"Destination {end[0]:.6f}, {end[1]:.6f}", file=sys.stderr)

    coords, osrm_speeds = fetch_route(start, end, session, args.osrm_url)
    default_mps = (args.default_speed_mph * MPH_TO_MPS
                   if args.default_speed_mph is not None else None)
    speeds, sources = assign_speeds(osrm_speeds, default_mps)

    fixes, total_t, total_d = sample_route(
        coords, speeds, sources, args.interval, args.include_arrival
    )
    if args.noise_m > 0:
        add_noise(fixes, args.noise_m, random.Random(args.seed))

    write_csv(fixes, args.output)

    estimated = sum(1 for s in sources if s == "osrm_estimate")
    print(
        f"Route {total_d / 1000:.2f} km, {total_t / 60:.1f} min; "
        f"{len(fixes)} fixes; {estimated}/{len(sources)} segments used OSRM's "
        "speed estimate.",
        file=sys.stderr,
    )
    if args.output != "-":
        print(f"Wrote {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
