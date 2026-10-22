#!/usr/bin/env python3
"""
route_gps_sim.py - simulate GPS fixes along a driving route between two addresses.

Pipeline
--------
1. Geocode both addresses with OpenStreetMap Nominatim.
2. Request a driving route from an OSRM server, with per-segment annotations
   that include the posted speed limit (``maxspeed``) from OpenStreetMap tags.
3. Assign each route segment a travel speed:
     * the posted limit, if OSM has one for that way;
     * otherwise a fallback: either OSRM's own speed estimate for the segment
       (the default) or a fixed speed you give with --default-speed-mph.
4. Walk the route at those speeds and emit a fix every --interval seconds
   (default 10 s), starting at t = 0 at the origin.

Output CSV columns
------------------
t_s          seconds since departure
lat, lon     WGS84 degrees (with optional Gaussian noise; see --noise-m)
speed_mps    travel speed on the current segment, m/s
heading_deg  course over ground, degrees clockwise from true north
dist_m       distance travelled along the route, metres (noise-free)
speed_source "posted", "osrm_estimate" or "default"

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
    """Great-circle distance in metres."""
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = p2 - p1
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * EARTH_RADIUS_M * math.asin(math.sqrt(a))


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

    Returns (coords, osrm_speeds, maxspeeds):
      coords       list of (lat, lon) vertices, length N
      osrm_speeds  list of OSRM-estimated segment speeds in m/s, length N-1
      maxspeeds    list of OSRM maxspeed annotation dicts, length N-1
    """
    url = (
        f"{osrm_url.rstrip('/')}/route/v1/driving/"
        f"{start[1]},{start[0]};{end[1]},{end[0]}"
    )
    params = {
        "overview": "full",
        "geometries": "geojson",
        "annotations": "distance,duration,maxspeed",
        "steps": "false",
    }
    resp = session.get(url, params=params, timeout=60)
    resp.raise_for_status()
    data = resp.json()
    if data.get("code") != "Ok" or not data.get("routes"):
        raise SystemExit(f"OSRM routing failed: {data.get('code')} {data.get('message', '')}")

    route = data["routes"][0]
    coords = [(lat, lon) for lon, lat in route["geometry"]["coordinates"]]

    osrm_speeds, maxspeeds = [], []
    for leg in route["legs"]:
        ann = leg.get("annotation", {})
        dists = ann.get("distance", [])
        durs = ann.get("duration", [])
        n = len(dists)
        for d, t in zip(dists, durs):
            osrm_speeds.append(d / t if t > 0 else 0.0)
        maxspeeds.extend(ann.get("maxspeed", [{"unknown": True}] * n))

    if len(osrm_speeds) != len(coords) - 1:
        raise SystemExit(
            f"Route geometry has {len(coords)} vertices but {len(osrm_speeds)} "
            "annotated segments; cannot align speeds to geometry."
        )
    return coords, osrm_speeds, maxspeeds


# --------------------------------------------------------------------------- #
# Speed assignment
# --------------------------------------------------------------------------- #
def parse_maxspeed(entry):
    """Convert an OSRM maxspeed annotation to m/s, or None if not posted."""
    if not isinstance(entry, dict) or "speed" not in entry:
        return None  # {"unknown": true} or {"none": true} (no limit)
    unit = entry.get("unit", "km/h").lower()
    speed = float(entry["speed"])
    if unit in ("mph", "mi/h"):
        return speed * MPH_TO_MPS
    return speed * KPH_TO_MPS


def assign_speeds(osrm_speeds, maxspeeds, default_mps):
    """Choose a travel speed (m/s) and its source label for each segment."""
    speeds, sources = [], []
    for est, ms in zip(osrm_speeds, maxspeeds):
        posted = parse_maxspeed(ms)
        if posted and posted > 0:
            speeds.append(posted)
            sources.append("posted")
        elif default_mps is not None:
            speeds.append(default_mps)
            sources.append("default")
        elif est > 0:
            speeds.append(est)
            sources.append("osrm_estimate")
        else:
            # Zero-length or zero-duration segment with no limit; any positive
            # speed works because it contributes no time.
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
                "t_s": t,
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
    fields = ["t_s", "lat", "lon", "speed_mps", "heading_deg", "dist_m", "speed_source"]
    out = sys.stdout if path == "-" else open(path, "w", newline="")
    try:
        w = csv.DictWriter(out, fieldnames=fields)
        w.writeheader()
        for fx in fixes:
            w.writerow(
                {
                    "t_s": f"{fx['t_s']:.1f}",
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
        description="Simulate GPS fixes along a driving route at posted speed limits."
    )
    ap.add_argument("origin", help="starting address")
    ap.add_argument("destination", help="destination address")
    ap.add_argument("-o", "--output", default="gps_fixes.csv",
                    help="output CSV path, or '-' for stdout (default: gps_fixes.csv)")
    ap.add_argument("--interval", type=float, default=10.0,
                    help="seconds between fixes (default: 10)")
    ap.add_argument("--default-speed-mph", type=float, default=None,
                    help="speed for segments with no posted limit; if omitted, "
                         "OSRM's own speed estimate is used")
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

    coords, osrm_speeds, maxspeeds = fetch_route(start, end, session, args.osrm_url)
    default_mps = (args.default_speed_mph * MPH_TO_MPS
                   if args.default_speed_mph is not None else None)
    speeds, sources = assign_speeds(osrm_speeds, maxspeeds, default_mps)

    fixes, total_t, total_d = sample_route(
        coords, speeds, sources, args.interval, args.include_arrival
    )
    if args.noise_m > 0:
        add_noise(fixes, args.noise_m, random.Random(args.seed))

    write_csv(fixes, args.output)

    posted = sum(1 for s in sources if s == "posted")
    print(
        f"Route {total_d / 1000:.2f} km, {total_t / 60:.1f} min at limits; "
        f"{len(fixes)} fixes; {posted}/{len(sources)} segments had a posted limit.",
        file=sys.stderr,
    )
    if args.output != "-":
        print(f"Wrote {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
