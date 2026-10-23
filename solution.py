from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from utils import haversine_distance as hdist


@dataclass
class Point:
    ts: int
    lat: float
    lon: float

@dataclass
class Trip:
    vehicle_id: str
    start_ts: int
    end_ts: int
    distance: float

def get_gps_fixes(filename: str|Path, vehicle_id: str = "Vehicle_1", start_ts: int = 0, end_ts: int = np.uint64.max) -> list[Point]:
    fixes = pd.read_csv(filename)
    return [Point(*t) for t in fixes.itertuples(index=False)]

def get_sorted_data(fixes: list[Point]) -> tuple[NDArray[np.uint64], NDArray[np.float64]]:
    stamps = np.array([f.ts for f in fixes], dtype=np.uint64)
    points = np.array([[f.lat, f.lon] for f in fixes], dtype=np.float64)
    indices = np.argsort(stamps)
    stamps, points = np.array(stamps[indices], dtype=np.uint64), points[indices,:]

    return stamps, points

def get_trips(stamps: NDArray[np.uint64], points: NDArray[np.float64], vehicle_id: str, max_gap: int) -> list[Trip]:
    deltas = np.diff(stamps)
    gaps: NDArray[np.intp] = np.nonzero(deltas >= max_gap)[0]
    gaps = np.append(gaps, len(stamps)-1)   # account for last sequence

    trips = []
    ifirst = 0
    for ilast in gaps:
        if ilast > ifirst:
            # Normal case 2 or more fixes
            # [0:2] + [1:3], [3:5] + [4:6], and [7:9] + [8:10]
            dists = hdist(points[ifirst:ilast,:], points[ifirst+1:ilast+1,:])
            dist = dists.sum()
        else:
            # Lone point case
            # ifirst == ilast == 6
            dist = 0
        trips.append(Trip(vehicle_id, int(stamps[ifirst]), int(stamps[ilast]), float(dist)))
        ifirst = ilast + 1

    return trips

def main(fixes_filename: str = "test_fixes.csv", max_gap: int = 600, vehicle_id: str = "Vehicle_1"):

    fixes = get_gps_fixes(fixes_filename)

    stamps, points = get_sorted_data(fixes)

    trips = get_trips(stamps, points, vehicle_id, max_gap)

    for i, trip in enumerate(trips):
        print(f"{i}: {trip}")

    return


if __name__ == "__main__":
    main()
