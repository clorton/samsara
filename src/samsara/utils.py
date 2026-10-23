#!/usr/bin/env python3
"""Shared geometry/math helpers for the samsara package."""

import numpy as np

EARTH_RADIUS_M = 6_371_008.8


def haversine_distance(points1, points2):
    """
    Vectorized great-circle distance between pairs of (lat, lon) points.

    Parameters
    ----------
    points1, points2 : array-like, shape (N, 2)
        Each row is [lat, lon] in decimal degrees. The two arrays must be
        broadcastable against each other (e.g. both (N, 2), or one (1, 2)
        to compare a single point against N points).

    Returns
    -------
    numpy.ndarray, shape (N,)
        Great-circle distance between each corresponding pair of points, in
        metres.
    """
    p1 = np.asarray(points1, dtype=float)
    p2 = np.asarray(points2, dtype=float)
    if p1.shape[-1] != 2 or p2.shape[-1] != 2:
        raise ValueError(f"points1 ({p1.shape=}) and points2 ({p2.shape}) must each have shape (N, 2): [lat, lon]")

    lat1 = np.radians(p1[..., 0])
    lon1 = np.radians(p1[..., 1])
    lat2 = np.radians(p2[..., 0])
    lon2 = np.radians(p2[..., 1])

    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2.0 * np.arcsin(np.sqrt(a))
    return EARTH_RADIUS_M * c
