"""The radiant: the point a meteor shower's paths trace back to.

With a rectilinear lens a meteor's path is a straight line in the image, so the radiant is the point closest to all
those lines, behind each meteor. Sporadic meteors from other directions do not belong; the fit drops them in rounds of
tightening tolerance and reports how many meteors remain.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from .astro import ra_dec
from .model import SkyObject
from .showers import SkyPath

# Tolerances, as fractions of the frame width, for each round of the fit.
_ROUNDS = (0.25, 0.12, 0.06)


@dataclass(frozen=True)
class Radiant:
    x: float  # pixels; may lie outside the frame
    y: float
    members: int  # meteors whose paths point back to it
    spread_px: float  # RMS distance of their lines from the point
    numbers: tuple[int, ...] = ()  # those meteors' numbers

    def in_frame(self, width: int, height: int) -> bool:
        return 0 <= self.x < width and 0 <= self.y < height


def _cross(a: np.ndarray, b: np.ndarray) -> float:
    return float(a[0] * b[1] - a[1] * b[0])


def _closest_point(lines: Sequence[tuple[np.ndarray, np.ndarray]]) -> np.ndarray | None:
    """Least-squares point nearest to every line (point, unit direction)."""
    a = np.zeros((2, 2))
    b = np.zeros(2)
    for point, direction in lines:
        projection = np.eye(2) - np.outer(direction, direction)
        a += projection
        b += projection @ point
    if abs(np.linalg.det(a)) < 1e-6:  # all lines parallel
        return None
    return np.linalg.solve(a, b)


def find_radiant(meteors: Sequence[SkyObject], width: int, min_members: int = 5) -> Radiant | None:
    """The shower's radiant, or None when fewer than `min_members` meteors agree on one."""
    lines = []
    owners: list[int] = []
    for m in meteors:
        start, end = np.array(m.start), np.array(m.end)
        direction = end - start
        norm = float(np.linalg.norm(direction))
        if norm > 0:
            lines.append((start, direction / norm))
            owners.append(m.number)
    if len(lines) < min_members:
        return None
    keep = lines
    for round_, tolerance in enumerate(_ROUNDS):
        point = _closest_point(keep)
        if point is None:
            return None
        # A meteor belongs when its line passes near the point and, once the point has settled, it flies away from it.
        keep = [
            (p, d)
            for p, d in lines
            if abs(_cross(d, point - p)) <= tolerance * width and (round_ == 0 or float(np.dot(p - point, d)) > 0)
        ]
        if len(keep) < 2:
            return None
    point = _closest_point(keep)
    if point is None or len(keep) < min_members:
        return None
    if np.linalg.norm(point - np.array([width / 2, 0])) > 4 * width:  # nearly parallel paths: no usable point
        return None
    spread = math.sqrt(sum(_cross(d, point - p) ** 2 for p, d in keep) / len(keep))
    kept = {id(line) for line in keep}
    numbers = tuple(n for n, line in zip(owners, lines, strict=True) if id(line) in kept)
    return Radiant(float(point[0]), float(point[1]), len(keep), spread, numbers)


# Tolerances in degrees for each round of the fit on the sky.
_SKY_ROUNDS = (15.0, 8.0, 4.0)


@dataclass(frozen=True)
class SkyRadiant:
    ra_deg: float
    dec_deg: float
    members: int
    spread_deg: float  # RMS distance of the members' great circles from the radiant
    numbers: tuple[int, ...] = ()


def find_sky_radiant(paths: Sequence[tuple[int, SkyPath]], min_members: int = 5) -> SkyRadiant | None:
    """The radiant among the stars, from meteors' paths as great circles (numbered by meteor).

    The point closest to every great circle is the eigenvector of the smallest eigenvalue of the sum of their poles'
    outer products; of its two opposite directions the radiant is the one the meteors fly away from. Rounds of
    tightening tolerance drop sporadic meteors, as in find_radiant.
    """
    usable = [(n, p) for n, p in paths if np.linalg.norm(np.cross(p.start, p.end)) > 1e-9]
    if len(usable) < min_members:
        return None

    def fit(members: Sequence[tuple[int, SkyPath]]) -> np.ndarray:
        poles = np.array([p.pole for _, p in members])
        point = np.linalg.eigh(poles.T @ poles)[1][:, 0]
        away = sum(1 for _, p in members if np.dot(p.end - p.start, point) < 0)
        return point if away * 2 >= len(members) else -point

    keep = usable
    for round_, tolerance in enumerate(_SKY_ROUNDS):
        if len(keep) < 2:
            return None
        point = fit(keep)
        limit = math.sin(math.radians(tolerance))
        keep = [
            (n, p)
            for n, p in usable
            if abs(float(np.dot(p.pole, point))) <= limit and (round_ == 0 or np.dot(p.end - p.start, point) < 0)
        ]
    if len(keep) < min_members:
        return None
    point = fit(keep)
    spread = math.degrees(
        math.sqrt(sum(math.asin(min(1.0, abs(float(np.dot(p.pole, point))))) ** 2 for _, p in keep) / len(keep))
    )
    ra, dec = ra_dec(point)
    return SkyRadiant(ra, dec, len(keep), spread, tuple(n for n, _ in keep))
