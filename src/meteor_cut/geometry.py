"""Shape measures for a list of tracked points."""

from __future__ import annotations

import math
from collections.abc import Sequence
from itertools import pairwise

from .model import Point


def path_length(points: Sequence[Point]) -> float:
    return sum(math.dist((a.cx, a.cy), (b.cx, b.cy)) for a, b in pairwise(points))


def chord_length(points: Sequence[Point]) -> float:
    """Straight-line distance from the first point to the last."""
    return math.dist((points[0].cx, points[0].cy), (points[-1].cx, points[-1].cy))


def straightness(points: Sequence[Point]) -> float:
    """1 for a straight line, lower the more the path wanders."""
    path = path_length(points)
    return chord_length(points) / path if path > 0 else 0.0


def line_error(points: Sequence[Point]) -> float:
    """RMS distance of the points from their best-fit straight line."""
    n = len(points)
    mx = sum(p.cx for p in points) / n
    my = sum(p.cy for p in points) / n
    sxx = sum((p.cx - mx) ** 2 for p in points)
    syy = sum((p.cy - my) ** 2 for p in points)
    sxy = sum((p.cx - mx) * (p.cy - my) for p in points)
    # The smallest eigenvalue of the scatter matrix is the variance across the principal axis.
    trace, det = sxx + syy, sxx * syy - sxy * sxy
    smallest = trace / 2 - math.sqrt(max(0.0, trace * trace / 4 - det))
    return math.sqrt(max(0.0, smallest) / n)


def heading_deg(points: Sequence[Point]) -> float:
    """Direction from the first point to the last, in image coordinates (y down)."""
    a, b = points[0], points[-1]
    return math.degrees(math.atan2(b.cy - a.cy, b.cx - a.cx))


def angle_between_deg(a: float, b: float) -> float:
    """Smallest difference between two headings, 0-180."""
    return abs((a - b + 180) % 360 - 180)


def any_within(a: Sequence[Point], b: Sequence[Point], radius: float) -> bool:
    """True when some point of `a` is within `radius` of some point of `b`."""
    return any(math.dist((p.cx, p.cy), (q.cx, q.cy)) <= radius for p in a for q in b)
