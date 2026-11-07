"""Which moving things are meteors.

A meteor crosses the sky in a straight line, fast, in well under a few seconds, and never in the same place twice.
People, torches, phones and cars fail at least one of those: they wander, move slowly, keep moving for longer, light
up the same patch again and again, or break into several moving pieces at once.

Distances are given as fractions of the frame width, so the same settings work for 720p and 4K.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from .optics import Camera, MeteorSpeed, meteor_speed
from .scan import Point, Scan, Track


@dataclass(frozen=True)
class Rules:
    min_detections: int = 4
    min_length: float = 0.015     # of frame width
    min_speed: float = 0.04       # frame widths per second
    max_duration_s: float = 3.0
    min_straightness: float = 0.9  # straight-line distance / path length
    max_line_error: float = 0.04  # RMS distance from the fitted line, as a fraction of the track's length
    # Lights and people: other moving things near the same spot within this many seconds...
    hotspot_radius: float = 0.08  # of frame width
    hotspot_window_s: float = 10.0
    # ...at another time (a light that keeps flaring), or at the same time (a person: head, arms, a torch).
    hotspot_repeats: int = 2
    hotspot_together: int = 2


@dataclass
class Meteor:
    id: int
    scene: int
    points: list[Point]           # detections only, in frame order
    start_frame: int
    end_frame: int
    fps: float
    straightness: float
    speed: MeteorSpeed | None = None
    rank: int = 0                 # 1 = fastest
    extra: dict = field(default_factory=dict)

    @property
    def start(self) -> tuple[float, float]:
        return self.points[0].cx, self.points[0].cy

    @property
    def end(self) -> tuple[float, float]:
        return self.points[-1].cx, self.points[-1].cy

    @property
    def start_s(self) -> float:
        return self.start_frame / self.fps

    @property
    def end_s(self) -> float:
        return self.end_frame / self.fps

    @property
    def duration_s(self) -> float:
        return max(1, self.end_frame - self.start_frame) / self.fps

    @property
    def length_px(self) -> float:
        return math.dist(self.start, self.end)

    @property
    def speed_px_s(self) -> float:
        return self.length_px / self.duration_s

    @property
    def direction_deg(self) -> float:
        """Direction of travel on screen: 0 = right, 90 = up."""
        return math.degrees(math.atan2(self.start[1] - self.end[1], self.end[0] - self.start[0])) % 360

    def sort_key(self) -> float:
        if self.speed and math.isfinite(self.speed.angular_speed_deg_s):
            return self.speed.angular_speed_deg_s
        return self.speed_px_s


@dataclass
class Rejected:
    id: int
    scene: int
    start_frame: int
    end_frame: int
    reason: str


def _line_error(points: list[Point]) -> float:
    """RMS distance of the points from their best-fit straight line."""
    n = len(points)
    mx = sum(p.cx for p in points) / n
    my = sum(p.cy for p in points) / n
    sxx = sum((p.cx - mx) ** 2 for p in points)
    syy = sum((p.cy - my) ** 2 for p in points)
    sxy = sum((p.cx - mx) * (p.cy - my) for p in points)
    # Smallest eigenvalue of the covariance: variance across the principal axis.
    trace, det = sxx + syy, sxx * syy - sxy * sxy
    smallest = trace / 2 - math.sqrt(max(0.0, trace * trace / 4 - det))
    return math.sqrt(max(0.0, smallest) / n)


def _shape(track: Track, fps: float, width: int, rules: Rules) -> tuple[str | None, float]:
    pts = track.detections
    if len(pts) < rules.min_detections:
        return "too brief", 0.0
    length = math.dist((pts[0].cx, pts[0].cy), (pts[-1].cx, pts[-1].cy))
    path = sum(math.dist((a.cx, a.cy), (b.cx, b.cy)) for a, b in zip(pts, pts[1:]))
    straightness = length / path if path > 0 else 0.0
    duration = max(1, pts[-1].frame - pts[0].frame) / fps
    if length < rules.min_length * width:
        return "moved too little", straightness
    if duration > rules.max_duration_s:
        return "moved for too long (plane, satellite or person)", straightness
    if length / duration < rules.min_speed * width:
        return "too slow", straightness
    if straightness < rules.min_straightness or _line_error(pts) > rules.max_line_error * length:
        return "not a straight line", straightness
    return None, straightness


def _near(track: Track, other: Track, radius: float) -> bool:
    return any(math.dist((p.cx, p.cy), (q.cx, q.cy)) <= radius for p in track.detections for q in other.detections)


def find_meteors(scan: Scan, camera: Camera | None = None, rules: Rules = Rules(),
                 altitude_km: float | None = None) -> tuple[list[Meteor], list[Rejected]]:
    moving = [t for t in scan.tracks if len(t.detections) >= 2]
    meteors: list[Meteor] = []
    rejected: list[Rejected] = []
    radius = rules.hotspot_radius * scan.width
    window = rules.hotspot_window_s * scan.fps
    for track in moving:
        pts = track.detections
        first, last = pts[0].frame, pts[-1].frame
        reason, straightness = _shape(track, scan.fps, scan.width, rules)
        if reason is None:
            repeats = together = 0
            for other in moving:
                if other is track or other.scene != track.scene:
                    continue
                o_first, o_last = other.detections[0].frame, other.detections[-1].frame
                if o_last < first - window or o_first > last + window or not _near(track, other, radius):
                    continue
                if o_first <= last and o_last >= first:
                    together += 1
                else:
                    repeats += 1
            if repeats >= rules.hotspot_repeats:
                reason = "lights up the same spot again and again (a lamp, phone or person)"
            elif together >= rules.hotspot_together:
                reason = "moves together with other things (a person or vehicle)"
        if reason:
            rejected.append(Rejected(track.id, track.scene, first, last, reason))
            continue
        meteor = Meteor(track.id, track.scene, pts, first, last, scan.fps, straightness)
        if camera is not None:
            kwargs = {"altitude_km": altitude_km} if altitude_km else {}
            meteor.speed = meteor_speed(camera, meteor.start, meteor.end, meteor.duration_s, **kwargs)
        meteors.append(meteor)

    meteors.sort(key=lambda m: m.start_frame)
    for rank, meteor in enumerate(sorted(meteors, key=Meteor.sort_key, reverse=True), start=1):
        meteor.rank = rank
    # Number meteors in the order they appear, as people will refer to them.
    for number, meteor in enumerate(meteors, start=1):
        meteor.extra["number"] = number
    return meteors, rejected
