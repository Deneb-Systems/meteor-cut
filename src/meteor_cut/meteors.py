"""Which moving things are meteors.

A meteor crosses the sky in a straight line, fast, in well under a few seconds, and never in the same place twice.
People, torches, phones and cars fail at least one of those: they wander, move slowly, keep moving for longer, light
up the same patch again and again, or break into several moving pieces at once.

Satellites and planes come from the slow tracker (see scan.py): straight, steady, and slower than meteors. A plane
blinks, shows gaps, or travels as several lights together; a satellite is one steady point. That split is a rule of
thumb: a satellite that flares can pass for a plane.

Distances are given as fractions of the frame width, so the same settings work for 720p and 4K.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from .optics import Camera, MeteorSpeed, meteor_speed
from .scan import SLOW_RATE_HZ, Point, Scan, Track


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
    # Satellites and planes (slow tracker).
    slow_min_detections: int = 6
    slow_min_duration_s: float = 2.0
    slow_min_length: float = 0.03   # of frame width
    slow_min_speed: float = 0.002   # frame widths per second
    slow_max_speed: float = 0.12
    slow_min_straightness: float = 0.85
    blink_variation: float = 0.25   # brightness spread (std / mean) above which a light counts as blinking
    blink_gaps: float = 0.7         # share of frames detected below which it counts as blinking

KINDS = ("meteor", "satellite", "plane")


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
    rank: int = 0                 # 1 = fastest, among objects of the same kind
    extra: dict = field(default_factory=dict)
    kind: str = "meteor"

    @property
    def label(self) -> str:
        return {"meteor": "#", "satellite": "S", "plane": "P"}[self.kind] + str(self.extra["number"])

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


def _speed_ok(m: Meteor) -> float:
    return m.speed.angular_speed_deg_s if m.speed and math.isfinite(m.speed.angular_speed_deg_s) else m.speed_px_s


def _overlaps(track: Track, meteors: list[Meteor], radius: float) -> bool:
    """The slow tracker's view of a meteor: no longer than the meteor plus a second, and in the same place."""
    pts = track.detections
    for m in meteors:
        if m.scene == track.scene and pts[0].frame >= m.start_frame - m.fps and pts[-1].frame <= m.end_frame + m.fps:
            if any(math.dist((p.cx, p.cy), m.start) <= radius or math.dist((p.cx, p.cy), m.end) <= radius
                   for p in pts):
                return True
    return False


def _heading(track: Track) -> float:
    a, b = track.detections[0], track.detections[-1]
    return math.degrees(math.atan2(b.cy - a.cy, b.cx - a.cx))


def find_slow_objects(scan: Scan, meteors: list[Meteor], camera: Camera | None = None,
                      rules: Rules = Rules()) -> tuple[list[Meteor], list[Rejected]]:
    """Satellites and planes among the slow tracker's tracks."""
    width, fps = scan.width, scan.fps
    candidates: list[Track] = []
    rejected: list[Rejected] = []
    for track in scan.slow_tracks:
        pts = track.detections
        if len(pts) < 2:
            continue
        first, last = pts[0].frame, pts[-1].frame
        duration = max(1, last - first) / fps
        length = math.dist((pts[0].cx, pts[0].cy), (pts[-1].cx, pts[-1].cy))
        path = sum(math.dist((a.cx, a.cy), (b.cx, b.cy)) for a, b in zip(pts, pts[1:]))
        reason = None
        if len(pts) < rules.slow_min_detections or duration < rules.slow_min_duration_s:
            reason = "slow and brief"
        elif _overlaps(track, meteors, rules.hotspot_radius * width):
            reason = "part of a meteor"
        elif length < rules.slow_min_length * width:
            reason = "slow and moved too little"
        elif not rules.slow_min_speed * width <= length / duration <= rules.slow_max_speed * width:
            reason = "slow object out of the satellite and plane speed range"
        elif (length / path if path else 0) < rules.slow_min_straightness or _line_error(pts) > 0.06 * length:
            reason = "slow and not a straight line"
        if reason:
            rejected.append(Rejected(track.id, track.scene, first, last, reason))
        else:
            candidates.append(track)

    objects: list[Meteor] = []
    radius = 0.05 * width
    for track in candidates:
        pts = track.detections
        first, last = pts[0].frame, pts[-1].frame
        peaks = [p.peak for p in pts]
        mean = sum(peaks) / len(peaks)
        variation = (sum((v - mean) ** 2 for v in peaks) / len(peaks)) ** 0.5 / mean if mean else 0.0
        # The slow tracker sees every `step`-th frame; a blinking light misses some of them.
        step = max(1, round(fps / SLOW_RATE_HZ))
        detected_share = min(1.0, len(pts) / ((last - first) / step + 1))
        companion = any(
            other is not track and other.scene == track.scene
            and other.detections[0].frame <= last and other.detections[-1].frame >= first
            and abs((_heading(other) - _heading(track) + 180) % 360 - 180) < 15
            and any(math.dist((p.cx, p.cy), (q.cx, q.cy)) <= radius for p in pts for q in other.detections)
            for other in candidates)
        kind = "plane" if variation > rules.blink_variation or detected_share < rules.blink_gaps or companion \
            else "satellite"
        length = math.dist((pts[0].cx, pts[0].cy), (pts[-1].cx, pts[-1].cy))
        path = sum(math.dist((a.cx, a.cy), (b.cx, b.cy)) for a, b in zip(pts, pts[1:]))
        obj = Meteor(track.id, track.scene, pts, first, last, fps, length / path if path else 0.0, kind=kind)
        obj.extra.update(brightness_variation=round(variation, 3), detected_share=round(detected_share, 3),
                         companion=companion)
        if camera is not None:
            speed = meteor_speed(camera, obj.start, obj.end, obj.duration_s)
            # The km/s estimate assumes a meteor's height; it means nothing for a satellite or a plane.
            obj.speed = MeteorSpeed(speed.angular_length_deg, speed.angular_speed_deg_s, speed.elevation_start_deg,
                                    speed.elevation_end_deg, None, None)
        objects.append(obj)

    # A plane's lights travelling together are one plane: keep the brightest track of each group.
    planes = [o for o in objects if o.kind == "plane"]
    dropped = set()
    for i, a in enumerate(planes):
        for b in planes[i + 1:]:
            if a.id in dropped or b.id in dropped:
                continue
            if a.start_frame <= b.end_frame and b.start_frame <= a.end_frame and math.dist(a.start, b.start) <= radius:
                weaker = a if max(p.peak for p in a.points) < max(p.peak for p in b.points) else b
                dropped.add(weaker.id)
    objects = [o for o in objects if o.id not in dropped]

    objects.sort(key=lambda o: o.start_frame)
    for kind in ("satellite", "plane"):
        group = [o for o in objects if o.kind == kind]
        for number, o in enumerate(group, start=1):
            o.extra["number"] = number
        for rank, o in enumerate(sorted(group, key=_speed_ok, reverse=True), start=1):
            o.rank = rank
    return objects, rejected

