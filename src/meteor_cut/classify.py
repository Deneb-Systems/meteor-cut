"""Which moving things are meteors, satellites or planes.

A meteor crosses the sky in a straight line, fast, in well under a few seconds, and never in the same place twice.
People, torches, phones and cars fail at least one of those: they wander, move slowly, keep moving for longer, light
up the same patch again and again, or break into several moving pieces at once.

Satellites and planes come from the slow tracker (see scan.py): straight, steady, and slower than meteors. A plane
blinks, shows gaps, or travels as several lights together; a satellite is one steady point. That split is a rule of
thumb: a satellite that flares can pass for a plane.

Distances are fractions of the frame width, so the same rules work for 720p and 4K.
"""

from __future__ import annotations

import math
import statistics
from collections.abc import Sequence
from dataclasses import dataclass

from .geometry import angle_between_deg, any_within, chord_length, heading_deg, line_error, straightness
from .model import Kind, Reason, Rejection, Scan, SkyObject, Track
from .optics import DEFAULT_METEOR_ALTITUDE_KM, Camera, measure_motion
from .scan import SLOW_RATE_HZ


@dataclass(frozen=True)
class Rules:
    """Thresholds for telling meteors, satellites and planes from everything else that moves."""

    # Meteors (fast tracker).
    min_detections: int = 4
    min_length: float = 0.015  # of frame width
    min_speed: float = 0.04  # frame widths per second
    max_duration_s: float = 3.0
    min_straightness: float = 0.9
    max_line_error: float = 0.04  # RMS distance from the fitted line, as a fraction of the track's length
    # Lights and people: other moving things near the same spot within hotspot_window_s, either at other times
    # (a light that keeps flaring) or at the same time (a person: head, arms, a torch).
    hotspot_radius: float = 0.08  # of frame width
    hotspot_window_s: float = 10.0
    hotspot_repeats: int = 2
    hotspot_together: int = 2
    # Satellites and planes (slow tracker).
    slow_min_detections: int = 6
    slow_min_duration_s: float = 2.0
    slow_min_length: float = 0.03  # of frame width
    slow_min_speed: float = 0.002  # frame widths per second
    slow_max_speed: float = 0.12
    slow_min_straightness: float = 0.85
    slow_max_line_error: float = 0.06
    blink_variation: float = 0.25  # brightness spread (std / mean) above which a light counts as blinking
    # Lights of one plane fly side by side (wing tips), close together. Satellites of a train (Starlink) fly one
    # behind another along the same line, so a companion counts only when it is beside the light, not ahead or behind.
    group_radius: float = 0.02  # of frame width
    group_heading_deg: float = 15.0
    # A slow light moving where the meteor pass already saw people or lamps (erratic, repeating or grouped motion)
    # is one of them, not a satellite or plane crossing open sky.
    people_evidence: int = 2


@dataclass
class Classification:
    meteors: list[SkyObject]
    satellites: list[SkyObject]
    planes: list[SkyObject]
    rejected: list[Rejection]

    @property
    def others(self) -> list[SkyObject]:
        """Satellites and planes, in order of appearance."""
        return sorted([*self.satellites, *self.planes], key=lambda o: o.start_frame)

    @property
    def objects(self) -> list[SkyObject]:
        return [*self.meteors, *self.others]


def classify(
    scan: Scan,
    camera: Camera | None = None,
    rules: Rules | None = None,
    altitude_km: float = DEFAULT_METEOR_ALTITUDE_KM,
) -> Classification:
    rules = rules or Rules()
    meteors, rejected = find_meteors(scan, camera, rules, altitude_km)
    others, rejected_slow = find_slow_objects(scan, meteors, camera, rules, rejected)
    return Classification(
        meteors=meteors,
        satellites=[o for o in others if o.kind is Kind.SATELLITE],
        planes=[o for o in others if o.kind is Kind.PLANE],
        rejected=sorted([*rejected, *rejected_slow], key=lambda r: r.start_frame),
    )


def _meteor_shape(track: Track, fps: float, width: int, rules: Rules) -> Reason | None:
    pts = track.detections
    if len(pts) < rules.min_detections:
        return Reason.TOO_BRIEF
    length = chord_length(pts)
    duration = max(1, pts[-1].frame - pts[0].frame) / fps
    if length < rules.min_length * width:
        return Reason.MOVED_TOO_LITTLE
    if duration > rules.max_duration_s:
        return Reason.TOO_LONG
    if length / duration < rules.min_speed * width:
        return Reason.TOO_SLOW
    if straightness(pts) < rules.min_straightness or line_error(pts) > rules.max_line_error * length:
        return Reason.NOT_STRAIGHT
    return None


def _crowd(track: Track, moving: Sequence[Track], radius: float, window: float) -> tuple[int, int]:
    """Other moving things near this one within the time window: (at other times, at the same time)."""
    pts = track.detections
    first, last = pts[0].frame, pts[-1].frame
    repeats = together = 0
    for other in moving:
        if other is track or other.scene != track.scene:
            continue
        o_first, o_last = other.detections[0].frame, other.detections[-1].frame
        if o_last < first - window or o_first > last + window or not any_within(pts, other.detections, radius):
            continue
        if o_first <= last and o_last >= first:
            together += 1
        else:
            repeats += 1
    return repeats, together


def _number_and_rank(objects: list[SkyObject]) -> None:
    """Number in order of appearance (how people refer to them) and rank fastest first."""
    objects.sort(key=lambda o: o.start_frame)
    for number, obj in enumerate(objects, start=1):
        obj.number = number
    for rank, obj in enumerate(sorted(objects, key=lambda o: o.speed_value, reverse=True), start=1):
        obj.rank = rank


def find_meteors(
    scan: Scan,
    camera: Camera | None = None,
    rules: Rules | None = None,
    altitude_km: float = DEFAULT_METEOR_ALTITUDE_KM,
) -> tuple[list[SkyObject], list[Rejection]]:
    rules = rules or Rules()
    moving = [t for t in scan.tracks if len(t.detections) >= 2]
    radius = rules.hotspot_radius * scan.width
    window = rules.hotspot_window_s * scan.fps
    meteors: list[SkyObject] = []
    rejected: list[Rejection] = []
    for track in moving:
        pts = track.detections
        reason = _meteor_shape(track, scan.fps, scan.width, rules)
        if reason is None:
            repeats, together = _crowd(track, moving, radius, window)
            if repeats >= rules.hotspot_repeats:
                reason = Reason.REPEATS_IN_PLACE
            elif together >= rules.hotspot_together:
                reason = Reason.MOVES_IN_GROUP
        if reason is not None:
            rejected.append(Rejection(track.id, track.scene, pts[0].frame, pts[-1].frame, reason))
            continue
        meteor = SkyObject(Kind.METEOR, track.id, track.scene, pts, scan.fps, straightness(pts))
        if camera is not None:
            meteor.motion = measure_motion(camera, meteor.start, meteor.end, meteor.duration_s, altitude_km)
        meteors.append(meteor)
    _number_and_rank(meteors)
    return meteors, rejected


def _is_meteor_echo(track: Track, meteors: Sequence[SkyObject], radius: float) -> bool:
    """The slow tracker's view of a meteor: no longer than the meteor plus a second, and in the same place."""
    pts = track.detections
    for m in meteors:
        if (
            m.scene == track.scene
            and pts[0].frame >= m.start_frame - m.fps
            and pts[-1].frame <= m.end_frame + m.fps
            and any(math.dist((p.cx, p.cy), m.start) <= radius or math.dist((p.cx, p.cy), m.end) <= radius for p in pts)
        ):
            return True
    return False


_PEOPLE = {Reason.NOT_STRAIGHT, Reason.REPEATS_IN_PLACE, Reason.MOVES_IN_GROUP}


def _among_people(track: Track, people: Sequence[Track], radius: float, window: float, needed: int) -> bool:
    """At least `needed` fast tracks rejected as people or lamps, near this track's path while it moved."""
    pts = track.detections
    first, last = pts[0].frame - window, pts[-1].frame + window
    near = 0
    for other in people:
        if other.scene == track.scene and other.points[0].frame <= last and other.points[-1].frame >= first:
            near += any_within(pts, other.points, radius)
            if near >= needed:
                return True
    return False


def _slow_shape(
    track: Track, meteors: Sequence[SkyObject], people: Sequence[Track], scan: Scan, rules: Rules
) -> Reason | None:
    pts = track.detections
    duration = max(1, pts[-1].frame - pts[0].frame) / scan.fps
    if len(pts) < rules.slow_min_detections or duration < rules.slow_min_duration_s:
        return Reason.SLOW_TOO_BRIEF
    if _is_meteor_echo(track, meteors, rules.hotspot_radius * scan.width):
        return Reason.PART_OF_METEOR
    length = chord_length(pts)
    if length < rules.slow_min_length * scan.width:
        return Reason.SLOW_MOVED_TOO_LITTLE
    if not rules.slow_min_speed * scan.width <= length / duration <= rules.slow_max_speed * scan.width:
        return Reason.SLOW_SPEED_RANGE
    if straightness(pts) < rules.slow_min_straightness or line_error(pts) > rules.slow_max_line_error * length:
        return Reason.SLOW_NOT_STRAIGHT
    radius, window = rules.hotspot_radius * scan.width, rules.hotspot_window_s * scan.fps
    if _among_people(track, people, radius, window, rules.people_evidence):
        return Reason.SLOW_AMONG_PEOPLE
    return None


def _beside(track: Track, other: Track, radius: float) -> bool:
    """`other` keeps pace beside `track` (offset mostly across the direction of travel), within `radius`."""
    pts = track.detections
    theirs = {p.frame: p for p in other.detections}
    heading = math.radians(heading_deg(pts))
    ux, uy = math.cos(heading), math.sin(heading)
    offsets = [(q.cx - p.cx, q.cy - p.cy) for p in pts if (q := theirs.get(p.frame)) is not None]
    if len(offsets) < 3:
        return False
    dx = statistics.fmean(o[0] for o in offsets)
    dy = statistics.fmean(o[1] for o in offsets)
    along, across = abs(dx * ux + dy * uy), abs(-dx * uy + dy * ux)
    return math.hypot(dx, dy) <= radius and across > along


def _travels_with(track: Track, others: Sequence[Track], radius: float, max_heading_deg: float) -> bool:
    """Another light flying beside this one in the same direction: a plane's lights."""
    pts = track.detections
    return any(
        other is not track
        and other.scene == track.scene
        and angle_between_deg(heading_deg(other.detections), heading_deg(pts)) < max_heading_deg
        and _beside(track, other, radius)
        for other in others
    )


def find_slow_objects(
    scan: Scan,
    meteors: Sequence[SkyObject],
    camera: Camera | None = None,
    rules: Rules | None = None,
    rejected_fast: Sequence[Rejection] = (),
) -> tuple[list[SkyObject], list[Rejection]]:
    """Satellites and planes among the slow tracker's tracks.

    rejected_fast is what find_meteors left out; tracks it judged to be people or lamps mark places where slow
    movement is people too.
    """
    rules = rules or Rules()
    by_id = {t.id: t for t in scan.tracks}
    people = [by_id[r.track_id] for r in rejected_fast if r.reason in _PEOPLE and r.track_id in by_id]
    rejected: list[Rejection] = []
    candidates: list[Track] = []
    for track in scan.slow_tracks:
        pts = track.detections
        if len(pts) < 2:
            continue
        reason = _slow_shape(track, meteors, people, scan, rules)
        if reason is None:
            candidates.append(track)
        else:
            rejected.append(Rejection(track.id, track.scene, pts[0].frame, pts[-1].frame, reason))

    step = max(1, round(scan.fps / SLOW_RATE_HZ))  # the slow tracker sees every step-th frame
    radius = rules.group_radius * scan.width
    objects: list[SkyObject] = []
    for track in candidates:
        pts = track.detections
        peaks = [p.peak for p in pts]
        mean = statistics.fmean(peaks)
        variation = statistics.pstdev(peaks) / mean if mean else 0.0
        detected_share = min(1.0, len(pts) / ((pts[-1].frame - pts[0].frame) / step + 1))
        companion = _travels_with(track, candidates, radius, rules.group_heading_deg)
        # Gaps alone do not make a plane: faint satellites drop in and out of detection too.
        kind = Kind.PLANE if variation > rules.blink_variation or companion else Kind.SATELLITE
        diagnostics: dict[str, float | bool] = {
            "brightness_variation": round(variation, 3),
            "detected_share": round(detected_share, 3),
            "travels_with_another_light": companion,
        }
        obj = SkyObject(kind, track.id, track.scene, pts, scan.fps, straightness(pts), diagnostics=diagnostics)
        if camera is not None:
            # A satellite's or plane's height is unknown, so there is no km/s estimate.
            obj.motion = measure_motion(camera, obj.start, obj.end, obj.duration_s, altitude_km=None)
        objects.append(obj)

    # A plane's lights travelling together are one plane: keep the brightest track of each group.
    planes = [o for o in objects if o.kind is Kind.PLANE]
    merged: set[int] = set()
    for i, a in enumerate(planes):
        for b in planes[i + 1 :]:
            if a.track_id in merged or b.track_id in merged:
                continue
            if a.start_frame <= b.end_frame and b.start_frame <= a.end_frame and math.dist(a.start, b.start) <= radius:
                weaker = a if max(p.peak for p in a.points) < max(p.peak for p in b.points) else b
                merged.add(weaker.track_id)
                rejected.append(
                    Rejection(weaker.track_id, weaker.scene, weaker.start_frame, weaker.end_frame, Reason.SAME_PLANE)
                )
    objects = [o for o in objects if o.track_id not in merged]
    for kind in (Kind.SATELLITE, Kind.PLANE):
        _number_and_rank([o for o in objects if o.kind is kind])
    return sorted(objects, key=lambda o: o.start_frame), rejected
