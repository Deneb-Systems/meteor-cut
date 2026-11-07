"""The data Meteor Cut works with: tracked points, tracks, a scanned video, and the sky objects found in it."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, NamedTuple

if TYPE_CHECKING:
    import numpy as np

    from .optics import AngularMotion
    from .photometry import Brightness
    from .showers import Membership, SkyPath


class VideoInfo(NamedTuple):
    fps: float
    width: int
    height: int
    frames: int  # as the container reports it; the scan counts the frames it actually decodes


@dataclass
class Point:
    """One tracked position, in source-video pixels."""

    frame: int
    cx: float
    cy: float
    w: float
    h: float
    detected: bool  # False: the tracker's prediction for a frame without a detection
    peak: int = 0  # brightest grey value in the box, 0-255


@dataclass
class Track:
    id: int
    scene: int
    points: list[Point] = field(default_factory=list)
    slow: bool = False  # found by the slow tracker (satellites, planes)

    @property
    def detections(self) -> list[Point]:
        return [p for p in self.points if p.detected]


@dataclass
class Scan:
    """Everything the first pass found. Field names are stable: scans are cached between runs while developing."""

    path: str
    fps: float
    width: int
    height: int
    frames: int
    cuts: list[int]  # first frame of every scene after the first
    tracks: list[Track]
    slow_tracks: list[Track] = field(default_factory=list)
    backgrounds: dict[int, np.ndarray] = field(default_factory=dict)  # each scene's darkest frame

    @property
    def duration_s(self) -> float:
        return self.frames / self.fps if self.fps else 0.0

    def scene_of(self, frame: int) -> int:
        return sum(1 for cut in self.cuts if cut <= frame)

    def scene_bounds(self, scene: int) -> tuple[int, int]:
        """First and last frame of a scene."""
        starts = [0, *self.cuts]
        start = starts[scene]
        end = (starts[scene + 1] - 1) if scene + 1 < len(starts) else self.frames - 1
        return start, end


class Kind(str, Enum):
    METEOR = "meteor"
    SATELLITE = "satellite"
    PLANE = "plane"

    @property
    def prefix(self) -> str:
        """Label prefix: #3 for a meteor, S2 for a satellite, P1 for a plane."""
        return {"meteor": "#", "satellite": "S", "plane": "P"}[self.value]


class Reason(str, Enum):
    """Why a moving thing was not kept. The value is a stable code for CSV files and scripts."""

    TOO_BRIEF = "too_brief"
    MOVED_TOO_LITTLE = "moved_too_little"
    TOO_LONG = "too_long"
    TOO_SLOW = "too_slow"
    NOT_STRAIGHT = "not_straight"
    REPEATS_IN_PLACE = "repeats_in_place"
    MOVES_IN_GROUP = "moves_in_group"
    SLOW_TOO_BRIEF = "slow_too_brief"
    SLOW_MOVED_TOO_LITTLE = "slow_moved_too_little"
    SLOW_SPEED_RANGE = "slow_speed_range"
    SLOW_NOT_STRAIGHT = "slow_not_straight"
    PART_OF_METEOR = "part_of_meteor"
    SAME_PLANE = "same_plane"
    METEOR_FRAGMENT = "meteor_fragment"
    SLOW_AMONG_PEOPLE = "slow_among_people"

    @property
    def description(self) -> str:
        return _REASONS[self]


_REASONS = {
    Reason.TOO_BRIEF: "seen in too few frames",
    Reason.MOVED_TOO_LITTLE: "moved too little",
    Reason.TOO_LONG: "moved for too long for a meteor (a plane, satellite or person)",
    Reason.TOO_SLOW: "too slow for a meteor",
    Reason.NOT_STRAIGHT: "did not move in a straight line",
    Reason.REPEATS_IN_PLACE: "lights up the same spot again and again (a lamp, phone or person)",
    Reason.MOVES_IN_GROUP: "moves together with other things (a person or vehicle)",
    Reason.SLOW_TOO_BRIEF: "slow, and seen too briefly for a satellite or plane",
    Reason.SLOW_MOVED_TOO_LITTLE: "slow, and moved too little for a satellite or plane",
    Reason.SLOW_SPEED_RANGE: "outside the speed range of satellites and planes",
    Reason.SLOW_NOT_STRAIGHT: "slow, and did not move in a straight line",
    Reason.PART_OF_METEOR: "the slow tracker's view of a meteor",
    Reason.SAME_PLANE: "another light of a plane already counted",
    Reason.METEOR_FRAGMENT: "another piece of a meteor already counted (it broke up)",
    Reason.SLOW_AMONG_PEOPLE: "moves where people or lamps were seen moving (not across open sky)",
}


@dataclass
class Rejection:
    track_id: int
    scene: int
    start_frame: int
    end_frame: int
    reason: Reason


@dataclass
class SkyObject:
    """A meteor, satellite or plane: its detections in frame order, and what was measured about it."""

    kind: Kind
    track_id: int
    scene: int
    points: list[Point]
    fps: float
    straightness: float  # straight-line distance / path length
    number: int = 0  # in order of appearance, per kind
    rank: int = 0  # 1 = fastest, per kind
    length_rank: int = 0  # 1 = longest trail (meteors)
    brightness_rank: int = 0  # 1 = brightest (meteors, when brightness was measured)
    motion: AngularMotion | None = None  # set when the lens or the stars are known
    sky_path: SkyPath | None = None  # start and end among the stars, when the video was plate-solved
    shower: Membership | None = None  # the shower it belongs to; None is sporadic (or unknown without a date)
    brightness: Brightness | None = None
    diagnostics: dict[str, float | bool] = field(default_factory=dict)

    @property
    def label(self) -> str:
        return f"{self.kind.prefix}{self.number}"

    @property
    def start_frame(self) -> int:
        return self.points[0].frame

    @property
    def end_frame(self) -> int:
        return self.points[-1].frame

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

    @property
    def length_value(self) -> float:
        """What trails are ranked by: angular length when known, else on-screen length."""
        if self.motion and math.isfinite(self.motion.angular_length_deg):
            return self.motion.angular_length_deg
        return self.length_px

    @property
    def shower_code(self) -> str:
        """Three-letter IAU code, SPO for sporadic, or "" when showers were not looked for."""
        if self.shower is not None:
            return self.shower.shower.code
        return "SPO" if self.diagnostics.get("shower_checked") else ""

    @property
    def speed_value(self) -> float:
        """What objects are ranked by: angular speed when the lens is known, else on-screen speed."""
        if self.motion and math.isfinite(self.motion.angular_speed_deg_s):
            return self.motion.angular_speed_deg_s
        return self.speed_px_s
