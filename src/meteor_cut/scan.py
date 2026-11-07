"""First pass over the video: everything that moves, found by Sky Tracker's MotionTracker.

MotionTracker runs in "bright-diff" mode, which detects what got brighter since the previous frame: the head of a
meteor rather than the trail it leaves. A second MotionTracker sees only every few frames (about 5 a second), so things
too slow to stand out from one frame to the next, such as satellites and planes, move far enough between its frames to
be found. Scene cuts (common in compilations) change the whole picture at once, so each cut restarts both trackers and
starts a new scene.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable

import cv2
import numpy as np

from .licence import motion_tracker

# Mean change of a 160x90 grey thumbnail between frames, 0-255. A still night sky changes by about 0.2; cuts by 3-9.
CUT_THRESHOLD = 2.5
# Wider videos are scaled to this width for detection; coordinates are scaled back.
DETECTION_WIDTH = 1920
# The slow tracker's frame rate, and the offset that keeps its track ids apart from the fast tracker's.
SLOW_RATE_HZ = 5.0
SLOW_ID_OFFSET = 10_000_000


@dataclass
class Point:
    frame: int
    cx: float
    cy: float
    w: float
    h: float
    detected: bool  # False: the tracker's prediction for a frame without a detection
    peak: int = 0   # brightest grey value in the box, 0-255


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
    path: str
    fps: float
    width: int
    height: int
    frames: int
    cuts: list[int]  # first frame of every scene after the first
    tracks: list[Track]
    slow_tracks: list[Track] = field(default_factory=list)
    backgrounds: dict[int, np.ndarray] = field(default_factory=dict)  # each scene's darkest frame, for the images

    def scene_bounds(self, scene: int) -> tuple[int, int]:
        """First and last frame of a scene."""
        starts = [0, *self.cuts]
        start = starts[scene]
        end = (starts[scene + 1] - 1) if scene + 1 < len(starts) else self.frames - 1
        return start, end


def video_info(path: str) -> tuple[float, int, int, int]:
    capture = cv2.VideoCapture(path)
    if not capture.isOpened():
        raise ValueError(f"cannot open video {path}")
    fps = capture.get(cv2.CAP_PROP_FPS) or 0.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    capture.release()
    if not math.isfinite(fps) or fps <= 0:
        fps = 30.0
    return fps, width, height, frames


def _record(tracks: dict[int, Track], found, scene: int, index: int, scale: float, work: np.ndarray,
            id_offset: int = 0, slow: bool = False) -> None:
    for t in found:
        key = t.id + id_offset
        track = tracks.get(key)
        if track is None:
            track = tracks[key] = Track(key, scene, slow=slow)
        x0, y0 = max(0, int(t.bbox_x)), max(0, int(t.bbox_y))
        box = work[y0:int(t.bbox_y + t.bbox_h) + 1, x0:int(t.bbox_x + t.bbox_w) + 1]
        track.points.append(Point(index, t.cx / scale, t.cy / scale, t.bbox_w / scale, t.bbox_h / scale,
                                  detected=t.misses == 0, peak=int(box.max()) if box.size else 0))


def scan(path: str, *, sky_region: tuple[int, int, int, int] | None = None, diff_threshold: int = 25,
         min_area: float = 3.0, slow_objects: bool = True, progress: Callable[[float], None] | None = None) -> Scan:
    fps, width, height, total = video_info(path)
    scale = min(1.0, DETECTION_WIDTH / width) if width else 1.0
    region = None
    if sky_region:
        x, y, w, h = sky_region
        region = (round(x * scale), round(y * scale), round(w * scale), round(h * scale))
    tracker = motion_tracker(mode="bright-diff", moving_only=True, diff_threshold=diff_threshold,
                             min_area=min_area, region=region)
    step = max(1, round(fps / SLOW_RATE_HZ))
    slow = motion_tracker(mode="bright-diff", moving_only=True, diff_threshold=diff_threshold, min_area=min_area,
                          region=region, coast_frames=6) if slow_objects else None

    capture = cv2.VideoCapture(path)
    tracks: dict[int, Track] = {}
    slow_tracks: dict[int, Track] = {}
    cuts: list[int] = []
    previous_thumb: np.ndarray | None = None
    backgrounds: dict[int, np.ndarray] = {}
    darkest: dict[int, float] = {}
    scene = 0
    scene_start = 0
    index = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        thumb = cv2.resize(gray, (160, 90), interpolation=cv2.INTER_AREA).astype(np.float32)
        if previous_thumb is not None and float(np.mean(np.abs(thumb - previous_thumb))) > CUT_THRESHOLD:
            cuts.append(index)
            scene += 1
            scene_start = index
            tracker.reset()
            if slow:
                slow.reset()
        previous_thumb = thumb
        if (index - scene_start) % max(1, round(fps * 5)) == min(3, round(fps)):
            level = float(thumb.mean())
            if level < darkest.get(scene, float("inf")):
                darkest[scene], backgrounds[scene] = level, frame.copy()

        work = gray if scale == 1.0 else cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        _record(tracks, tracker.update(work, dt=1.0 / fps), scene, index, scale, work)
        if slow and (index - scene_start) % step == 0:
            _record(slow_tracks, slow.update(work, dt=step / fps), scene, index, scale, work, SLOW_ID_OFFSET, True)
        index += 1
        if progress and total and index % 24 == 0:
            progress(min(1.0, index / total))
    capture.release()
    if progress:
        progress(1.0)
    return Scan(path, fps, width, height, index, cuts, list(tracks.values()), list(slow_tracks.values()), backgrounds)
