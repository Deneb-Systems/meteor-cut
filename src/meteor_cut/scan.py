"""First pass over the video: everything that moves, found by Sky Tracker's MotionTracker.

MotionTracker runs in "bright-diff" mode, which detects what got brighter since the previous frame: the head of a
meteor rather than the trail it leaves. A second MotionTracker sees only every few frames (about 5 a second), so things
too slow to stand out from one frame to the next, such as satellites and planes, move far enough between its frames to
be found. Scene cuts (common in compilations) change the whole picture at once, so each cut restarts both trackers and
starts a new scene.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Callable, Sequence
from pathlib import Path

import cv2
import numpy as np

from .errors import VideoError
from .model import Point, Scan, Track, VideoInfo
from .tracking import MotionTrack, MotionTracker, TrackerFactory, sky_tracker_factory

__all__ = [
    "CUT_THRESHOLD",
    "DETECTION_WIDTH",
    "SLOW_ID_OFFSET",
    "SLOW_RATE_HZ",
    "Point",
    "Scan",
    "Track",
    "probe",
    "scan",
]

log = logging.getLogger(__name__)

# Mean change of a 160x90 grey thumbnail between frames, 0-255. A still night sky changes by about 0.2; cuts by 3-9.
CUT_THRESHOLD = 2.5
# Wider videos are scaled to this width for detection; coordinates are scaled back.
DETECTION_WIDTH = 1920
# The slow tracker's frame rate, and the offset that keeps its track ids apart from the fast tracker's.
SLOW_RATE_HZ = 5.0
SLOW_ID_OFFSET = 10_000_000
# How often a scene's darkest frame is looked for (seconds), for the still images.
BACKGROUND_EVERY_S = 5.0


def probe(path: str | Path) -> VideoInfo:
    """Frame rate, size and frame count, as the container reports them."""
    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            raise VideoError(f"cannot open the video {path}")
        fps = capture.get(cv2.CAP_PROP_FPS) or 0.0
        info = VideoInfo(
            fps=fps if math.isfinite(fps) and fps > 0 else 30.0,
            width=int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
            height=int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            frames=int(capture.get(cv2.CAP_PROP_FRAME_COUNT)),
        )
    finally:
        capture.release()
    if info.width <= 0 or info.height <= 0:
        raise VideoError(f"the video {path} has no picture")
    return info


def _record(
    tracks: dict[int, Track],
    found: Sequence[MotionTrack],
    scene: int,
    index: int,
    scale: float,
    work: np.ndarray,
    id_offset: int = 0,
    slow: bool = False,
) -> None:
    for t in found:
        key = t.id + id_offset
        track = tracks.get(key)
        if track is None:
            track = tracks[key] = Track(key, scene, slow=slow)
        x0, y0 = max(0, int(t.bbox_x)), max(0, int(t.bbox_y))
        box = work[y0 : int(t.bbox_y + t.bbox_h) + 1, x0 : int(t.bbox_x + t.bbox_w) + 1]
        track.points.append(
            Point(
                index,
                t.cx / scale,
                t.cy / scale,
                t.bbox_w / scale,
                t.bbox_h / scale,
                detected=t.misses == 0,
                peak=int(box.max()) if box.size else 0,
            )
        )


def scan(
    path: str | Path,
    *,
    sky_region: tuple[int, int, int, int] | None = None,
    diff_threshold: int = 25,
    min_area: float = 3.0,
    slow_objects: bool = True,
    tracker_factory: TrackerFactory = sky_tracker_factory,
    progress: Callable[[float], None] | None = None,
) -> Scan:
    """Decodes the whole video once and returns every track the two motion trackers followed.

    sky_region (x, y, width, height in source pixels) limits detection to part of the frame.
    """
    info = probe(path)
    scale = min(1.0, DETECTION_WIDTH / info.width)
    region = tuple(round(v * scale) for v in sky_region) if sky_region else None
    options = {
        "mode": "bright-diff",
        "moving_only": True,
        "diff_threshold": diff_threshold,
        "min_area": min_area,
        "region": region,
    }
    fast = tracker_factory(**options)
    step = max(1, round(info.fps / SLOW_RATE_HZ))
    slow: MotionTracker | None = tracker_factory(**options, coast_frames=6) if slow_objects else None

    tracks: dict[int, Track] = {}
    slow_tracks: dict[int, Track] = {}
    cuts: list[int] = []
    backgrounds: dict[int, np.ndarray] = {}
    darkest: dict[int, float] = {}
    previous_thumb: np.ndarray | None = None
    scene = scene_start = index = 0
    background_every = max(1, round(info.fps * BACKGROUND_EVERY_S))

    capture = cv2.VideoCapture(str(path))
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
            thumb = cv2.resize(gray, (160, 90), interpolation=cv2.INTER_AREA).astype(np.float32)
            if previous_thumb is not None and float(np.mean(np.abs(thumb - previous_thumb))) > CUT_THRESHOLD:
                cuts.append(index)
                scene += 1
                scene_start = index
                fast.reset()
                if slow:
                    slow.reset()
            previous_thumb = thumb
            if (index - scene_start) % background_every == min(3, round(info.fps)):
                level = float(thumb.mean())
                if level < darkest.get(scene, math.inf):
                    darkest[scene], backgrounds[scene] = level, frame.copy()

            work = gray if scale == 1.0 else cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            _record(tracks, fast.update(work, dt=1.0 / info.fps), scene, index, scale, work)
            if slow and (index - scene_start) % step == 0:
                _record(
                    slow_tracks,
                    slow.update(work, dt=step / info.fps),
                    scene,
                    index,
                    scale,
                    work,
                    SLOW_ID_OFFSET,
                    slow=True,
                )
            index += 1
            if progress and info.frames and index % 24 == 0:
                progress(min(1.0, index / info.frames))
    finally:
        capture.release()
    if index == 0:
        raise VideoError(f"no frames could be read from {path}")
    if progress:
        progress(1.0)
    log.info(
        "scanned %d frames: %d scenes, %d fast and %d slow tracks", index, len(cuts) + 1, len(tracks), len(slow_tracks)
    )
    return Scan(
        str(path),
        info.fps,
        info.width,
        info.height,
        index,
        cuts,
        list(tracks.values()),
        list(slow_tracks.values()),
        backgrounds,
    )
