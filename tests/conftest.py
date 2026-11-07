"""Shared test helpers: a stand-in for Sky Tracker, synthetic tracks, and a small synthetic night video."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar

import cv2
import numpy as np
import pytest

from meteor_cut.model import Point, Scan, Track

FPS = 25.0
WIDTH, HEIGHT = 1920, 1080


# ── a stand-in for Sky Tracker's MotionTracker ────────────────────────────────


@dataclass
class FakeTrack:
    id: int
    cx: float
    cy: float
    bbox_x: float
    bbox_y: float
    bbox_w: float
    bbox_h: float
    misses: int = 0


class FakeTracker:
    """Finds bright blobs and links them frame to frame by distance. Enough to exercise everything around it."""

    created: ClassVar[list[FakeTracker]] = []

    def __init__(self, **options: Any):
        self.options = options
        self.next_id = 1
        self.last: dict[int, tuple[float, float]] = {}
        self.resets = 0
        self.frames = 0
        FakeTracker.created.append(self)

    def update(self, frame: np.ndarray, dt: float = -1.0) -> list[FakeTrack]:
        self.frames += 1
        x0, y0 = 0, 0
        if self.options.get("region"):
            x0, y0, w, h = self.options["region"]
            frame = frame[y0 : y0 + h, x0 : x0 + w]
        count, _, stats, centroids = cv2.connectedComponentsWithStats((frame > 200).astype(np.uint8))
        found: list[FakeTrack] = []
        seen: dict[int, tuple[float, float]] = {}
        for i in range(1, count):
            cx, cy = centroids[i][0] + x0, centroids[i][1] + y0
            near = [(math.dist((cx, cy), p), tid) for tid, p in self.last.items() if tid not in seen]
            tid = min(near)[1] if near and min(near)[0] < 120 else self._new_id()
            seen[tid] = (cx, cy)
            x, y, w, h = stats[i][:4]
            found.append(FakeTrack(tid, cx, cy, x + x0, y + y0, w, h))
        self.last = seen
        return sorted(found, key=lambda t: t.id)

    def _new_id(self) -> int:
        self.next_id += 1
        return self.next_id - 1

    def reset(self) -> None:
        self.resets += 1
        self.last = {}


@pytest.fixture
def fake_tracker() -> type[FakeTracker]:
    FakeTracker.created = []
    return FakeTracker


# ── synthetic tracks ──────────────────────────────────────────────────────────


def line(x0: float, y0: float, dx: float, dy: float, n: int) -> list[tuple[float, float]]:
    return [(x0 + dx * i, y0 + dy * i) for i in range(n)]


def make_track(
    track_id: int,
    points: list[tuple[float, float]],
    scene: int = 0,
    start: int = 0,
    step: int = 1,
    peaks: list[int] | None = None,
    slow: bool = False,
) -> Track:
    pts = [
        Point(start + i * step, x, y, 6, 6, True, (peaks or [200] * len(points))[i]) for i, (x, y) in enumerate(points)
    ]
    return Track(track_id, scene, pts, slow=slow)


def make_scan(*tracks: Track, slow: tuple[Track, ...] = (), cuts: tuple[int, ...] = (), frames: int = 2000) -> Scan:
    return Scan("video.mp4", FPS, WIDTH, HEIGHT, frames, list(cuts), list(tracks), list(slow))


# ── a synthetic night video ───────────────────────────────────────────────────


@dataclass(frozen=True)
class NightVideo:
    path: Path
    fps: float
    width: int
    height: int
    frames: int
    meteor_frames: tuple[int, int]
    cut: int


@pytest.fixture(scope="session")
def night_video(tmp_path_factory: pytest.TempPathFactory) -> NightVideo:
    """8 s at 25 fps, 640x360: a meteor at 1.0-1.4 s, a satellite crossing slowly, and a scene cut at 6 s."""
    fps, width, height, frames, cut = 25.0, 640, 360, 200, 150
    path = tmp_path_factory.mktemp("video") / "night.avi"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), fps, (width, height))
    for i in range(frames):
        frame = np.full((height, width, 3), 12 if i < cut else 40, np.uint8)
        if 25 <= i <= 35:  # meteor: 24 px a frame, down and to the right
            k = i - 25
            cv2.circle(frame, (100 + 24 * k, 60 + 10 * k), 3, (255, 255, 255), -1)
        if i < cut:  # satellite: 1.2 px a frame across the top
            cv2.circle(frame, (40 + int(1.2 * i), 300), 2, (255, 255, 255), -1)
        writer.write(frame)
    writer.release()
    return NightVideo(path, fps, width, height, frames, (25, 35), cut)
