"""Reading short stretches of a long video: the frames around each meteor, and quiet frames of the sky.

The first pass keeps only positions; brightness, the composite picture and the vertical clips need the pixels again,
for a second or two around each meteor. Seeking in H.264 lands on a keyframe, so the reader lands a little early and
decodes forward to the exact frame, and decodes straight on when the next stretch is near.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .errors import VideoError
from .model import SkyObject


class FrameReader:
    """Frame-exact random access, cheapest when stretches are read in order."""

    def __init__(self, path: str | Path):
        self.capture = cv2.VideoCapture(str(path))
        if not self.capture.isOpened():
            raise VideoError(f"cannot open {path}")
        self.fps = self.capture.get(cv2.CAP_PROP_FPS) or 25.0
        self.index = 0  # the frame the next read() returns
        self._seekable = True

    def close(self) -> None:
        self.capture.release()

    def __enter__(self) -> FrameReader:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def _go_to(self, frame: int) -> None:
        far = round(10 * self.fps)
        if self._seekable and (frame < self.index or frame - self.index > far):
            target = max(0, frame - round(2 * self.fps))
            if (
                self.capture.set(cv2.CAP_PROP_POS_FRAMES, target)
                and int(self.capture.get(cv2.CAP_PROP_POS_FRAMES)) == target
            ):
                self.index = target
            else:
                self._seekable = False
        if frame < self.index:  # cannot seek: start over
            self.capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
            self.index = 0
        while self.index < frame:
            if not self.capture.grab():
                raise VideoError(f"the video ends before frame {frame}")
            self.index += 1

    def read(self, first: int, last: int) -> list[np.ndarray]:
        """Frames first..last (inclusive); fewer when the video ends sooner."""
        self._go_to(max(0, first))
        frames = []
        while self.index <= last:
            ok, image = self.capture.read()
            if not ok:
                break
            frames.append(image)
            self.index += 1
        return frames


def median_frame(frames: Iterable[np.ndarray]) -> np.ndarray:
    """Per-pixel median, float32: what stays put (sky, stars, landscape) without what passes (meteors, noise)."""
    stack = np.stack([np.asarray(f, np.float32) for f in frames])
    median: np.ndarray = np.median(stack, axis=0).astype(np.float32)
    return median


@dataclass
class Window:
    """The frames around one object, and the sky just before it."""

    first: int  # frame number of frames[0]
    frames: list[np.ndarray]  # BGR uint8
    background: np.ndarray  # BGR float32: median of the frames before the object appeared

    def frame(self, number: int) -> np.ndarray | None:
        i = number - self.first
        return self.frames[i] if 0 <= i < len(self.frames) else None


def windows(
    reader: FrameReader, objects: Iterable[SkyObject], *, background_frames: int = 9, after: int = 3
) -> Iterator[tuple[SkyObject, Window]]:
    """Each object's frames (in order of appearance) with a background from the frames just before it."""
    for obj in sorted(objects, key=lambda o: o.start_frame):
        first = max(0, obj.start_frame - background_frames - 1)
        frames = reader.read(first, obj.end_frame + after)
        before = frames[: max(1, obj.start_frame - 1 - first)]
        yield obj, Window(first, frames, median_frame(before))
