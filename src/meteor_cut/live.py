"""Watching a camera live: keep only the meteors, instead of recording the whole night and processing it later.

Frames go through the same motion tracker and the same meteor rules as a recorded video. The last few seconds are
kept in memory (as JPEG, about 5 MB a second at 1080p), so a meteor's clip can start before it appeared. Lamps and
people are recognised by what else moves near them in the following seconds, so each candidate is decided
`hotspot_window_s` after it ends; then its clip is written, its brightness measured, its streak added to the night's
composite and a row appended to meteors.csv. Satellites and planes are not looked for: they take minutes to cross.

Works with a camera number (0 is the first webcam or capture card), a stream URL (rtsp://, http://) or a video file,
which is then read as fast as the computer allows, as if it were live.
"""

from __future__ import annotations

import csv
import json
import logging
import math
import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path

import cv2
import numpy as np

from .classify import Rules, find_meteors
from .composite import Composite
from .errors import VideoError
from .frames import Window, median_frame
from .model import Scan, SkyObject, Track
from .optics import Camera, measure_motion
from .photometry import Brightness, measure
from .scan import DETECTION_WIDTH, _record
from .text import length_text, speed_text
from .tracking import TrackerFactory, sky_tracker_factory

log = logging.getLogger(__name__)

LIVE_COLUMNS = (
    "number",
    "clock_time",
    "seconds_since_start",
    "duration_s",
    "start_x",
    "start_y",
    "end_x",
    "end_y",
    "length_px",
    "speed_px_s",
    "angular_length_deg",
    "angular_speed_deg_s",
    "sky_speed_km_s",
    "peak_brightness",
    "saturated",
    "clip",
)


@dataclass
class LiveMeteor:
    meteor: SkyObject
    time: datetime
    brightness: Brightness
    clip: Path | None


@dataclass
class WatchResult:
    out_dir: Path
    meteors: list[LiveMeteor] = field(default_factory=list)
    frames: int = 0
    seconds: float = 0.0
    files: list[Path] = field(default_factory=list)


def open_source(source: str) -> cv2.VideoCapture:
    """A camera number, a stream URL or a file."""
    capture = cv2.VideoCapture(int(source)) if source.isdigit() else cv2.VideoCapture(source)
    if not capture.isOpened():
        raise VideoError(f"cannot open {source}")
    return capture


class _Frames:
    """Frames from a live source in a thread of their own, so a slow moment never makes the camera drop frames
    inside the reader; files are read in step instead, as fast as they are processed."""

    def __init__(self, capture: cv2.VideoCapture, live: bool, stop: threading.Event):
        self.capture, self.live, self.stop = capture, live, stop
        self.queue: deque[np.ndarray | None] = deque(maxlen=256)
        self.ready = threading.Condition()
        if live:
            threading.Thread(target=self._pump, name="meteor-cut-capture", daemon=True).start()

    def _pump(self) -> None:
        while not self.stop.is_set():
            ok, frame = self.capture.read()
            with self.ready:
                self.queue.append(frame if ok else None)
                self.ready.notify()
            if not ok:
                return

    def next(self) -> np.ndarray | None:
        if not self.live:
            ok, frame = self.capture.read()
            return frame if ok else None
        with self.ready:
            while not self.queue and not self.stop.is_set():
                self.ready.wait(timeout=0.5)
            return self.queue.popleft() if self.queue else None


class Watcher:
    """Finds meteors in a live source and saves each one as it is confirmed."""

    def __init__(
        self,
        source: str,
        out_dir: str | Path,
        *,
        rules: Rules | None = None,
        camera_lens: Callable[[int, int], Camera | None] | None = None,
        altitude_km: float = 95.0,
        sky_region: tuple[int, int, int, int] | None = None,
        before_s: float = 1.5,
        after_s: float = 1.5,
        tracker_factory: TrackerFactory = sky_tracker_factory,
        on_meteor: Callable[[LiveMeteor], None] | None = None,
        clock: Callable[[], datetime] = lambda: datetime.now().astimezone(),
    ):
        self.source = source
        self.out = Path(out_dir)
        self.rules = rules or Rules()
        self.camera_lens = camera_lens
        self.altitude_km = altitude_km
        self.sky_region = sky_region
        self.before_s, self.after_s = before_s, after_s
        self.tracker_factory = tracker_factory
        self.on_meteor = on_meteor
        self.clock = clock

    def run(self, *, stop: threading.Event | None = None, max_seconds: float | None = None) -> WatchResult:
        stop = stop or threading.Event()
        capture = open_source(self.source)
        live = not Path(self.source).is_file()
        fps = capture.get(cv2.CAP_PROP_FPS)
        fps = fps if math.isfinite(fps) and 1 <= fps <= 240 else 30.0
        started = self.clock()
        self.out.mkdir(parents=True, exist_ok=True)
        (self.out / "clips").mkdir(exist_ok=True)
        result = WatchResult(self.out)
        table = (self.out / "meteors.csv").open("w", newline="", encoding="utf-8")
        rows = csv.writer(table)
        rows.writerow(LIVE_COLUMNS)

        frames = _Frames(capture, live, stop)
        first = frames.next()
        if first is None:
            capture.release()
            table.close()
            raise VideoError(f"no frames from {self.source}")
        height, width = first.shape[:2]
        camera = self.camera_lens(width, height) if self.camera_lens else None
        scale = min(1.0, DETECTION_WIDTH / width)
        region = tuple(round(v * scale) for v in self.sky_region) if self.sky_region else None
        tracker = self.tracker_factory(
            mode="bright-diff", moving_only=True, diff_threshold=25, min_area=3.0, region=region
        )
        decide = round((self.rules.hotspot_window_s + 0.5) * fps)
        keep = round((self.rules.hotspot_window_s + self.before_s + self.rules.max_duration_s + self.after_s + 2) * fps)
        ring: deque[tuple[int, bytes]] = deque(maxlen=keep)
        tracks: dict[int, Track] = {}
        decided: set[int] = set()
        composite: Composite | None = None
        number = 0

        def confirm(track: Track, now: int) -> None:
            nonlocal composite, number
            window_tracks = [
                t for t in tracks.values() if t.points and t.points[-1].frame >= track.points[0].frame - decide
            ]
            scan = Scan(self.source, fps, width, height, now + 1, [], window_tracks)
            meteors, _ = find_meteors(scan, camera, self.rules, self.altitude_km)
            meteor = next((m for m in meteors if m.track_id == track.id), None)
            if meteor is None:
                return
            number += 1
            meteor.number = number
            if camera is not None:
                meteor.motion = measure_motion(camera, meteor.start, meteor.end, meteor.duration_s, self.altitude_km)
            lo = meteor.start_frame - round(self.before_s * fps)
            hi = meteor.end_frame + round(self.after_s * fps)
            decoded = [
                (i, cv2.imdecode(np.frombuffer(d, np.uint8), cv2.IMREAD_COLOR)) for i, d in ring if lo <= i <= hi
            ]
            stretch: list[tuple[int, np.ndarray]] = [(i, f) for i, f in decoded if f is not None]
            if not stretch:
                return
            before = [f for i, f in stretch if i < meteor.start_frame - 1][-9:] or [stretch[0][1]]
            window = Window(stretch[0][0], [f for _, f in stretch], median_frame(before))
            brightness = measure(meteor, window)
            meteor.brightness = brightness
            if composite is None:
                composite = Composite(window.background, 0.0)
            composite.add(meteor, window)
            when = started + timedelta(seconds=meteor.start_s)
            clip = self.out / "clips" / f"meteor-{number:04d}-{when:%Y%m%d-%H%M%S}.mp4"
            _write_clip(clip, [f for _, f in stretch], fps)
            mo = meteor.motion
            rows.writerow(
                [
                    number,
                    when.isoformat(timespec="milliseconds"),
                    round(meteor.start_s, 3),
                    round(meteor.duration_s, 3),
                    round(meteor.start[0], 1),
                    round(meteor.start[1], 1),
                    round(meteor.end[0], 1),
                    round(meteor.end[1], 1),
                    round(meteor.length_px, 1),
                    round(meteor.speed_px_s, 1),
                    round(mo.angular_length_deg, 2) if mo else "",
                    round(mo.angular_speed_deg_s, 2) if mo else "",
                    round(mo.sky_speed_km_s, 1) if mo and mo.sky_speed_km_s else "",
                    round(brightness.peak, 4),
                    int(brightness.saturated),
                    clip.relative_to(self.out).as_posix(),
                ]
            )
            table.flush()
            found = LiveMeteor(meteor, when, brightness, clip)
            result.meteors.append(found)
            log.info(
                "meteor %d at %s: %s, %s", number, when.strftime("%H:%M:%S"), speed_text(meteor)[0], length_text(meteor)
            )
            if self.on_meteor:
                self.on_meteor(found)

        index = 0
        frame: np.ndarray | None = first
        clock_start = time.monotonic()
        try:
            while frame is not None and not stop.is_set():
                ok, data = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
                if ok:
                    ring.append((index, data.tobytes()))
                grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                work = (
                    grey if scale == 1.0 else cv2.resize(grey, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
                )
                _record(tracks, tracker.update(work, dt=1.0 / fps), 0, index, scale, work)
                for track in list(tracks.values()):
                    last = track.points[-1].frame
                    if track.id not in decided and index - last >= decide:
                        decided.add(track.id)
                        confirm(track, index)
                    if index - last > 3 * decide:
                        del tracks[track.id]
                        decided.discard(track.id)
                index += 1
                if max_seconds is not None and index / fps >= max_seconds:
                    break
                frame = frames.next()
            # The source ended or was stopped: decide on what had already ended. Anything still moving was cut off
            # and could pass for a meteor only because it was cut short.
            ended = index - round(0.3 * fps)
            for track in list(tracks.values()):
                if track.id not in decided and track.points[-1].frame < ended:
                    confirm(track, index)
        finally:
            stop.set()
            capture.release()
            table.close()
        result.frames = index
        result.seconds = time.monotonic() - clock_start
        result.files.append(self.out / "meteors.csv")
        result.files.extend(m.clip for m in result.meteors if m.clip)
        if composite is not None:
            cv2.imwrite(str(self.out / "composite.png"), composite.image())
            result.files.append(self.out / "composite.png")
        summary = {
            "source": self.source,
            "started": started.isoformat(timespec="seconds"),
            "seconds_watched": round(index / fps, 1),
            "meteors": len(result.meteors),
            "brightest": max(result.meteors, key=lambda m: m.brightness.peak).meteor.number if result.meteors else None,
        }
        (self.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        result.files.append(self.out / "summary.json")
        return result


def _write_clip(path: Path, frames: list[np.ndarray], fps: float) -> None:
    import imageio_ffmpeg

    if not frames:
        return
    height, width = frames[0].shape[:2]
    writer = imageio_ffmpeg.write_frames(
        str(path),
        (width, height),
        fps=fps,
        codec="libx264",
        pix_fmt_in="bgr24",
        pix_fmt_out="yuv420p",
        macro_block_size=1,
        output_params=["-crf", "20", "-preset", "veryfast", "-movflags", "+faststart"],
    )
    writer.send(None)
    try:
        for frame in frames:
            writer.send(np.ascontiguousarray(frame))
    finally:
        writer.close()
