"""Second pass: the highlight video, made only of the moments with meteors (and, if asked, satellites and planes).

Layout (1920x1080 by default): the video on the left, scaled to fit, with a header (source time, what is on screen)
and a footer (totals); on the right a panel listing every meteor sorted by speed, with the ones on screen highlighted.
Each object gets a box at its head, its trail, and a label with its number and speed. Trails of what has already
passed stay faintly on screen until the scene changes. The video can end on still images (trajectories, heat map).
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

from .model import Kind, Scan, SkyObject
from .text import Timeline, ascii_text, clock, font, speed_text

log = logging.getLogger(__name__)

CYAN = (131, 232, 250)  # RGB
WHITE = (240, 244, 245)
MUTED = (140, 152, 158)
BG = (6, 9, 12)
PANEL = (12, 17, 21)
KIND_RGB = {Kind.METEOR: CYAN, Kind.SATELLITE: (120, 230, 140), Kind.PLANE: (255, 170, 80)}
KIND_NAME = {Kind.METEOR: "", Kind.SATELLITE: "satellite ", Kind.PLANE: "plane "}


@dataclass(frozen=True)
class Style:
    """How the highlight video looks and what it shows."""

    width: int = 1920
    height: int = 1080
    panel_width: int = 480
    bar_height: int = 120  # header and footer around the video
    before_s: float = 0.7  # video kept before an object appears...
    after_s: float = 1.0  # ...and after it ends
    merge_gap_s: float = 0.6  # clips closer than this are joined
    box_padding: int = 14
    show: frozenset[Kind] = field(default_factory=lambda: frozenset({Kind.METEOR}))
    trails: bool = True  # keep past trails faintly on screen
    end_card_s: float = 4.0  # seconds for each closing image
    # Satellites and planes take half a minute or more to cross; the video keeps this many seconds from the middle of
    # each pass (their whole path is in the tables and on the trajectories picture). None keeps whole passes.
    pass_clip_s: float | None = 3.0
    crf: int = 20  # H.264 quality: lower is better and larger


def _span(obj: SkyObject, style: Style, fps: float) -> tuple[int, int]:
    """The frames of an object to show: all of a meteor, the middle of a long satellite or plane pass."""
    if obj.kind is Kind.METEOR or style.pass_clip_s is None or obj.duration_s <= style.pass_clip_s:
        return obj.start_frame - round(style.before_s * fps), obj.end_frame + round(style.after_s * fps)
    middle, half = (obj.start_frame + obj.end_frame) // 2, round(style.pass_clip_s * fps / 2)
    return middle - half, middle + half


def clips(scan: Scan, objects: Sequence[SkyObject], style: Style) -> list[tuple[int, int]]:
    """Frame ranges to keep, inside each object's scene, merged when they nearly touch."""
    ranges = sorted(
        (max(scan.scene_bounds(o.scene)[0], start), min(scan.scene_bounds(o.scene)[1], end))
        for o in objects
        for start, end in [_span(o, style, scan.fps)]
    )
    merged: list[list[int]] = []
    gap = round(style.merge_gap_s * scan.fps)
    for start, end in ranges:
        if merged and start <= merged[-1][1] + gap and scan.scene_of(start) == scan.scene_of(merged[-1][1]):
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return [(a, b) for a, b in merged]


def _bgr(rgb: tuple[int, int, int], factor: float = 1.0) -> tuple[int, int, int]:
    r, g, b = rgb
    return int(b * factor), int(g * factor), int(r * factor)


def _to_bgr(image: Image.Image) -> np.ndarray:
    return cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)


class Renderer:
    """Draws one output frame at a time. Frames must come in source order: trails build up as the video plays."""

    def __init__(
        self,
        scan: Scan,
        meteors: Sequence[SkyObject],
        others: Sequence[SkyObject] = (),
        style: Style | None = None,
        timeline: Timeline | None = None,
    ):
        self.scan = scan
        self.style = style = style or Style()
        self.meteors = list(meteors)
        self.others = list(others)
        self.objects = [o for o in [*self.meteors, *self.others] if o.kind in style.show]
        self.by_speed = sorted(self.meteors, key=lambda m: m.rank)
        area_w = style.width - style.panel_width
        area_h = style.height - 2 * style.bar_height
        self.scale = min(area_w / scan.width, area_h / scan.height)
        self.video_w, self.video_h = round(scan.width * self.scale), round(scan.height * self.scale)
        self.video_x = (area_w - self.video_w) // 2
        self.video_y = style.bar_height + (area_h - self.video_h) // 2
        self.linger = round(0.5 * scan.fps)
        self.long = scan.duration_s >= 3600  # times as H:MM:SS throughout
        self.timeline = timeline or Timeline(None, self.long)
        self._panel_cache: dict[tuple[int, ...], np.ndarray] = {}
        self._trails = np.zeros((self.video_h, self.video_w, 3), np.uint8)
        self._trailed: set[tuple[Kind, int]] = set()
        self._trail_scene = -1
        self.footer = self._footer_text()

    def _footer_text(self) -> str:
        counts = [f"{len(self.meteors)} meteors"]
        for kind in (Kind.SATELLITE, Kind.PLANE):
            n = sum(1 for o in self.others if o.kind is kind)
            if n:
                counts.append(f"{n} {kind.value}s")
        duration = self.scan.duration_s
        rate = f"  ·  {len(self.meteors) / duration * 3600:.0f} meteors per hour" if duration >= 600 else ""
        fastest = self.by_speed[0] if self.by_speed else None
        best = f"  ·  fastest {fastest.label} {speed_text(fastest)[0]}" if fastest else ""
        return ", ".join(counts) + f" in {clock(duration)}{rate}{best}"

    def active(self, frame: int) -> list[SkyObject]:
        return [o for o in self.objects if o.start_frame <= frame <= o.end_frame + self.linger]

    def frame(self, image: np.ndarray, index: int) -> np.ndarray:
        s = self.style
        canvas = np.empty((s.height, s.width, 3), np.uint8)
        canvas[:] = _bgr(BG)
        video = cv2.resize(image, (self.video_w, self.video_h), interpolation=cv2.INTER_AREA)
        if s.trails:
            self._update_trails(index)
            np.maximum(video, self._trails, out=video)
        active = self.active(index)
        for obj in active:
            self._draw_object(video, obj, index)
        canvas[self.video_y : self.video_y + self.video_h, self.video_x : self.video_x + self.video_w] = video
        meteors_on = tuple(sorted(o.number for o in active if o.kind is Kind.METEOR))
        canvas[:, s.width - s.panel_width :] = self._panel(meteors_on)
        self._bars(canvas, index, active)
        return canvas

    def card(self, image: np.ndarray) -> np.ndarray:
        """A closing still, letterboxed to the video size."""
        s = self.style
        canvas = np.empty((s.height, s.width, 3), np.uint8)
        canvas[:] = _bgr(BG)
        k = min(s.width / image.shape[1], s.height / image.shape[0])
        w, h = round(image.shape[1] * k), round(image.shape[0] * k)
        x, y = (s.width - w) // 2, (s.height - h) // 2
        canvas[y : y + h, x : x + w] = cv2.resize(image, (w, h), interpolation=cv2.INTER_AREA)
        return canvas

    def _update_trails(self, index: int) -> None:
        scene = self.scan.scene_of(index)
        if scene != self._trail_scene:
            self._trails[:] = 0
            self._trailed.clear()
            self._trail_scene = scene
        k = self.scale
        for obj in self.objects:
            key = (obj.kind, obj.number)
            if obj.scene == scene and key not in self._trailed and index > obj.end_frame + self.linger:
                pts = np.array([[p.cx * k, p.cy * k] for p in obj.points], np.int32)
                if len(pts) > 1:
                    cv2.polylines(self._trails, [pts], False, _bgr(KIND_RGB[obj.kind], 0.45), 1, cv2.LINE_AA)
                self._trailed.add(key)

    def _draw_object(self, video: np.ndarray, obj: SkyObject, index: int) -> None:
        k = self.scale
        seen = [p for p in obj.points if p.frame <= index]
        if not seen:
            return
        colour = _bgr(KIND_RGB[obj.kind], 0.55 if index > obj.end_frame else 1.0)
        trail = np.array([[p.cx * k, p.cy * k] for p in seen], np.int32)
        if len(trail) > 1:
            cv2.polylines(video, [trail], False, colour, 2, cv2.LINE_AA)
        head = seen[-1]
        pad = self.style.box_padding
        x0, y0 = int(head.cx * k - head.w * k / 2 - pad), int(head.cy * k - head.h * k / 2 - pad)
        x1, y1 = int(head.cx * k + head.w * k / 2 + pad), int(head.cy * k + head.h * k / 2 + pad)
        cv2.rectangle(video, (x0, y0), (x1, y1), colour, 2, cv2.LINE_AA)
        main, extra = speed_text(obj)
        label = ascii_text(f"{KIND_NAME[obj.kind]}{obj.label}  {main}" + (f"  {extra}" if extra else ""))
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        lx = min(max(0, x0), video.shape[1] - tw - 8)
        ly = y0 - 8 if y0 - th - 14 > 0 else y1 + th + 10
        cv2.rectangle(video, (lx - 4, ly - th - 6), (lx + tw + 4, ly + 6), _bgr(PANEL), -1)
        cv2.putText(video, label, (lx, ly), cv2.FONT_HERSHEY_SIMPLEX, 0.6, colour, 1, cv2.LINE_AA)

    def _panel(self, meteors_on: tuple[int, ...]) -> np.ndarray:
        cached = self._panel_cache.get(meteors_on)
        if cached is not None:
            return cached
        s = self.style
        img = Image.new("RGB", (s.panel_width, s.height), PANEL)
        draw = ImageDraw.Draw(img)
        draw.text((28, 34), "Meteors by speed", font=font(30, bold=True), fill=WHITE)
        unit = "angular speed" if any(m.motion for m in self.meteors) else "on-screen speed"
        draw.text((28, 76), f"{len(self.meteors)} found · fastest first · {unit}", font=font(19), fill=MUTED)
        top, row_h = 124, 32
        rows = (s.height - top - 24) // row_h
        first = 0  # scroll so the meteors on screen are visible
        ranks = [m.rank for m in self.by_speed if m.number in meteors_on]
        if ranks and max(ranks) > rows:
            first = min(max(ranks) - rows // 2, len(self.by_speed) - rows)
        for i, m in enumerate(self.by_speed[first : first + rows]):
            y = top + i * row_h
            on = m.number in meteors_on
            if on:
                draw.rounded_rectangle((16, y - 4, s.panel_width - 16, y + row_h - 6), 6, fill=(22, 52, 60))
            main, extra = speed_text(m)
            row_font = font(19, bold=on)
            draw.text((28, y), f"{m.rank:>2}", font=row_font, fill=CYAN if on else MUTED)
            draw.text((72, y), m.label, font=row_font, fill=CYAN if on else WHITE)
            draw.text((140, y), self.timeline.label(m.start_s), font=font(19), fill=MUTED)
            draw.text((250, y), main, font=row_font, fill=CYAN if on else WHITE)
            if extra:
                draw.text((356, y), extra, font=font(19), fill=CYAN if on else MUTED)
        panel = _to_bgr(img)
        if len(self._panel_cache) > 256:
            self._panel_cache.clear()
        self._panel_cache[meteors_on] = panel
        return panel

    def _bars(self, canvas: np.ndarray, index: int, active: Sequence[SkyObject]) -> None:
        s = self.style
        area_w = s.width - s.panel_width
        header = Image.new("RGB", (area_w, s.bar_height), BG)
        draw = ImageDraw.Draw(header)
        draw.text((32, 30), "Meteor Cut", font=font(30, bold=True), fill=WHITE)
        draw.text((32, 72), "tracked with Sky Tracker", font=font(19), fill=MUTED)
        now = self.timeline.label(index / self.scan.fps)
        if self.timeline.start is None:
            now = f"source {now}"
        if active:
            now += "  ·  " + "  ".join(f"{o.label} {speed_text(o)[0]}" for o in active[:4])
        draw.text((area_w - 32, 44), now, font=font(24), fill=CYAN if active else MUTED, anchor="ra")
        canvas[: s.bar_height, :area_w] = _to_bgr(header)
        footer = Image.new("RGB", (area_w, s.bar_height), BG)
        ImageDraw.Draw(footer).text((32, 44), self.footer, font=font(24), fill=MUTED)
        canvas[s.height - s.bar_height :, :area_w] = _to_bgr(footer)


def write_highlights(
    scan: Scan,
    meteors: Sequence[SkyObject],
    out_path: str | Path,
    *,
    others: Sequence[SkyObject] = (),
    style: Style | None = None,
    end_cards: Sequence[np.ndarray] = (),
    timeline: Timeline | None = None,
    progress: Callable[[float], None] | None = None,
) -> list[tuple[int, int]]:
    """Writes the highlight video (H.264 MP4) and returns the source frame ranges used. Nothing is written when
    there is nothing to show."""
    import imageio_ffmpeg

    style = style or Style()
    renderer = Renderer(scan, meteors, others, style, timeline)
    ranges = clips(scan, renderer.objects, style)
    if not ranges:
        return ranges
    card_frames = round(style.end_card_s * scan.fps)
    total = sum(b - a + 1 for a, b in ranges) + card_frames * len(end_cards)
    writer = imageio_ffmpeg.write_frames(
        str(out_path),
        (style.width, style.height),
        fps=scan.fps,
        codec="libx264",
        pix_fmt_in="bgr24",
        pix_fmt_out="yuv420p",
        macro_block_size=1,
        output_params=["-crf", str(style.crf), "-preset", "medium", "-movflags", "+faststart"],
    )
    writer.send(None)
    written = 0

    def send(frame: np.ndarray) -> None:
        nonlocal written
        writer.send(np.ascontiguousarray(frame))
        written += 1
        if progress and written % 12 == 0:
            progress(written / total)

    capture = cv2.VideoCapture(scan.path)
    try:
        _send_clips(capture, ranges, scan.fps, lambda image, index: send(renderer.frame(image, index)))
        for card in end_cards:
            still = renderer.card(card)
            for _ in range(card_frames):
                send(still)
    finally:
        capture.release()
        writer.close()
    if progress:
        progress(1.0)
    log.info("wrote %d frames to %s", written, out_path)
    return ranges


def _send_clips(
    capture: cv2.VideoCapture, ranges: Sequence[tuple[int, int]], fps: float, emit: Callable[[np.ndarray, int], None]
) -> None:
    """Feeds every frame inside `ranges` to `emit`, in order, without decoding the long gaps between them."""
    far = round(20 * fps)
    index = 0
    queue = iter(ranges)
    current: tuple[int, int] | None = next(queue, None)
    while current is not None:
        while current is not None and index > current[1]:
            current = next(queue, None)
        if current is None:
            break
        if current[0] - index > far:
            # Jump close to the next clip; land a little early and read on, so frame numbers stay exact.
            target = current[0] - round(2 * fps)
            if capture.set(cv2.CAP_PROP_POS_FRAMES, target) and int(capture.get(cv2.CAP_PROP_POS_FRAMES)) == target:
                index = target
            else:
                capture.set(cv2.CAP_PROP_POS_FRAMES, index)
                far = 1 << 62  # this video cannot seek reliably: decode straight through
        if index < current[0]:
            if not capture.grab():
                break
            index += 1
            continue
        ok, image = capture.read()
        if not ok:
            break
        emit(image, index)
        index += 1
