"""Vertical clips (1080x1920, 9:16) of the best meteors, for Shorts, Reels and TikTok.

Each clip shows a close-up of the meteor's path above the whole frame, with the meteor's details as titles: it plays
at normal speed, then again at a quarter speed, so a meteor that lasts half a second can be seen. A phone screen is
narrow, so the close-up follows the whole path (never cutting it off) and the full frame below shows where in the sky
it was.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

from .frames import FrameReader
from .model import SkyObject
from .render import BG, CYAN, MUTED, WHITE, _bgr, _to_bgr
from .text import Timeline, font, length_text, magnitude_text, speed_text

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class ShortStyle:
    width: int = 1080
    height: int = 1920
    title_height: int = 200
    before_s: float = 1.2
    after_s: float = 1.5
    slow_factor: int = 4  # the replay shows each frame this many times
    min_zoom_side: int = 300  # source pixels: closest the close-up gets
    crf: int = 20


def best(meteors: Sequence[SkyObject], count: int) -> list[SkyObject]:
    """The meteors most worth a clip: brightest first when brightness is known, then longest."""

    def key(m: SkyObject) -> tuple[int, float]:
        rank = m.brightness_rank or (len(meteors) + 1)
        return rank, -m.length_value

    return sorted(meteors, key=key)[:count]


def _zoom_box(obj: SkyObject, width: int, height: int, style: ShortStyle) -> tuple[int, int, int]:
    """Square crop (x, y, side) in source pixels holding the whole path with a margin."""
    xs = [p.cx for p in obj.points]
    ys = [p.cy for p in obj.points]
    side = max(max(xs) - min(xs), max(ys) - min(ys)) * 1.5 + 60
    side = int(min(max(side, style.min_zoom_side), height, width))
    cx, cy = (max(xs) + min(xs)) / 2, (max(ys) + min(ys)) / 2
    x = int(min(max(cx - side / 2, 0), width - side))
    y = int(min(max(cy - side / 2, 0), height - side))
    return x, y, side


def _title(obj: SkyObject, timeline: Timeline, style: ShortStyle, subtitle_extra: str) -> np.ndarray:
    image = Image.new("RGB", (style.width, style.title_height), BG)
    draw = ImageDraw.Draw(image)
    name = f"{obj.shower.shower.name[:-1]} meteor" if obj.shower else "Meteor"
    draw.text((56, 30), f"{name} {obj.label}", font=font(64, bold=True), fill=WHITE)
    draw.text((style.width - 56, 124), "Meteor Cut · by Deneb Systems", font=font(26), fill=MUTED, anchor="ra")
    when = timeline.at(obj.start_s)
    stamp = when.strftime("%d %b %Y · %H:%M:%S") if when else f"at {timeline.label(obj.start_s)} in the video"
    draw.text((58, 118), stamp + subtitle_extra, font=font(34), fill=MUTED)
    return _to_bgr(image)


def _facts(obj: SkyObject) -> str:
    main, km = speed_text(obj)
    parts = [main, km, length_text(obj) + " long", magnitude_text(obj)]
    if obj.brightness and obj.brightness.fireball:
        parts.append("fireball")
    return "  ·  ".join(p for p in parts if p)


class _Layout:
    def __init__(self, obj: SkyObject, width: int, height: int, timeline: Timeline, style: ShortStyle, extra: str):
        self.style = style
        self.zoom = _zoom_box(obj, width, height, style)
        self.zoom_y = style.title_height
        self.zoom_side = style.width
        self.full_w = style.width
        self.full_h = round(height * style.width / width)
        self.full_y = self.zoom_y + self.zoom_side + 16
        self.title = _title(obj, timeline, style, extra)
        self.facts = _facts(obj)
        self.scale = style.width / width

    def frame(self, image: np.ndarray, slow: bool) -> np.ndarray:
        s = self.style
        canvas = np.empty((s.height, s.width, 3), np.uint8)
        canvas[:] = _bgr(BG)
        canvas[: s.title_height] = self.title
        x, y, side = self.zoom
        crop = image[y : y + side, x : x + side]
        canvas[self.zoom_y : self.zoom_y + self.zoom_side] = cv2.resize(
            crop, (self.zoom_side, self.zoom_side), interpolation=cv2.INTER_CUBIC
        )
        full = cv2.resize(image, (self.full_w, self.full_h), interpolation=cv2.INTER_AREA)
        k = self.scale
        cv2.rectangle(full, (int(x * k), int(y * k)), (int((x + side) * k), int((y + side) * k)), _bgr(CYAN), 2)
        bottom = min(s.height, self.full_y + self.full_h)
        canvas[self.full_y : bottom] = full[: bottom - self.full_y]
        caption = Image.new("RGB", (s.width, 76), BG)
        draw = ImageDraw.Draw(caption)
        draw.text((56, 14), self.facts, font=font(34, bold=True), fill=CYAN)
        if slow:
            draw.text((s.width - 56, 14), "¼ speed", font=font(34), fill=WHITE, anchor="ra")
        top = self.zoom_y + self.zoom_side - 76
        canvas[top : top + 76] = cv2.addWeighted(canvas[top : top + 76], 0.25, _to_bgr(caption), 0.75, 0)
        return canvas


def write_short(
    reader: FrameReader,
    obj: SkyObject,
    out_path: Path,
    *,
    width: int,
    height: int,
    timeline: Timeline,
    style: ShortStyle | None = None,
    extra: str = "",
) -> None:
    """One vertical clip of one meteor."""
    import imageio_ffmpeg

    style = style or ShortStyle()
    fps = obj.fps
    first = max(0, obj.start_frame - round(style.before_s * fps))
    last = obj.end_frame + round(style.after_s * fps)
    frames = reader.read(first, last)
    if not frames:
        return
    layout = _Layout(obj, width, height, timeline, style, extra)
    writer = imageio_ffmpeg.write_frames(
        str(out_path),
        (style.width, style.height),
        fps=fps,
        codec="libx264",
        pix_fmt_in="bgr24",
        pix_fmt_out="yuv420p",
        macro_block_size=1,
        output_params=["-crf", str(style.crf), "-preset", "medium", "-movflags", "+faststart"],
    )
    writer.send(None)
    try:
        for image in frames:
            writer.send(np.ascontiguousarray(layout.frame(image, slow=False)))
        slow_first = max(0, obj.start_frame - round(0.3 * fps) - first)
        slow_last = min(len(frames), obj.end_frame + round(0.4 * fps) - first + 1)
        for image in frames[slow_first:slow_last]:
            still = np.ascontiguousarray(layout.frame(image, slow=True))
            for _ in range(style.slow_factor):
                writer.send(still)
    finally:
        writer.close()


def write_shorts(
    video: str | Path,
    meteors: Sequence[SkyObject],
    out_dir: Path,
    *,
    count: int,
    width: int,
    height: int,
    timeline: Timeline,
    style: ShortStyle | None = None,
    progress: Callable[[float], None] | None = None,
) -> list[Path]:
    """Clips of the `count` best meteors, named by rank: shorts/01-meteor-12.mp4 is the best one."""
    chosen = best(meteors, count)
    if not chosen:
        return []
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    with FrameReader(video) as reader:
        for place, obj in enumerate(sorted(chosen, key=lambda m: m.start_frame)):
            order = chosen.index(obj) + 1
            path = out_dir / f"{order:02d}-meteor-{obj.number}.mp4"
            write_short(reader, obj, path, width=width, height=height, timeline=timeline, style=style)
            written.append(path)
            if progress:
                progress((place + 1) / len(chosen))
    return sorted(written)
