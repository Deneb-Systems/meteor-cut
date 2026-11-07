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


# Each top list: its folder, its heading, and the rank it orders meteors by (0 = not ranked).
GROUPS: dict[str, tuple[str, str, Callable[[SkyObject], int]]] = {
    "fastest": ("Fastest", "fastest", lambda m: m.rank),
    "longest": ("Longest trails", "longest trail", lambda m: m.length_rank),
    "brightest": ("Brightest", "brightest", lambda m: m.brightness_rank),
}


def top_lists(meteors: Sequence[SkyObject], count: int) -> dict[str, list[SkyObject]]:
    """The `count` fastest, longest and brightest meteors; a list is left out when nothing was ranked for it."""
    lists = {}
    for group, (_, _, rank) in GROUPS.items():
        ranked = sorted((m for m in meteors if rank(m)), key=rank)[:count]
        if ranked:
            lists[group] = ranked
    return lists


def place_text(place: int, word: str) -> str:
    """'fastest', '2nd fastest', '3rd longest trail'."""
    if place == 1:
        return word
    suffix = "th" if 10 <= place % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(place % 10, "th")
    return f"{place}{suffix} {word}"


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
    when = timeline.at(obj.start_s)
    stamp = when.strftime("%d %b %Y · %H:%M:%S") if when else f"at {timeline.label(obj.start_s)} in the video"
    draw.text((58, 118), stamp, font=font(34), fill=MUTED)
    if subtitle_extra:
        draw.text(
            (58 + draw.textlength(stamp, font=font(34)), 118), subtitle_extra, font=font(34, bold=True), fill=CYAN
        )
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
        mark = Image.new("RGB", (460, 44), (0, 0, 0))
        ImageDraw.Draw(mark).text((452, 8), "Meteor Cut · by Deneb Systems", font=font(26), fill=MUTED, anchor="ra")
        self.watermark = _to_bgr(mark)

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
        mh, mw = self.watermark.shape[:2]
        corner = canvas[bottom - mh - 8 : bottom - 8, s.width - mw - 16 : s.width - 16]
        np.maximum(corner, self.watermark, out=corner)  # light text over the dark sky
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
) -> dict[str, list[Path]]:
    """Clips of the `count` fastest, longest and brightest meteors, a folder per list: shorts/fastest/01-meteor-12.mp4
    is the fastest. A meteor on several lists gets a clip in each, titled with its place on that list."""
    lists = top_lists(meteors, count)
    jobs = [(group, place, obj) for group, chosen in lists.items() for place, obj in enumerate(chosen, start=1)]
    written: dict[str, list[Path]] = {group: [] for group in lists}
    if not jobs:
        return written
    with FrameReader(video) as reader:
        # In order of appearance, so the reader mostly moves forward through the video.
        for done, (group, place, obj) in enumerate(sorted(jobs, key=lambda j: j[2].start_frame), start=1):
            folder = out_dir / group
            folder.mkdir(parents=True, exist_ok=True)
            path = folder / f"{place:02d}-meteor-{obj.number}.mp4"
            extra = f"  ·  {place_text(place, GROUPS[group][1])}"
            write_short(reader, obj, path, width=width, height=height, timeline=timeline, style=style, extra=extra)
            written[group].append(path)
            if progress:
                progress(done / len(jobs))
    return {group: sorted(paths) for group, paths in written.items()}
