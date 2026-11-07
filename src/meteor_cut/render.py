"""Second pass: the highlight video, made only of the moments with meteors.

Layout (1920x1080 by default): the video on the left, scaled to fit, with a header (source time, which meteor) and a
footer (totals); on the right a panel listing every meteor sorted by speed, with the ones on screen highlighted. Each
meteor gets a box at its head, its trail, and a label with its number and speed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .meteors import Meteor
from .scan import Scan

CYAN = (131, 232, 250)
WHITE = (240, 244, 245)
MUTED = (140, 152, 158)
DIM = (80, 88, 92)
BG = (6, 9, 12)
PANEL = (12, 17, 21)


@dataclass(frozen=True)
class Style:
    width: int = 1920
    height: int = 1080
    panel_width: int = 480
    bar_height: int = 120         # header and footer around the video
    before_s: float = 0.7         # video kept before a meteor appears...
    after_s: float = 1.0          # ...and after it ends
    merge_gap_s: float = 0.6      # clips closer than this are joined
    box_padding: int = 14


def clips(scan: Scan, meteors: list[Meteor], style: Style) -> list[tuple[int, int]]:
    """Frame ranges to keep, inside each meteor's scene, merged when they nearly touch."""
    ranges = []
    for m in meteors:
        lo, hi = scan.scene_bounds(m.scene)
        ranges.append((max(lo, m.start_frame - round(style.before_s * scan.fps)),
                       min(hi, m.end_frame + round(style.after_s * scan.fps))))
    ranges.sort()
    merged: list[list[int]] = []
    gap = round(style.merge_gap_s * scan.fps)
    for start, end in ranges:
        if merged and start <= merged[-1][1] + gap and scan_scene(scan, start) == scan_scene(scan, merged[-1][1]):
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return [(a, b) for a, b in merged]


def scan_scene(scan: Scan, frame: int) -> int:
    return sum(1 for cut in scan.cuts if cut <= frame)


def _font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    for name in (("DejaVuSans-Bold.ttf", "Arial Bold.ttf", "Helvetica-Bold.ttf") if bold else
                 ("DejaVuSans.ttf", "Arial.ttf", "Helvetica.ttc")):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default(size)


def _clock(seconds: float) -> str:
    minutes, rest = divmod(max(0.0, seconds), 60)
    return f"{int(minutes):02d}:{rest:04.1f}"


def speed_text(m: Meteor) -> tuple[str, str]:
    """(main speed, extra) for labels and the panel."""
    if m.speed and math.isfinite(m.speed.angular_speed_deg_s):
        km = f"≥{m.speed.sky_speed_km_s:.0f} km/s" if m.speed.sky_speed_km_s else ""
        return f"{m.speed.angular_speed_deg_s:.1f}°/s", km
    return f"{m.speed_px_s:.0f} px/s", ""


class Renderer:
    def __init__(self, scan: Scan, meteors: list[Meteor], style: Style = Style()):
        self.scan, self.meteors, self.style = scan, meteors, style
        self.by_speed = sorted(meteors, key=lambda m: m.rank)
        area_w = style.width - style.panel_width
        area_h = style.height - 2 * style.bar_height
        self.scale = min(area_w / scan.width, area_h / scan.height)
        self.video_w, self.video_h = round(scan.width * self.scale), round(scan.height * self.scale)
        self.video_x = (area_w - self.video_w) // 2
        self.video_y = style.bar_height + (area_h - self.video_h) // 2
        self.f_title, self.f_text, self.f_small = _font(30, True), _font(24), _font(19)
        self.f_row, self.f_row_bold = _font(19), _font(19, True)
        self.linger = round(0.5 * scan.fps)
        self._panel_cache: dict[tuple, np.ndarray] = {}
        rate = len(meteors) / (scan.frames / scan.fps) * 3600 if scan.frames else 0
        fastest = self.by_speed[0] if self.by_speed else None
        per_hour = f"  ·  {rate:.0f} per hour" if scan.frames / scan.fps >= 600 else ""
        self.footer = (f"{len(meteors)} meteors in {_clock(scan.frames / scan.fps)}{per_hour}"
                       + (f"  ·  fastest #{fastest.extra['number']} {speed_text(fastest)[0]}" if fastest else ""))

    def active(self, frame: int) -> list[Meteor]:
        return [m for m in self.meteors if m.start_frame <= frame <= m.end_frame + self.linger]

    def frame(self, image: np.ndarray, index: int) -> np.ndarray:
        s = self.style
        canvas = np.empty((s.height, s.width, 3), np.uint8)
        canvas[:] = BG[::-1]
        video = cv2.resize(image, (self.video_w, self.video_h), interpolation=cv2.INTER_AREA)
        active = self.active(index)
        for m in active:
            self._draw_meteor(video, m, index)
        canvas[self.video_y:self.video_y + self.video_h, self.video_x:self.video_x + self.video_w] = video
        canvas[:, s.width - s.panel_width:] = self._panel(tuple(sorted(m.id for m in active)))
        self._bars(canvas, index, active)
        return canvas

    def _draw_meteor(self, video: np.ndarray, m: Meteor, index: int) -> None:
        k = self.scale
        seen = [p for p in m.points if p.frame <= index]
        if not seen:
            return
        ended = index > m.end_frame
        colour = tuple(int(c * (0.55 if ended else 1.0)) for c in CYAN[::-1])
        trail = np.array([[p.cx * k, p.cy * k] for p in seen], np.int32)
        if len(trail) > 1:
            cv2.polylines(video, [trail], False, colour, 2, cv2.LINE_AA)
        head = seen[-1]
        pad = self.style.box_padding
        x0, y0 = int(head.cx * k - head.w * k / 2 - pad), int(head.cy * k - head.h * k / 2 - pad)
        x1, y1 = int(head.cx * k + head.w * k / 2 + pad), int(head.cy * k + head.h * k / 2 + pad)
        cv2.rectangle(video, (x0, y0), (x1, y1), colour, 2, cv2.LINE_AA)
        main, extra = speed_text(m)
        # OpenCV's built-in font has no degree or >= signs.
        label = (f"#{m.extra['number']}  {main}" + (f"  {extra}" if extra else "")).replace("°", " deg").replace("≥", ">")
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        lx = min(max(0, x0), video.shape[1] - tw - 8)
        ly = y0 - 8 if y0 - th - 14 > 0 else y1 + th + 10
        cv2.rectangle(video, (lx - 4, ly - th - 6), (lx + tw + 4, ly + 6), (12, 17, 21), -1)
        cv2.putText(video, label, (lx, ly), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    colour, 1, cv2.LINE_AA)

    def _panel(self, active_ids: tuple[int, ...]) -> np.ndarray:
        if active_ids in self._panel_cache:
            return self._panel_cache[active_ids]
        s = self.style
        img = Image.new("RGB", (s.panel_width, s.height), PANEL)
        d = ImageDraw.Draw(img)
        d.text((28, 34), "Meteors by speed", font=self.f_title, fill=WHITE)
        unit = "angular speed" if any(m.speed for m in self.meteors) else "on-screen speed"
        d.text((28, 76), f"{len(self.meteors)} found · fastest first · {unit}", font=self.f_small, fill=MUTED)
        top, row_h = 124, 32
        rows = (s.height - top - 24) // row_h
        # Scroll so the meteors on screen are visible.
        first = 0
        ranks = [m.rank for m in self.by_speed if m.id in active_ids]
        if ranks and max(ranks) > rows:
            first = min(max(ranks) - rows // 2, len(self.by_speed) - rows)
        for i, m in enumerate(self.by_speed[first:first + rows]):
            y = top + i * row_h
            on = m.id in active_ids
            if on:
                d.rounded_rectangle((16, y - 4, s.panel_width - 16, y + row_h - 6), 6, fill=(22, 52, 60))
            main, extra = speed_text(m)
            colour = CYAN if on else WHITE
            font = self.f_row_bold if on else self.f_row
            d.text((28, y), f"{m.rank:>2}", font=font, fill=MUTED if not on else CYAN)
            d.text((72, y), f"#{m.extra['number']}", font=font, fill=colour)
            d.text((132, y), _clock(m.start_s), font=self.f_row, fill=MUTED)
            d.text((240, y), main, font=font, fill=colour)
            if extra:
                d.text((352, y), extra, font=self.f_row, fill=MUTED if not on else CYAN)
        panel = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
        if len(self._panel_cache) > 256:
            self._panel_cache.clear()
        self._panel_cache[active_ids] = panel
        return panel

    def _bars(self, canvas: np.ndarray, index: int, active: list[Meteor]) -> None:
        s = self.style
        area_w = s.width - s.panel_width
        img = Image.new("RGB", (area_w, s.bar_height), BG)
        d = ImageDraw.Draw(img)
        d.text((32, 30), "Meteor Cut", font=self.f_title, fill=WHITE)
        d.text((32, 72), "tracked with Sky Tracker", font=self.f_small, fill=MUTED)
        now = f"source {_clock(index / self.scan.fps)}"
        if active:
            now += "  ·  " + "  ".join(f"#{m.extra['number']} {speed_text(m)[0]}" for m in active)
        d.text((area_w - 32, 44), now, font=self.f_text, fill=CYAN if active else MUTED, anchor="ra")
        canvas[:s.bar_height, :area_w] = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
        foot = Image.new("RGB", (area_w, s.bar_height), BG)
        ImageDraw.Draw(foot).text((32, 44), self.footer, font=self.f_text, fill=MUTED)
        canvas[s.height - s.bar_height:, :area_w] = cv2.cvtColor(np.asarray(foot), cv2.COLOR_RGB2BGR)


def write_highlights(scan: Scan, meteors: list[Meteor], out_path: str, style: Style = Style(),
                     progress: Callable[[float], None] | None = None) -> list[tuple[int, int]]:
    """Writes the highlight video (H.264 MP4) and returns the source frame ranges used."""
    import imageio_ffmpeg

    ranges = clips(scan, meteors, style)
    if not ranges:
        return ranges
    renderer = Renderer(scan, meteors, style)
    total = sum(b - a + 1 for a, b in ranges)
    writer = imageio_ffmpeg.write_frames(out_path, (style.width, style.height), fps=scan.fps, codec="libx264",
                                         pix_fmt_in="bgr24", pix_fmt_out="yuv420p", macro_block_size=1,
                                         output_params=["-crf", "20", "-preset", "medium", "-movflags", "+faststart"])
    writer.send(None)
    capture = cv2.VideoCapture(scan.path)
    index = written = 0
    queue = iter(ranges)
    current = next(queue)
    try:
        while current is not None:
            ok, image = capture.read()
            if not ok:
                break
            if index > current[1]:
                current = next(queue, None)
            if current is not None and current[0] <= index <= current[1]:
                writer.send(np.ascontiguousarray(renderer.frame(image, index)))
                written += 1
                if progress and written % 12 == 0:
                    progress(written / total)
            index += 1
    finally:
        capture.release()
        writer.close()
    if progress:
        progress(1.0)
    return ranges
