"""Second pass: the highlight video, made only of the moments with meteors (and, if asked, satellites and planes).

Layout (1920x1080 by default): the video on the left, scaled to fit, with a header (source time, what is on screen)
and a footer (totals); on the right a panel listing every meteor sorted by speed, with the ones on screen highlighted.
Each object gets a box at its head, its trail, and a label with its number and speed. Trails of what has already
passed stay faintly on screen until the scene changes. The video ends on the trajectories and heat map images.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .meteors import Meteor
from .scan import Scan

CYAN = (131, 232, 250)      # RGB
WHITE = (240, 244, 245)
MUTED = (140, 152, 158)
BG = (6, 9, 12)
PANEL = (12, 17, 21)
KIND_COLOUR = {"meteor": CYAN, "satellite": (120, 230, 140), "plane": (255, 170, 80)}


@dataclass(frozen=True)
class Style:
    width: int = 1920
    height: int = 1080
    panel_width: int = 480
    bar_height: int = 120         # header and footer around the video
    before_s: float = 0.7         # video kept before an object appears...
    after_s: float = 1.0          # ...and after it ends
    merge_gap_s: float = 0.6      # clips closer than this are joined
    box_padding: int = 14
    show: tuple[str, ...] = ("meteor",)   # kinds that get clips and boxes: meteor, satellite, plane
    trails: bool = True           # keep past trails faintly on screen
    end_card_s: float = 4.0       # seconds for each closing image (trajectories, heat map)
    end_cards: list = field(default_factory=list, compare=False)  # BGR images, set by the pipeline


def scan_scene(scan: Scan, frame: int) -> int:
    return sum(1 for cut in scan.cuts if cut <= frame)


def clips(scan: Scan, objects: list[Meteor], style: Style) -> list[tuple[int, int]]:
    """Frame ranges to keep, inside each object's scene, merged when they nearly touch."""
    ranges = []
    for m in objects:
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


def _font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    for name in (("DejaVuSans-Bold.ttf", "Arial Bold.ttf", "Helvetica-Bold.ttf") if bold else
                 ("DejaVuSans.ttf", "Arial.ttf", "Helvetica.ttc")):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default(size)


def _clock(seconds: float) -> str:
    seconds = max(0.0, seconds)
    hours, rest = divmod(seconds, 3600)
    minutes, rest = divmod(rest, 60)
    if hours:
        return f"{int(hours)}:{int(minutes):02d}:{int(rest):02d}"
    return f"{int(minutes):02d}:{rest:04.1f}"


def speed_text(m: Meteor) -> tuple[str, str]:
    """(main speed, extra) for labels and the panel."""
    if m.speed and math.isfinite(m.speed.angular_speed_deg_s):
        km = f"≥{m.speed.sky_speed_km_s:.0f} km/s" if m.speed.sky_speed_km_s else ""
        return f"{m.speed.angular_speed_deg_s:.1f}°/s", km
    return f"{m.speed_px_s:.0f} px/s", ""


def _bgr(rgb: tuple[int, int, int], factor: float = 1.0) -> tuple[int, int, int]:
    return tuple(int(c * factor) for c in rgb[::-1])


class Renderer:
    def __init__(self, scan: Scan, meteors: list[Meteor], style: Style = Style(), others: list[Meteor] = ()):
        self.scan, self.meteors, self.style = scan, meteors, style
        self.others = list(others)
        self.objects = [o for o in [*meteors, *self.others] if o.kind in style.show]
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
        self._trails = np.zeros((self.video_h, self.video_w, 3), np.uint8)
        self._trailed: set[int] = set()
        self._trail_scene = -1
        duration = scan.frames / scan.fps if scan.fps else 0
        counts = [f"{len(meteors)} meteors"]
        for kind in ("satellite", "plane"):
            n = sum(1 for o in self.others if o.kind == kind)
            if n:
                counts.append(f"{n} {kind}s")
        rate = f"  ·  {len(meteors) / duration * 3600:.0f} meteors per hour" if duration >= 600 else ""
        fastest = self.by_speed[0] if self.by_speed else None
        self.footer = (", ".join(counts) + f" in {_clock(duration)}{rate}"
                       + (f"  ·  fastest {fastest.label} {speed_text(fastest)[0]}" if fastest else ""))

    def active(self, frame: int) -> list[Meteor]:
        return [m for m in self.objects if m.start_frame <= frame <= m.end_frame + self.linger]

    def frame(self, image: np.ndarray, index: int) -> np.ndarray:
        s = self.style
        canvas = np.empty((s.height, s.width, 3), np.uint8)
        canvas[:] = BG[::-1]
        video = cv2.resize(image, (self.video_w, self.video_h), interpolation=cv2.INTER_AREA)
        if s.trails:
            self._update_trails(index)
            np.maximum(video, self._trails, out=video)
        active = self.active(index)
        for m in active:
            self._draw_object(video, m, index)
        canvas[self.video_y:self.video_y + self.video_h, self.video_x:self.video_x + self.video_w] = video
        canvas[:, s.width - s.panel_width:] = self._panel(tuple(sorted(m.id for m in active if m.kind == "meteor")))
        self._bars(canvas, index, active)
        return canvas

    def _update_trails(self, index: int) -> None:
        scene = scan_scene(self.scan, index)
        if scene != self._trail_scene:
            self._trails[:] = 0
            self._trailed.clear()
            self._trail_scene = scene
        k = self.scale
        for m in self.objects:
            if m.scene == scene and m.id not in self._trailed and index > m.end_frame + self.linger:
                pts = np.array([[p.cx * k, p.cy * k] for p in m.points], np.int32)
                if len(pts) > 1:
                    cv2.polylines(self._trails, [pts], False, _bgr(KIND_COLOUR[m.kind], 0.45), 1, cv2.LINE_AA)
                self._trailed.add(m.id)

    def _draw_object(self, video: np.ndarray, m: Meteor, index: int) -> None:
        k = self.scale
        seen = [p for p in m.points if p.frame <= index]
        if not seen:
            return
        colour = _bgr(KIND_COLOUR[m.kind], 0.55 if index > m.end_frame else 1.0)
        trail = np.array([[p.cx * k, p.cy * k] for p in seen], np.int32)
        if len(trail) > 1:
            cv2.polylines(video, [trail], False, colour, 2, cv2.LINE_AA)
        head = seen[-1]
        pad = self.style.box_padding
        x0, y0 = int(head.cx * k - head.w * k / 2 - pad), int(head.cy * k - head.h * k / 2 - pad)
        x1, y1 = int(head.cx * k + head.w * k / 2 + pad), int(head.cy * k + head.h * k / 2 + pad)
        cv2.rectangle(video, (x0, y0), (x1, y1), colour, 2, cv2.LINE_AA)
        main, extra = speed_text(m)
        name = {"meteor": "", "satellite": "satellite ", "plane": "plane "}[m.kind]
        # OpenCV's built-in font has no degree or >= signs.
        label = (f"{name}{m.label}  {main}" + (f"  {extra}" if extra else "")).replace("°", " deg").replace("≥", ">")
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        lx = min(max(0, x0), video.shape[1] - tw - 8)
        ly = y0 - 8 if y0 - th - 14 > 0 else y1 + th + 10
        cv2.rectangle(video, (lx - 4, ly - th - 6), (lx + tw + 4, ly + 6), (21, 17, 12), -1)
        cv2.putText(video, label, (lx, ly), cv2.FONT_HERSHEY_SIMPLEX, 0.6, colour, 1, cv2.LINE_AA)

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
            d.text((72, y), m.label, font=font, fill=colour)
            d.text((140, y), _clock(m.start_s), font=self.f_row, fill=MUTED)
            d.text((250, y), main, font=font, fill=colour)
            if extra:
                d.text((356, y), extra, font=self.f_row, fill=MUTED if not on else CYAN)
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
            now += "  ·  " + "  ".join(f"{m.label} {speed_text(m)[0]}" for m in active[:4])
        d.text((area_w - 32, 44), now, font=self.f_text, fill=CYAN if active else MUTED, anchor="ra")
        canvas[:s.bar_height, :area_w] = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
        foot = Image.new("RGB", (area_w, s.bar_height), BG)
        ImageDraw.Draw(foot).text((32, 44), self.footer, font=self.f_text, fill=MUTED)
        canvas[s.height - s.bar_height:, :area_w] = cv2.cvtColor(np.asarray(foot), cv2.COLOR_RGB2BGR)

    def card(self, image: np.ndarray) -> np.ndarray:
        """A closing still, letterboxed to the video size."""
        s = self.style
        canvas = np.empty((s.height, s.width, 3), np.uint8)
        canvas[:] = BG[::-1]
        k = min(s.width / image.shape[1], s.height / image.shape[0])
        w, h = round(image.shape[1] * k), round(image.shape[0] * k)
        x, y = (s.width - w) // 2, (s.height - h) // 2
        canvas[y:y + h, x:x + w] = cv2.resize(image, (w, h), interpolation=cv2.INTER_AREA)
        return canvas


def write_highlights(scan: Scan, meteors: list[Meteor], out_path: str, style: Style = Style(),
                     progress: Callable[[float], None] | None = None,
                     others: list[Meteor] = ()) -> list[tuple[int, int]]:
    """Writes the highlight video (H.264 MP4) and returns the source frame ranges used."""
    import imageio_ffmpeg

    renderer = Renderer(scan, meteors, style, others)
    ranges = clips(scan, renderer.objects, style)
    if not ranges:
        return ranges
    card_frames = round(style.end_card_s * scan.fps) * len(style.end_cards)
    total = sum(b - a + 1 for a, b in ranges) + card_frames
    writer = imageio_ffmpeg.write_frames(out_path, (style.width, style.height), fps=scan.fps, codec="libx264",
                                         pix_fmt_in="bgr24", pix_fmt_out="yuv420p", macro_block_size=1,
                                         output_params=["-crf", "20", "-preset", "medium", "-movflags", "+faststart"])
    writer.send(None)
    capture = cv2.VideoCapture(scan.path)
    index = written = 0
    queue = iter(ranges)
    current = next(queue)

    def sent() -> None:
        nonlocal written
        written += 1
        if progress and written % 12 == 0:
            progress(written / total)

    try:
        far = round(20 * scan.fps)
        while current is not None:
            while current is not None and index > current[1]:
                current = next(queue, None)
            if current is None:
                break
            if current[0] - index > far:
                # Jump close to the next clip instead of decoding the hours between; land a little early and read on,
                # so the frame count stays exact even where seeking is approximate.
                target = current[0] - round(2 * scan.fps)
                if capture.set(cv2.CAP_PROP_POS_FRAMES, target) and \
                        int(capture.get(cv2.CAP_PROP_POS_FRAMES)) == target:
                    index = target
                else:
                    capture.set(cv2.CAP_PROP_POS_FRAMES, index)
                    far = float("inf")  # this video cannot seek reliably: decode straight through
            if index < current[0]:
                if not capture.grab():
                    break
                index += 1
                continue
            ok, image = capture.read()
            if not ok:
                break
            writer.send(np.ascontiguousarray(renderer.frame(image, index)))
            sent()
            index += 1
        for card in style.end_cards:
            frame = np.ascontiguousarray(renderer.card(card))
            for _ in range(round(style.end_card_s * scan.fps)):
                writer.send(frame)
                sent()
    finally:
        capture.release()
        writer.close()
    if progress:
        progress(1.0)
    return ranges
