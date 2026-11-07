"""Still images of the whole night: every trajectory on one picture, and a heat map of where meteors appeared.

In a compilation the camera points somewhere else in each scene, so the trajectories and the radiant use the scene
with the most meteors; the heat map uses every meteor, by position in the frame.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Sequence

import cv2
import numpy as np
from PIL import Image, ImageDraw

from .model import Kind, Scan, SkyObject
from .radiant import Radiant
from .text import font, speed_text

KIND_RGB = {Kind.METEOR: (131, 232, 250), Kind.SATELLITE: (120, 230, 140), Kind.PLANE: (255, 170, 80)}
TEXT = (235, 240, 242)
MUTED = (150, 160, 166)
CAPTION_BG = (5, 8, 11)


def busiest_scene(meteors: Sequence[SkyObject]) -> int:
    counts = Counter(m.scene for m in meteors)
    return counts.most_common(1)[0][0] if counts else 0


def background(scan: Scan, scene: int) -> np.ndarray:
    """The scene's darkest sampled frame, falling back to any scene's, then to the first frame of the video."""
    frame = scan.backgrounds.get(scene)
    if frame is None and scan.backgrounds:
        frame = next(iter(scan.backgrounds.values()))
    if frame is None:
        capture = cv2.VideoCapture(scan.path)
        ok, frame = capture.read()
        capture.release()
        if not ok:
            frame = np.zeros((scan.height, scan.width, 3), np.uint8)
    return frame


def _speed_rgb(fraction: float) -> tuple[int, int, int]:
    """Slow blue to fast red (turbo colour map)."""
    value = np.array([[min(255, max(0, int(40 + fraction * 215)))]], np.uint8)
    b, g, r = cv2.applyColorMap(value, cv2.COLORMAP_TURBO)[0, 0]
    return int(r), int(g), int(b)


def _caption(image: Image.Image, title: str, lines: Sequence[str]) -> None:
    draw = ImageDraw.Draw(image)
    size = max(16, image.width // 70)
    height = int(size * 1.9) + len(lines) * int(size * 1.4) + size
    draw.rectangle((0, 0, image.width, height), fill=CAPTION_BG)
    draw.text((size, size // 2), title, font=font(int(size * 1.3), bold=True), fill=TEXT)
    for i, line in enumerate(lines):
        draw.text((size, int(size * 2.1) + i * int(size * 1.4)), line, font=font(size), fill=MUTED)


def _to_bgr(image: Image.Image) -> np.ndarray:
    return cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)


def heatmap(scan: Scan, meteors: Sequence[SkyObject]) -> np.ndarray:
    """Where meteors crossed the frame, as a glow over the darkest frame of the busiest scene."""
    base = background(scan, busiest_scene(meteors)).astype(np.float32) * 0.6
    small = 4
    density: np.ndarray = np.zeros((scan.height // small + 1, scan.width // small + 1), np.float32)
    for m in meteors:
        pts = np.array([[p.cx / small, p.cy / small] for p in m.points], np.int32)
        if len(pts) > 1:
            layer = np.zeros_like(density)
            cv2.polylines(layer, [pts], False, 1.0, max(1, scan.width // 600))
            density += layer
    density = cv2.GaussianBlur(density, (0, 0), max(2.0, scan.width / small * 0.025))
    density = cv2.resize(density, (scan.width, scan.height), interpolation=cv2.INTER_LINEAR)
    peak = float(np.max(density))
    if peak > 0:
        density = density / peak
    colour = cv2.applyColorMap((density * 255).astype(np.uint8), cv2.COLORMAP_INFERNO).astype(np.float32)
    alpha = np.clip(density * 1.4, 0, 0.85)[..., None]
    out = (base * (1 - alpha) + colour * alpha).clip(0, 255).astype(np.uint8)
    image = Image.fromarray(cv2.cvtColor(out, cv2.COLOR_BGR2RGB))
    scenes = len(scan.cuts) + 1
    overlaid = f" (all {scenes} scenes overlaid)" if scenes > 1 else ""
    _caption(image, "Where the meteors were", [f"{len(meteors)} meteors, brighter where more crossed{overlaid}"])
    return _to_bgr(image)


def _arrow(draw: ImageDraw.ImageDraw, obj: SkyObject, colour: tuple[int, int, int], width: int) -> None:
    ex, ey = obj.end
    angle = math.atan2(ey - obj.start[1], ex - obj.start[0])
    size = width * 5
    draw.polygon(
        [
            (ex, ey),
            (ex - size * math.cos(angle - 0.45), ey - size * math.sin(angle - 0.45)),
            (ex - size * math.cos(angle + 0.45), ey - size * math.sin(angle + 0.45)),
        ],
        fill=colour,
    )


def trajectories(scan: Scan, meteors: Sequence[SkyObject], radiant: Radiant | None) -> np.ndarray:
    """Every meteor of the busiest scene on one picture, coloured by speed, with the radiant they trace back to."""
    scene = busiest_scene(meteors)
    base = (background(scan, scene).astype(np.float32) * 0.55).astype(np.uint8)
    image = Image.fromarray(cv2.cvtColor(base, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(image, "RGBA")
    shown = [m for m in meteors if m.scene == scene]
    width = max(2, scan.width // 640)

    if radiant:
        members = set(radiant.numbers)
        for m in shown:
            if m.number in members:
                draw.line((m.start, (radiant.x, radiant.y)), fill=(255, 255, 255, 46), width=1)
    speeds = [m.speed_value for m in shown]
    lo, hi = (min(speeds), max(speeds)) if speeds else (0.0, 1.0)
    for m in sorted(shown, key=lambda m: m.speed_value):
        colour = _speed_rgb((m.speed_value - lo) / (hi - lo) if hi > lo else 0.5)
        draw.line([(p.cx, p.cy) for p in m.points], fill=colour, width=width + 1)
        _arrow(draw, m, colour, width)
    if radiant and radiant.in_frame(scan.width, scan.height):
        r = scan.width // 80
        draw.ellipse((radiant.x - r, radiant.y - r, radiant.x + r, radiant.y + r), outline=TEXT, width=width)
        draw.text((radiant.x + r + 6, radiant.y - r), "radiant", font=font(max(14, scan.width // 90)), fill=TEXT)

    lines = [f"{len(shown)} meteors, coloured by speed: blue slow, red fast"]
    if shown:
        slowest, fastest = min(shown, key=lambda m: m.speed_value), max(shown, key=lambda m: m.speed_value)
        lines[0] += f" ({speed_text(slowest)[0]} to {speed_text(fastest)[0]})"
    if radiant:
        where = "in the frame" if radiant.in_frame(scan.width, scan.height) else "outside the frame"
        lines.append(f"{radiant.members} of them trace back to one radiant, {where} (faint lines)")
    _caption(image, "Every meteor" + (f" · scene {scene + 1}" if scan.cuts else ""), lines)
    return _to_bgr(image)


def satellites_and_planes(scan: Scan, others: Sequence[SkyObject], label_limit: int = 40) -> np.ndarray:
    """Every satellite and plane of the busiest scene: satellites faint green (busy crossings show as density),
    planes orange, labelled when there are few enough to read."""
    scene = Counter(o.scene for o in others).most_common(1)[0][0] if others else 0
    base = (background(scan, scene).astype(np.float32) * 0.55).astype(np.uint8)
    image = Image.fromarray(cv2.cvtColor(base, cv2.COLOR_BGR2RGB)).convert("RGBA")
    layer = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    shown = [o for o in others if o.scene == scene]
    width = max(2, scan.width // 640)
    label_font = font(max(12, scan.width // 110))
    satellites = [o for o in shown if o.kind is Kind.SATELLITE]
    planes = [o for o in shown if o.kind is Kind.PLANE]
    alpha = 255 if len(satellites) <= label_limit else max(60, int(255 * label_limit / len(satellites)))
    for o in satellites:
        draw.line([(p.cx, p.cy) for p in o.points], fill=(*KIND_RGB[Kind.SATELLITE], alpha), width=width)
    for o in planes:
        draw.line([(p.cx, p.cy) for p in o.points], fill=(*KIND_RGB[Kind.PLANE], 255), width=width + 1)
        _arrow(draw, o, KIND_RGB[Kind.PLANE], width)
    for o in [*planes, *(satellites if len(satellites) <= label_limit else [])]:
        draw.text((o.end[0] + 6, o.end[1] - 6), o.label, font=label_font, fill=(*KIND_RGB[o.kind], 255))
    image = Image.alpha_composite(image, layer).convert("RGB")
    hours = scan.duration_s / 3600
    rate = f", about {len(satellites) / hours:.0f} satellites an hour" if hours >= 1 else ""
    _caption(
        image,
        "Satellites and planes" + (f" · scene {scene + 1}" if scan.cuts else ""),
        [f"{len(satellites)} satellites (green), {len(planes)} planes (orange){rate}"],
    )
    return _to_bgr(image)
