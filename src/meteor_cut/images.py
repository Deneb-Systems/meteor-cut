"""Still images of the whole night: a heat map of where meteors appeared, and every trajectory on one picture.

With a rectilinear lens a meteor's path is a straight line in the image, and the meteors of a shower all trace back
to one point, the radiant. radiant() finds the point closest to all their lines, ignoring the meteors that do not
point at it (sporadic meteors from other directions).

In a compilation the camera points somewhere else in each scene, so the trajectories and the radiant use the scene
with the most meteors; the heat map uses every meteor, by position in the frame.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image, ImageDraw

from .meteors import Meteor
from .render import _font, speed_text
from .scan import Scan

SATELLITE = (120, 230, 140)   # RGB
PLANE = (255, 170, 80)
TEXT = (235, 240, 242)
MUTED = (150, 160, 166)


@dataclass(frozen=True)
class Radiant:
    x: float              # pixels; may lie outside the frame
    y: float
    members: int          # meteors whose paths point back to it
    spread_px: float      # RMS distance of their lines from the point


def main_scene(scan: Scan, meteors: list[Meteor]) -> int:
    counts = Counter(m.scene for m in meteors)
    return counts.most_common(1)[0][0] if counts else 0


def background(scan: Scan, scene: int) -> np.ndarray:
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


def _cross(a: np.ndarray, b: np.ndarray) -> float:
    return float(a[0] * b[1] - a[1] * b[0])


def _fit(lines: list[tuple[np.ndarray, np.ndarray]]) -> np.ndarray | None:
    a = np.zeros((2, 2))
    b = np.zeros(2)
    for point, direction in lines:
        m = np.eye(2) - np.outer(direction, direction)
        a += m
        b += m @ point
    if abs(np.linalg.det(a)) < 1e-6:
        return None
    return np.linalg.solve(a, b)


def radiant(meteors: list[Meteor], width: int, min_members: int = 5) -> Radiant | None:
    """Least-squares point behind the meteors' paths, refitted without the meteors that miss it."""
    lines = []
    for m in meteors:
        start, end = np.array(m.start), np.array(m.end)
        d = end - start
        if np.linalg.norm(d) > 0:
            lines.append((start, d / np.linalg.norm(d)))
    if len(lines) < min_members:
        return None
    keep = lines
    point = None
    for tolerance in (0.25, 0.12, 0.06):
        point = _fit(keep)
        if point is None:
            return None
        # A meteor belongs if its line passes near the point and it flies away from it.
        keep = [(p, d) for p, d in lines
                if abs(_cross(d, point - p)) <= tolerance * width and np.dot(p - point, d) > 0]
        if len(keep) < min_members:
            return None
    point = _fit(keep)
    if point is None or np.linalg.norm(point - np.array([width / 2, 0])) > 4 * width:
        return None
    spread = math.sqrt(sum(_cross(d, point - p) ** 2 for p, d in keep) / len(keep))
    return Radiant(float(point[0]), float(point[1]), len(keep), spread)


def _speed_colour(fraction: float) -> tuple[int, int, int]:
    """Slow blue to fast yellow-red (turbo), as RGB."""
    value = np.uint8([[min(255, max(0, int(40 + fraction * 215)))]])
    b, g, r = cv2.applyColorMap(value, cv2.COLORMAP_TURBO)[0, 0]
    return int(r), int(g), int(b)


def _caption(image: Image.Image, title: str, lines: list[str]) -> None:
    d = ImageDraw.Draw(image)
    w = image.width
    size = max(16, w // 70)
    f_title, f_text = _font(int(size * 1.3), True), _font(size)
    height = int(size * 1.9) + len(lines) * int(size * 1.4) + size
    d.rectangle((0, 0, w, height), fill=(5, 8, 11))
    d.text((size, size // 2), title, font=f_title, fill=TEXT)
    for i, line in enumerate(lines):
        d.text((size, int(size * 2.1) + i * int(size * 1.4)), line, font=f_text, fill=MUTED)


def heatmap(scan: Scan, meteors: list[Meteor]) -> np.ndarray:
    """Where meteors crossed the frame, as a glow over the darkest frame of the busiest scene."""
    base = background(scan, main_scene(scan, meteors)).astype(np.float32) * 0.6
    small = 4
    acc = np.zeros((scan.height // small + 1, scan.width // small + 1), np.float32)
    for m in meteors:
        pts = np.array([[p.cx / small, p.cy / small] for p in m.points], np.int32)
        if len(pts) > 1:
            layer = np.zeros_like(acc)
            cv2.polylines(layer, [pts], False, 1.0, max(1, scan.width // 600))
            acc += layer
    sigma = max(2.0, scan.width / small * 0.025)
    acc = cv2.GaussianBlur(acc, (0, 0), sigma)
    acc = cv2.resize(acc, (scan.width, scan.height), interpolation=cv2.INTER_LINEAR)
    if acc.max() > 0:
        acc /= acc.max()
    colour = cv2.applyColorMap((acc * 255).astype(np.uint8), cv2.COLORMAP_INFERNO).astype(np.float32)
    alpha = np.clip(acc * 1.4, 0, 0.85)[..., None]
    out = (base * (1 - alpha) + colour * alpha).clip(0, 255).astype(np.uint8)
    image = Image.fromarray(cv2.cvtColor(out, cv2.COLOR_BGR2RGB))
    scenes = len(scan.cuts) + 1
    _caption(image, "Where the meteors were", [
        f"{len(meteors)} meteors, brighter where more crossed"
        + (f" (all {scenes} scenes overlaid)" if scenes > 1 else "")])
    return cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)


def trails(scan: Scan, meteors: list[Meteor], others: list[Meteor], found: Radiant | None) -> np.ndarray:
    """Every trajectory of the busiest scene on one picture: meteors coloured by speed, satellites and planes."""
    scene = main_scene(scan, meteors)
    base = (background(scan, scene).astype(np.float32) * 0.55).astype(np.uint8)
    image = Image.fromarray(cv2.cvtColor(base, cv2.COLOR_BGR2RGB))
    d = ImageDraw.Draw(image, "RGBA")
    shown = [m for m in meteors if m.scene == scene]
    line = max(2, scan.width // 640)

    if found:
        for m in shown:
            d.line((m.start, (found.x, found.y)), fill=(255, 255, 255, 38), width=1)
    for o in (o for o in others if o.scene == scene):
        d.line([(p.cx, p.cy) for p in o.points], fill=SATELLITE if o.kind == "satellite" else PLANE, width=line)
        d.text((o.end[0] + 6, o.end[1] - 6), o.label, font=_font(max(12, scan.width // 110)),
               fill=SATELLITE if o.kind == "satellite" else PLANE)
    speeds = [m.sort_key() for m in shown]
    lo, hi = (min(speeds), max(speeds)) if speeds else (0, 1)
    for m in sorted(shown, key=Meteor.sort_key):
        colour = _speed_colour((m.sort_key() - lo) / (hi - lo) if hi > lo else 0.5)
        d.line([(p.cx, p.cy) for p in m.points], fill=colour, width=line + 1)
        # Arrowhead in the direction of travel.
        ex, ey = m.end
        angle = math.atan2(ey - m.start[1], ex - m.start[0])
        size = line * 5
        d.polygon([(ex, ey), (ex - size * math.cos(angle - 0.45), ey - size * math.sin(angle - 0.45)),
                   (ex - size * math.cos(angle + 0.45), ey - size * math.sin(angle + 0.45))], fill=colour)
    if found and 0 <= found.x < scan.width and 0 <= found.y < scan.height:
        r = scan.width // 80
        d.ellipse((found.x - r, found.y - r, found.x + r, found.y + r), outline=(255, 255, 255), width=line)
        d.text((found.x + r + 6, found.y - r), "radiant", font=_font(max(14, scan.width // 90)), fill=TEXT)

    satellites = sum(1 for o in others if o.kind == "satellite" and o.scene == scene)
    planes = sum(1 for o in others if o.kind == "plane" and o.scene == scene)
    lines = [f"{len(shown)} meteors, coloured by speed: blue slow, red fast"
             + (f" ({speed_text(min(shown, key=Meteor.sort_key))[0]} to "
                f"{speed_text(max(shown, key=Meteor.sort_key))[0]})" if shown else "")]
    if satellites or planes:
        lines.append(f"{satellites} satellites (green), {planes} planes (orange)")
    if found:
        where = "in the frame" if 0 <= found.x < scan.width and 0 <= found.y < scan.height else "outside the frame"
        lines.append(f"{found.members} meteors trace back to one radiant, {where} (faint lines)")
    _caption(image, "Every trajectory" + (f" · scene {scene + 1}" if scan.cuts else ""), lines)
    return cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
