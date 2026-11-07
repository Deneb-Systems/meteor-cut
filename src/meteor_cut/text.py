"""Fonts and the few ways numbers are written in the video, images and tables."""

from __future__ import annotations

import math
from functools import lru_cache

from PIL import ImageFont

from .model import SkyObject

_FONTS = {
    False: ("DejaVuSans.ttf", "Arial.ttf", "Helvetica.ttc", "LiberationSans-Regular.ttf"),
    True: ("DejaVuSans-Bold.ttf", "Arial Bold.ttf", "Helvetica-Bold.ttf", "LiberationSans-Bold.ttf"),
}


@lru_cache(maxsize=32)
def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """A system sans-serif font, or Pillow's built-in one."""
    for name in _FONTS[bold]:
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default(size)


def clock(seconds: float) -> str:
    """01:15.2 under an hour, 4:01:02 above."""
    seconds = max(0.0, seconds)
    hours, rest = divmod(seconds, 3600)
    minutes, rest = divmod(rest, 60)
    if hours:
        return f"{int(hours)}:{int(minutes):02d}:{int(rest):02d}"
    return f"{int(minutes):02d}:{rest:04.1f}"


def speed_text(obj: SkyObject) -> tuple[str, str]:
    """(main speed, km/s estimate or ""): degrees per second when the lens is known, else pixels per second."""
    if obj.motion and math.isfinite(obj.motion.angular_speed_deg_s):
        km = f"≥{obj.motion.sky_speed_km_s:.0f} km/s" if obj.motion.sky_speed_km_s else ""
        return f"{obj.motion.angular_speed_deg_s:.1f}°/s", km
    return f"{obj.speed_px_s:.0f} px/s", ""


def ascii_text(text: str) -> str:
    """OpenCV's built-in font has no degree or ≥ signs."""
    return text.replace("°", " deg").replace("≥", ">")
