"""Fonts and the few ways numbers are written in the video, images and tables."""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timedelta
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


def clock(seconds: float, *, hours_always: bool = False) -> str:
    """01:15.2 under an hour, 4:01:02 above, or always with hours (for lists from a video over an hour long)."""
    seconds = max(0.0, seconds)
    hours, rest = divmod(seconds, 3600)
    minutes, rest = divmod(rest, 60)
    if hours or hours_always:
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


def time_left_s(fraction: float, start_fraction: float, elapsed_s: float) -> float | None:
    """Seconds left in a step, from its average speed so far; None until there is enough to go on."""
    done = fraction - start_fraction
    if fraction >= 1:
        return 0.0
    if done <= 0.0005 or elapsed_s < 3:
        return None
    return (1 - fraction) * elapsed_s / done


def duration_text(seconds: float) -> str:
    """ "under a minute", "about 42 min", "about 1 h 05 min"."""
    if seconds < 60:
        return "under a minute"
    minutes = round(seconds / 60)
    if minutes < 60:
        return f"about {minutes} min"
    return f"about {minutes // 60} h {minutes % 60:02d} min"


@dataclass(frozen=True)
class Timeline:
    """Turns seconds into the video into times people read: the clock time when the start is known."""

    start: datetime | None = None
    long: bool = False  # a video of an hour or more: source times always with hours

    def at(self, seconds: float) -> datetime | None:
        return self.start + timedelta(seconds=seconds) if self.start else None

    def label(self, seconds: float) -> str:
        """22:41:07 with a start time, else the time into the video."""
        when = self.at(seconds)
        return when.strftime("%H:%M:%S") if when else clock(seconds, hours_always=self.long)

    def iso(self, seconds: float) -> str:
        when = self.at(seconds)
        return when.isoformat(timespec="milliseconds") if when else ""


def magnitude_text(obj: SkyObject) -> str:
    """'mag -2.4', '≤ mag -4.1' when the sensor clipped, or ''."""
    b = obj.brightness
    if b is None or b.magnitude is None:
        return ""
    return f"{'≤ ' if b.saturated else ''}mag {b.magnitude:.1f}"


def length_text(obj: SkyObject) -> str:
    if obj.motion and math.isfinite(obj.motion.angular_length_deg):
        return f"{obj.motion.angular_length_deg:.1f}°"
    return f"{obj.length_px:.0f} px"
