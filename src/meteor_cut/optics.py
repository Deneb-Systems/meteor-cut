"""Camera geometry: where a pixel points in the sky, and how fast a meteor crossed it.

A rectilinear (pinhole) lens is assumed: the usual wide-angle camera lens, not a fisheye. The video is assumed to show
the sensor's full width; a 16:9 video from a 3:2 sensor is cropped top and bottom, which leaves the width alone.

Coordinates: image x to the right, y down. The camera's own axes are x right, y up, z out of the lens. "Tilt" turns
the camera up from the horizon (0 = level, 90 = straight up); "roll" turns it about the lens axis, clockwise as seen
from behind the camera.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .errors import SettingsError

EARTH_RADIUS_KM = 6371.0
# Most meteors burn up 80-110 km up; the speed estimate scales with the altitude assumed.
DEFAULT_METEOR_ALTITUDE_KM = 95.0

# Sensor width x height in millimetres.
SENSORS: dict[str, tuple[float, float]] = {
    "full-frame": (36.0, 24.0),
    "aps-c": (23.5, 15.6),
    "aps-c-canon": (22.3, 14.9),
    "micro-four-thirds": (17.3, 13.0),
    "1-inch": (13.2, 8.8),
    "1/1.3-inch": (9.6, 7.2),
    "1/2.3-inch": (6.17, 4.55),
}


def parse_sensor(value: str) -> tuple[float, float]:
    """A preset name from SENSORS, or WIDTHxHEIGHT in millimetres ("36x24")."""
    key = value.strip().lower()
    if key in SENSORS:
        return SENSORS[key]
    try:
        width, height = (float(part) for part in key.split("x"))
    except ValueError:
        presets = ", ".join(SENSORS)
        raise SettingsError(f"sensor must be one of {presets} or WIDTHxHEIGHT in mm, not {value!r}") from None
    if width <= 0 or height <= 0:
        raise SettingsError("sensor width and height must be positive")
    return width, height


Vector = tuple[float, float, float]


@dataclass(frozen=True)
class Camera:
    width_px: int
    height_px: int
    focal_px: float
    tilt_deg: float | None = None  # None: the mounting angle is unknown, so elevations are too
    roll_deg: float = 0.0

    @classmethod
    def from_lens(
        cls,
        width_px: int,
        height_px: int,
        focal_mm: float,
        sensor_width_mm: float,
        tilt_deg: float | None = None,
        roll_deg: float = 0.0,
    ) -> Camera:
        if focal_mm <= 0 or sensor_width_mm <= 0:
            raise SettingsError("focal length and sensor width must be positive")
        return cls(width_px, height_px, focal_mm / sensor_width_mm * width_px, tilt_deg, roll_deg)

    @classmethod
    def from_fov(
        cls,
        width_px: int,
        height_px: int,
        horizontal_fov_deg: float,
        tilt_deg: float | None = None,
        roll_deg: float = 0.0,
    ) -> Camera:
        if not 1.0 <= horizontal_fov_deg < 179.0:
            raise SettingsError("horizontal field of view must be between 1 and 179 degrees")
        focal_px = (width_px / 2) / math.tan(math.radians(horizontal_fov_deg) / 2)
        return cls(width_px, height_px, focal_px, tilt_deg, roll_deg)

    @property
    def horizontal_fov_deg(self) -> float:
        return math.degrees(2 * math.atan((self.width_px / 2) / self.focal_px))

    def ray(self, x: float, y: float) -> Vector:
        """Unit direction for pixel (x, y): x right, y up, z towards the horizon the camera faces."""
        cx = x - self.width_px / 2
        cy = -(y - self.height_px / 2)
        roll = math.radians(self.roll_deg)
        cx, cy = cx * math.cos(roll) - cy * math.sin(roll), cx * math.sin(roll) + cy * math.cos(roll)
        tilt = math.radians(self.tilt_deg or 0.0)
        up = cy * math.cos(tilt) + self.focal_px * math.sin(tilt)
        forward = self.focal_px * math.cos(tilt) - cy * math.sin(tilt)
        norm = math.sqrt(cx * cx + up * up + forward * forward)
        return cx / norm, up / norm, forward / norm

    def elevation_deg(self, x: float, y: float) -> float | None:
        """Degrees above the horizon, or None without a tilt."""
        if self.tilt_deg is None:
            return None
        return math.degrees(math.asin(max(-1.0, min(1.0, self.ray(x, y)[1]))))


def angle_deg(a: Vector, b: Vector) -> float:
    dot = max(-1.0, min(1.0, a[0] * b[0] + a[1] * b[1] + a[2] * b[2]))
    return math.degrees(math.acos(dot))


def slant_range_km(elevation_deg: float, altitude_km: float = DEFAULT_METEOR_ALTITUDE_KM) -> float:
    """Distance to a point `altitude_km` up, seen `elevation_deg` above the horizon (curved Earth)."""
    e = math.radians(elevation_deg)
    r = EARTH_RADIUS_KM
    return math.sqrt((r + altitude_km) ** 2 - (r * math.cos(e)) ** 2) - r * math.sin(e)


@dataclass(frozen=True)
class AngularMotion:
    """How far and how fast an object crossed the sky, and, for a meteor, an estimate in km/s."""

    angular_length_deg: float
    angular_speed_deg_s: float
    elevation_start_deg: float | None
    elevation_end_deg: float | None
    range_km: float | None
    # Speed across the line of sight at the assumed altitude. The meteor's motion towards or away from the camera is
    # invisible to one camera, so the true speed is at least this.
    sky_speed_km_s: float | None


def measure_motion(
    camera: Camera,
    start: tuple[float, float],
    end: tuple[float, float],
    duration_s: float,
    altitude_km: float | None = DEFAULT_METEOR_ALTITUDE_KM,
    min_elevation_deg: float = 3.0,
) -> AngularMotion:
    """Angular length and speed between two pixels; km/s too when the tilt and an altitude are known.

    Pass altitude_km=None for objects that are not meteors (satellites, planes): their height is unknown, so there is
    no km/s estimate.
    """
    length = angle_deg(camera.ray(*start), camera.ray(*end))
    speed = length / duration_s if duration_s > 0 else float("nan")
    e0, e1 = camera.elevation_deg(*start), camera.elevation_deg(*end)
    range_km = sky_speed = None
    if e0 is not None and e1 is not None and altitude_km is not None:
        mid = camera.elevation_deg((start[0] + end[0]) / 2, (start[1] + end[1]) / 2)
        if mid is not None and mid >= min_elevation_deg:
            range_km = slant_range_km(mid, altitude_km)
            sky_speed = math.radians(speed) * range_km
    return AngularMotion(length, speed, e0, e1, range_km, sky_speed)
