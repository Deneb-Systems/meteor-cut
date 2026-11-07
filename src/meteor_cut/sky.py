"""The camera against the stars: one plate-solved frame, followed through the night.

A frame is taken from the middle of the video (a median of a few frames, so meteors, planes and noise drop out) and
plate-solved by astrometry.net, locally with solve-field or by uploading sky-frame.png to nova.astrometry.net and
passing the wcs.fits it returns. The solution says exactly which part of the sky each pixel saw at that moment; the
Earth's rotation (see astro.py) carries it to any other moment of the video, as long as the camera did not move.

With it, a meteor's path is a piece of a great circle on the sky rather than a line in the picture: lens distortion
no longer matters, the shower radiant stays fixed among the stars all night, and the brightness of each meteor can
be measured against the catalogued stars around it.
"""

from __future__ import annotations

import logging
import math
import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import cv2
import numpy as np

from .astro import SIDEREAL_DEG_PER_S, alt_az, separation_deg, unit_vector
from .errors import SettingsError
from .optics import AngularMotion, slant_range_km
from .wcs import Wcs

log = logging.getLogger(__name__)

INDEX_URL = "http://data.astrometry.net/4100/index-{}.fits"
# Index files for wide fields (quads of 5.6 to 30 degrees): what meteor cameras need, 1.6 MB in all.
WIDE_FIELD_INDEXES = (4115, 4116, 4117, 4118, 4119)


@dataclass(frozen=True)
class Site:
    """Where the camera stood: latitude north and longitude east, in degrees."""

    latitude_deg: float
    longitude_deg: float

    def __post_init__(self) -> None:
        if not (-90 <= self.latitude_deg <= 90 and -180 <= self.longitude_deg <= 360):
            raise SettingsError("site latitude must be -90..90 and longitude -180..180 degrees")

    @classmethod
    def parse(cls, value: str) -> Site:
        """'50.45,30.52' (north and east positive)."""
        try:
            lat, lon = (float(part) for part in value.replace(" ", "").split(","))
        except ValueError:
            raise SettingsError(
                f"site takes LATITUDE,LONGITUDE in degrees, such as 50.45,30.52, not {value!r}"
            ) from None
        return cls(lat, lon)


def parse_start(value: str) -> datetime:
    """When the video started: ISO 8601, such as 2026-08-12T22:30 or 2026-08-12 22:30:05+02:00.

    A time without a zone is taken as this computer's local time.
    """
    try:
        when = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    except ValueError:
        raise SettingsError(f"start takes a date and time such as 2026-08-12T22:30+02:00, not {value!r}") from None
    return when if when.tzinfo else when.astimezone()


def recorded_start(path: str | Path) -> datetime | None:
    """The recording time a camera wrote into the file (creation_time), or None.

    Cameras disagree on whether it marks the start or the end of a clip, and phones often store the moment the file
    was saved, so this is only a fallback for an explicit start time.
    """
    import imageio_ffmpeg

    try:
        output = subprocess.run(  # noqa: S603 - ffmpeg from imageio-ffmpeg, our own arguments
            [imageio_ffmpeg.get_ffmpeg_exe(), "-hide_banner", "-i", str(path)],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        ).stderr
    except (OSError, subprocess.SubprocessError):
        return None
    found = re.search(r"creation_time\s*:\s*(\S+)", output)
    if not found:
        return None
    try:
        when = datetime.fromisoformat(found.group(1).replace("Z", "+00:00"))
    except ValueError:
        return None
    if when.year < 2000:  # unset clocks write 1904 or 1970
        return None
    return when if when.tzinfo else when.replace(tzinfo=timezone.utc)


@dataclass(frozen=True)
class SkyCalibration:
    """A plate solution and the moment (seconds into the video) of the frame it was made from."""

    wcs: Wcs
    at_s: float

    def rotation_deg(self, t_s: float) -> float:
        """How far the sky has turned (degrees of right ascension) since the solved frame."""
        return SIDEREAL_DEG_PER_S * (t_s - self.at_s)

    def ra_dec(self, x: float, y: float, t_s: float) -> tuple[float, float]:
        """Where pixel (x, y) pointed t_s seconds into the video."""
        ra, dec = self.wcs.pixel_to_world(x, y)
        return (float(ra) + self.rotation_deg(t_s)) % 360.0, float(dec)

    def vector(self, x: float, y: float, t_s: float) -> np.ndarray:
        return unit_vector(*self.ra_dec(x, y, t_s))

    def pixel(self, ra: np.ndarray | float, dec: np.ndarray | float, t_s: float) -> tuple[np.ndarray, np.ndarray]:
        """Where a sky position appeared t_s seconds into the video (NaN behind the camera)."""
        return self.wcs.world_to_pixel(np.asarray(ra, float) - self.rotation_deg(t_s), dec)

    def field_deg(self, width: int, height: int) -> tuple[float, float]:
        """Horizontal and vertical field of view, edge to edge through the centre."""
        cx, cy = (width - 1) / 2, (height - 1) / 2
        across = separation_deg(self.vector(0, cy, self.at_s), self.vector(width - 1, cy, self.at_s))
        down = separation_deg(self.vector(cx, 0, self.at_s), self.vector(cx, height - 1, self.at_s))
        return across, down

    def centre(self, width: int, height: int) -> tuple[float, float]:
        return self.ra_dec((width - 1) / 2, (height - 1) / 2, self.at_s)


def sky_motion(
    sky: SkyCalibration,
    start: tuple[float, float],
    end: tuple[float, float],
    start_s: float,
    end_s: float,
    *,
    when: datetime | None,
    site: Site | None,
    altitude_km: float | None,
    min_elevation_deg: float = 3.0,
) -> AngularMotion:
    """Like optics.measure_motion, but on the sky: exact angles despite lens distortion, and elevations from the
    stars when the site and time are known (no camera tilt needed)."""
    a, b = sky.vector(*start, start_s), sky.vector(*end, end_s)
    duration = end_s - start_s
    length = separation_deg(a, b)
    speed = length / duration if duration > 0 else float("nan")
    e0 = e1 = range_km = sky_speed = None
    if when is not None and site is not None:
        e0 = alt_az(
            *sky.ra_dec(*start, start_s), when + timedelta(seconds=start_s), site.latitude_deg, site.longitude_deg
        )[0]
        e1 = alt_az(*sky.ra_dec(*end, end_s), when + timedelta(seconds=end_s), site.latitude_deg, site.longitude_deg)[0]
        mid = (e0 + e1) / 2
        if altitude_km is not None and mid >= min_elevation_deg:
            range_km = slant_range_km(mid, altitude_km)
            sky_speed = math.radians(speed) * range_km
    return AngularMotion(length, speed, e0, e1, range_km, sky_speed)


def sky_frame(path: str | Path, at_s: float, *, frames: int = 15, spacing: int = 4) -> np.ndarray:
    """A clean grey picture of the stars around `at_s`: the median of several frames (meteors, planes and noise drop
    out), with the sky glow and the landscape's gradients flattened so faint stars stand out."""
    capture = cv2.VideoCapture(str(path))
    try:
        fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
        first = max(0, round(at_s * fps) - frames * spacing // 2)
        capture.set(cv2.CAP_PROP_POS_FRAMES, first)
        stack = []
        for i in range(frames * spacing):
            ok, image = capture.read()
            if not ok:
                break
            if i % spacing == 0:
                stack.append(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32))
    finally:
        capture.release()
    if not stack:
        raise SettingsError("could not read frames for the star picture")
    median: np.ndarray = np.median(stack, axis=0)
    glow: np.ndarray = cv2.GaussianBlur(median, (0, 0), 25)
    picture: np.ndarray = np.clip((median - glow) * 4 + 20, 0, 255).astype(np.uint8)
    return picture


def find_solver() -> str | None:
    """astrometry.net's solve-field, when installed (brew install astrometry-net, apt install astrometry.net)."""
    return shutil.which("solve-field")


def default_index_dir() -> Path | None:
    """$METEOR_CUT_INDEX_DIR, else ~/.cache/meteor-cut/astrometry, when it holds index files."""
    configured = os.environ.get("METEOR_CUT_INDEX_DIR")
    folder = Path(configured) if configured else Path.home() / ".cache" / "meteor-cut" / "astrometry"
    return folder if any(folder.glob("index-*.fits")) else None


def download_indexes(folder: Path | None = None) -> Path:
    """Fetches the five small wide-field index files into `folder` (default: ~/.cache/meteor-cut/astrometry)."""
    import urllib.request

    folder = folder or Path(os.environ.get("METEOR_CUT_INDEX_DIR") or Path.home() / ".cache/meteor-cut/astrometry")
    folder.mkdir(parents=True, exist_ok=True)
    for number in WIDE_FIELD_INDEXES:
        target = folder / f"index-{number}.fits"
        if not target.exists():
            partial = target.with_suffix(".part")
            with urllib.request.urlopen(INDEX_URL.format(number), timeout=120) as response:  # noqa: S310 - fixed URL
                partial.write_bytes(response.read())
            partial.rename(target)
    return folder


def solve(
    image: np.ndarray,
    *,
    index_dir: Path | None = None,
    min_width_deg: float = 5.0,
    max_width_deg: float = 140.0,
    timeout_s: float = 180.0,
) -> Wcs:
    """Plate-solves a star picture with solve-field; raises SettingsError when it is missing or finds no match."""
    solver = find_solver()
    if solver is None:
        raise SettingsError(
            "plate solving needs astrometry.net's solve-field (brew install astrometry-net, or apt install "
            "astrometry.net); or upload sky-frame.png to nova.astrometry.net and pass the wcs.fits it returns"
        )
    index_dir = index_dir or default_index_dir()
    with tempfile.TemporaryDirectory(prefix="meteor-cut-solve-") as tmp:
        picture = Path(tmp) / "sky.png"
        cv2.imwrite(str(picture), image)
        command = [
            solver, "--overwrite", "--no-plots", "--no-verify", "--crpix-center",
            "--scale-units", "degwidth", "--scale-low", str(min_width_deg), "--scale-high", str(max_width_deg),
            "--downsample", "2", "--objs", "100", "--cpulimit", str(round(timeout_s)),
            "--dir", tmp, "--new-fits", "none", "--rdls", "none", "--corr", "none", "--match", "none",
            "--index-xyls", "none", "--solved", str(Path(tmp) / "solved"), "--wcs", str(Path(tmp) / "sky.wcs"),
        ]  # fmt: skip
        if index_dir is not None:
            command[1:1] = ["--index-dir", str(index_dir)]
        command.append(str(picture))
        log.info("plate solving: %s", " ".join(command))
        try:
            subprocess.run(command, capture_output=True, timeout=timeout_s + 60, check=False)  # noqa: S603
        except subprocess.TimeoutExpired:
            raise SettingsError("the plate solver ran out of time") from None
        solution = Path(tmp) / "sky.wcs"
        if not solution.exists():
            raise SettingsError(
                "the stars could not be identified: is the sky clear, and the camera still? Without an index "
                "folder, run `meteor-cut star-index` first"
            )
        return Wcs.read(solution)
