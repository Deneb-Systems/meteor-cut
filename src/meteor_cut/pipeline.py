"""One run: scan, pick the meteors, measure them, write the highlight video and the reports."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

from .meteors import Rules, find_meteors
from .optics import DEFAULT_METEOR_ALTITUDE_KM, Camera, parse_sensor
from .render import Style, write_highlights
from .report import summary, write_meteors_csv, write_points_csv, write_rejected_csv, write_summary
from .scan import scan, video_info


@dataclass
class Lens:
    """What the user knows about the camera. Either focal_mm with a sensor, or a horizontal field of view."""
    focal_mm: float | None = None
    sensor: str | None = None         # preset name or WIDTHxHEIGHT mm
    hfov_deg: float | None = None
    tilt_deg: float | None = None     # camera pointing this far above the horizon
    roll_deg: float = 0.0
    altitude_km: float = DEFAULT_METEOR_ALTITUDE_KM

    def camera(self, width: int, height: int) -> Camera | None:
        if self.hfov_deg:
            return Camera.from_fov(width, height, self.hfov_deg, self.tilt_deg, self.roll_deg)
        if self.focal_mm:
            sensor_width = parse_sensor(self.sensor or "full-frame")[0]
            return Camera.from_lens(width, height, self.focal_mm, sensor_width, self.tilt_deg, self.roll_deg)
        if self.tilt_deg is not None:
            raise ValueError("a mounting angle needs the lens too: give a focal length (and sensor) or a field of view")
        return None


@dataclass
class Result:
    out_dir: Path
    video: Path | None
    meteors: int
    summary: dict


Progress = Callable[[str, float], None]


def run(video: str, out_dir: str | Path, lens: Lens = Lens(), *, sky_region: tuple[int, int, int, int] | None = None,
        rules: Rules = Rules(), style: Style = Style(), make_video: bool = True,
        progress: Progress | None = None) -> Result:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    report = progress or (lambda stage, fraction: None)

    fps, width, height, _ = video_info(video)
    camera = lens.camera(width, height)   # fail on bad lens settings before the long scan

    found = scan(video, sky_region=sky_region, progress=lambda f: report("Finding moving objects", f))
    meteors, rejected = find_meteors(found, camera, rules, altitude_km=lens.altitude_km)

    highlight = out / "meteors.mp4"
    ranges: list[tuple[int, int]] = []
    if make_video and meteors:
        ranges = write_highlights(found, meteors, str(highlight), style,
                                  progress=lambda f: report("Writing the highlight video", f))
    write_meteors_csv(out / "meteors.csv", meteors)
    write_points_csv(out / "meteor_points.csv", meteors, found.fps, camera)
    write_rejected_csv(out / "rejected.csv", rejected, found.fps)
    data = summary(found, meteors, rejected, camera, ranges,
                   {"lens": asdict(lens), "sky_region": sky_region, "rules": asdict(rules)})
    write_summary(out / "summary.json", data)
    report("Done", 1.0)
    return Result(out, highlight if ranges else None, len(meteors), data)
