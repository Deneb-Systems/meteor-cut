"""One run: scan, pick the meteors (and satellites and planes), measure them, write the video, images and reports."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Callable

import cv2

from .images import heatmap, main_scene, radiant, trails
from .meteors import KINDS, Rules, find_meteors, find_slow_objects
from .optics import DEFAULT_METEOR_ALTITUDE_KM, Camera, parse_sensor
from .render import Style, write_highlights
from .report import summary, write_meteors_csv, write_objects_csv, write_points_csv, write_rejected_csv, write_summary
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


def parse_show(value: str | list[str] | tuple[str, ...]) -> tuple[str, ...]:
    """'meteors', 'satellites,planes', 'all' ... -> kinds."""
    words = value.split(",") if isinstance(value, str) else list(value)
    kinds: list[str] = []
    for word in (w.strip().lower().rstrip("s") for w in words if w.strip()):
        if word == "all":
            return KINDS
        if word not in KINDS:
            raise ValueError(f"show takes meteors, satellites, planes or all, not {word!r}")
        kinds.append(word)
    return tuple(dict.fromkeys(kinds)) or ("meteor",)


@dataclass
class Result:
    out_dir: Path
    video: Path | None
    meteors: int
    summary: dict


Progress = Callable[[str, float], None]


def run(video: str, out_dir: str | Path, lens: Lens = Lens(), *, sky_region: tuple[int, int, int, int] | None = None,
        rules: Rules = Rules(), style: Style = Style(), make_video: bool = True, slow_objects: bool = True,
        progress: Progress | None = None) -> Result:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    report = progress or (lambda stage, fraction: None)

    fps, width, height, _ = video_info(video)
    camera = lens.camera(width, height)   # fail on bad lens settings before the long scan

    found = scan(video, sky_region=sky_region, slow_objects=slow_objects or any(k != "meteor" for k in style.show),
                 progress=lambda f: report("Finding moving objects", f))
    report("Sorting meteors from everything else", 0.0)
    meteors, rejected = find_meteors(found, camera, rules, altitude_km=lens.altitude_km)
    others, rejected_slow = find_slow_objects(found, meteors, camera, rules)

    scene = main_scene(found, meteors)
    point = radiant([m for m in meteors if m.scene == scene], found.width)
    images = {}
    if meteors:
        images["trajectories.png"] = trails(found, meteors, others, point)
        images["heatmap.png"] = heatmap(found, meteors)
        for name, image in images.items():
            cv2.imwrite(str(out / name), image)

    highlight = out / "meteors.mp4"
    ranges: list[tuple[int, int]] = []
    if make_video:
        styled = replace(style, end_cards=list(images.values()))
        ranges = write_highlights(found, meteors, str(highlight), styled,
                                  progress=lambda f: report("Writing the highlight video", f), others=others)
    write_meteors_csv(out / "meteors.csv", meteors)
    write_objects_csv(out / "satellites_planes.csv", others)
    write_points_csv(out / "meteor_points.csv", [*meteors, *others], found.fps, camera)
    write_rejected_csv(out / "rejected.csv", [*rejected, *rejected_slow], found.fps)
    data = summary(found, meteors, [*rejected, *rejected_slow], camera, ranges,
                   {"lens": asdict(lens), "sky_region": sky_region, "show": list(style.show),
                    "rules": asdict(rules)}, others=others, radiant=point, radiant_scene=scene)
    write_summary(out / "summary.json", data)
    report("Done", 1.0)
    return Result(out, highlight if ranges else None, len(meteors), data)
