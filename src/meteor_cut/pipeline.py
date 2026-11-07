"""One run: scan the video, classify what moved, measure it, and write the video, images and tables."""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import cv2

from .classify import Classification, Rules, classify
from .errors import SettingsError
from .images import busiest_scene, heatmap, trajectories
from .model import Kind, Scan
from .optics import DEFAULT_METEOR_ALTITUDE_KM, Camera, parse_sensor
from .radiant import find_radiant
from .render import Style, write_highlights
from .report import summary, write_meteors, write_points, write_rejected, write_satellites_and_planes, write_summary
from .scan import probe, scan
from .tracking import TrackerFactory, sky_tracker_factory

log = logging.getLogger(__name__)

Progress = Callable[[str, float], None]


@dataclass(frozen=True)
class Lens:
    """What is known about the camera: a focal length with a sensor, or a horizontal field of view; and the tilt."""

    focal_mm: float | None = None
    sensor: str | None = None  # preset name or WIDTHxHEIGHT in mm; full-frame when only a focal length is given
    hfov_deg: float | None = None
    tilt_deg: float | None = None  # how far above the horizon the camera points
    roll_deg: float = 0.0
    altitude_km: float = DEFAULT_METEOR_ALTITUDE_KM  # where meteors burn, for the km/s estimate

    def camera(self, width: int, height: int) -> Camera | None:
        if self.hfov_deg:
            return Camera.from_fov(width, height, self.hfov_deg, self.tilt_deg, self.roll_deg)
        if self.focal_mm:
            sensor_width = parse_sensor(self.sensor or "full-frame")[0]
            return Camera.from_lens(width, height, self.focal_mm, sensor_width, self.tilt_deg, self.roll_deg)
        if self.tilt_deg is not None:
            raise SettingsError("a camera tilt needs the lens too: give a focal length (and sensor) or a field of view")
        return None


def parse_show(value: str | Iterable[str]) -> frozenset[Kind]:
    """'meteors', 'satellites,planes', 'all' ... as kinds."""
    words = value.split(",") if isinstance(value, str) else list(value)
    kinds: set[Kind] = set()
    for word in (w.strip().lower() for w in words if w.strip()):
        if word == "all":
            return frozenset(Kind)
        try:
            kinds.add(Kind(word.removesuffix("s")))
        except ValueError:
            raise SettingsError(f"show takes meteors, satellites, planes or all, not {word!r}") from None
    return frozenset(kinds) or frozenset({Kind.METEOR})


@dataclass
class Result:
    out_dir: Path
    found: Classification
    video: Path | None  # None when there was nothing to show or no video was asked for
    files: list[Path] = field(default_factory=list)
    summary: dict[str, Any] = field(default_factory=dict)


def process(
    video: str | Path,
    out_dir: str | Path,
    *,
    lens: Lens | None = None,
    rules: Rules | None = None,
    style: Style | None = None,
    sky_region: tuple[int, int, int, int] | None = None,
    make_video: bool = True,
    find_slow_objects: bool = True,
    tracker_factory: TrackerFactory = sky_tracker_factory,
    progress: Progress | None = None,
) -> Result:
    """Finds the meteors (and satellites and planes) in a video and writes everything into `out_dir`.

    Writes meteors.mp4 (when anything is shown), trajectories.png and heatmap.png (when there are meteors),
    meteors.csv, satellites_planes.csv, meteor_points.csv, rejected.csv and summary.json. This is scan() followed by
    analyse(); call those two yourself to try other rules or styles without scanning the video again.
    """
    lens, style = lens or Lens(), style or Style()
    report = progress or (lambda stage, fraction: None)
    info = probe(video)
    lens.camera(info.width, info.height)  # bad lens settings fail now, not after an hour of scanning
    scanned = scan(
        video,
        sky_region=sky_region,
        tracker_factory=tracker_factory,
        slow_objects=find_slow_objects or bool(style.show - {Kind.METEOR}),
        progress=lambda f: report("Finding moving objects", f),
    )
    return analyse(
        scanned,
        out_dir,
        lens=lens,
        rules=rules,
        style=style,
        sky_region=sky_region,
        make_video=make_video,
        progress=progress,
    )


def analyse(
    scanned: Scan,
    out_dir: str | Path,
    *,
    lens: Lens | None = None,
    rules: Rules | None = None,
    style: Style | None = None,
    sky_region: tuple[int, int, int, int] | None = None,
    make_video: bool = True,
    progress: Progress | None = None,
) -> Result:
    """Everything after the scan: classify, measure, and write the video, pictures and tables into `out_dir`.

    sky_region is only recorded in the summary here; it takes effect in scan().
    """
    lens, rules, style = lens or Lens(), rules or Rules(), style or Style()
    report = progress or (lambda stage, fraction: None)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    camera = lens.camera(scanned.width, scanned.height)

    report("Sorting meteors from everything else", 0.0)
    found = classify(scanned, camera, rules, lens.altitude_km)
    log.info(
        "%d meteors, %d satellites, %d planes, %d rejected",
        len(found.meteors),
        len(found.satellites),
        len(found.planes),
        len(found.rejected),
    )

    scene = busiest_scene(found.meteors)
    radiant = find_radiant([m for m in found.meteors if m.scene == scene], scanned.width)
    files: list[Path] = []
    stills = []
    if found.meteors:
        for name, image in (
            ("trajectories.png", trajectories(scanned, found.meteors, found.others, radiant)),
            ("heatmap.png", heatmap(scanned, found.meteors)),
        ):
            cv2.imwrite(str(out / name), image)
            files.append(out / name)
            stills.append(image)

    video_path: Path | None = None
    clips: list[tuple[int, int]] = []
    if make_video:
        clips = write_highlights(
            scanned,
            found.meteors,
            out / "meteors.mp4",
            others=found.others,
            style=style,
            end_cards=stills,
            progress=lambda f: report("Writing the highlight video", f),
        )
        if clips:
            video_path = out / "meteors.mp4"
            files.append(video_path)

    tables: dict[str, Callable[[Path], None]] = {
        "meteors.csv": lambda p: write_meteors(p, found.meteors),
        "satellites_planes.csv": lambda p: write_satellites_and_planes(p, found.others),
        "meteor_points.csv": lambda p: write_points(p, found.objects, camera),
        "rejected.csv": lambda p: write_rejected(p, found.rejected, scanned.fps),
    }
    for name, write in tables.items():
        write(out / name)
        files.append(out / name)
    settings = {
        "lens": asdict(lens),
        "rules": asdict(rules),
        "sky_region": sky_region,
        "show": sorted(k.value for k in style.show),
        "trails": style.trails,
    }
    data = summary(scanned, found, camera, radiant, scene, clips, settings)
    write_summary(out / "summary.json", data)
    files.append(out / "summary.json")
    report("Done", 1.0)
    return Result(out, found, video_path, files, data)
