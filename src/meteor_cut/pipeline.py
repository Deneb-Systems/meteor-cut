"""One run: scan the video, classify what moved, measure it, and write the video, images and tables."""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .astro import solar_longitude_deg
from .classify import Classification, Rules, classify
from .composite import Composite
from .enrich import apply_sky, assign_showers, radiant_in_picture, rank_by_speed, rank_meteors, top
from .errors import MeteorCutError, SettingsError
from .frames import FrameReader, median_frame, windows
from .html_report import write_html_report
from .images import busiest_scene, heatmap, satellites_and_planes, top_five, trajectories
from .model import Kind, Scan, SkyObject
from .optics import DEFAULT_METEOR_ALTITUDE_KM, Camera, parse_sensor
from .photometry import measure
from .radiant import Radiant, SkyRadiant, find_radiant, find_sky_radiant
from .render import Style, write_highlights
from .report import summary, write_meteors, write_points, write_rejected, write_satellites_and_planes, write_summary
from .scan import probe, scan
from .shorts import write_shorts
from .showers import by_code, nearest
from .sky import Site, SkyCalibration, recorded_start, sky_frame, solve
from .text import Timeline
from .tracking import TrackerFactory, sky_tracker_factory
from .wcs import Wcs

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


@dataclass(frozen=True)
class Observation:
    """When and where the video was taken, and how to find the camera's place among the stars.

    start: when the video's first frame was recorded (clock times, the shower calendar); site: where (elevations,
    km/s, whether a radiant was up). wcs: a plate solution (wcs.fits from nova.astrometry.net) of the frame taken
    wcs_at_s seconds into the video, by default the middle, where Meteor Cut's sky-frame.png comes from; or solve=True
    to plate-solve it here with astrometry.net's solve-field.
    """

    start: datetime | None = None
    site: Site | None = None
    wcs: str | Path | None = None
    wcs_at_s: float | None = None
    solve: bool = False
    use_file_time: bool = True  # fall back to the recording time stored in the file when start is not given


def calibrate(video: str | Path, duration_s: float, observation: Observation, out_dir: Path) -> SkyCalibration | None:
    """The plate solution for the video: read from a file, or solved now. Writes sky-frame.png when solving, so the
    frame can be solved elsewhere if solving here fails."""
    at_s = observation.wcs_at_s if observation.wcs_at_s is not None else duration_s / 2
    if observation.wcs is not None:
        return SkyCalibration(Wcs.read(observation.wcs), at_s)
    if not observation.solve:
        return None
    out_dir.mkdir(parents=True, exist_ok=True)
    frame = sky_frame(video, at_s)
    cv2.imwrite(str(out_dir / "sky-frame.png"), frame)
    return SkyCalibration(solve(frame), at_s)


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
    shorts: list[Path] = field(default_factory=list)


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
    observation: Observation | None = None,
    composite: bool = True,
    shorts: int = 5,
    tracker_factory: TrackerFactory = sky_tracker_factory,
    progress: Progress | None = None,
) -> Result:
    """Finds the meteors (and satellites and planes) in a video and writes everything into `out_dir`.

    Writes report.html (everything on one page), meteors.mp4 (when anything is shown), trajectories.png,
    composite.png, top.png and heatmap.png (when there are meteors), satellites.png (when there are satellites or
    planes), shorts/ (vertical clips of the best meteors), meteors.csv, satellites_planes.csv, meteor_points.csv,
    rejected.csv and summary.json. This is scan() followed by analyse(); call those two yourself to try other rules or
    styles without scanning the video again.
    """
    lens, style, observation = lens or Lens(), style or Style(), observation or Observation()
    report = progress or (lambda stage, fraction: None)
    info = probe(video)
    lens.camera(info.width, info.height)  # bad lens settings fail now, not after an hour of scanning
    sky = None
    if observation.wcs is not None or observation.solve:
        report("Finding the stars", 0.0)
        sky = calibrate(video, info.frames / info.fps if info.fps else 0.0, observation, Path(out_dir))
    scanned = scan(
        video,
        sky_region=sky_region,
        tracker_factory=tracker_factory,
        slow_objects=find_slow_objects or bool(style.show - {Kind.METEOR}),
        progress=lambda f: report("Finding moving objects", f),
    )
    start = observation.start
    if start is None and observation.use_file_time:
        start = recorded_start(video)
        if start is not None:
            log.info("start time %s, from the file's creation time", start.isoformat())
    return analyse(
        scanned,
        out_dir,
        lens=lens,
        rules=rules,
        style=style,
        sky_region=sky_region,
        make_video=make_video,
        sky=sky,
        start=start,
        start_from_file=observation.start is None and start is not None,
        site=observation.site,
        composite=composite,
        shorts=shorts,
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
    sky: SkyCalibration | None = None,
    start: datetime | None = None,
    start_from_file: bool = False,
    site: Site | None = None,
    composite: bool = True,
    shorts: int = 5,
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
    # A plate solution describes one camera position: the scene its frame came from.
    sky_scene = scanned.scene_of(round(sky.at_s * scanned.fps)) if sky is not None else -1
    if sky is not None and scanned.cuts:
        log.info("the plate solution applies to scene %d of %d", sky_scene + 1, len(scanned.cuts) + 1)
    timeline = Timeline(start, scanned.duration_s >= 3600)

    report("Sorting meteors from everything else", 0.0)
    found = classify(scanned, camera, rules, lens.altitude_km)
    log.info(
        "%d meteors, %d satellites, %d planes, %d rejected",
        len(found.meteors),
        len(found.satellites),
        len(found.planes),
        len(found.rejected),
    )

    sky_radiant: SkyRadiant | None = None
    radiant_shower = None
    scene = busiest_scene(found.meteors)
    if sky is not None:
        apply_sky(
            [o for o in found.objects if o.scene == sky_scene],
            sky,
            start=start,
            site=site,
            altitude_km=lens.altitude_km,
        )
        for kind_objects in (found.meteors, found.satellites, found.planes):
            rank_by_speed(kind_objects)
        sky_radiant = find_sky_radiant([(m.number, m.sky_path) for m in found.meteors if m.sky_path is not None])
        if sky_radiant is not None:
            middle = solar_longitude_deg(start + timedelta(seconds=scanned.duration_s / 2)) if start else None
            radiant_shower = nearest(sky_radiant.ra_deg, sky_radiant.dec_deg, middle)
        if start is not None:
            prefer = radiant_shower[0].code if radiant_shower else None
            assign_showers(found.meteors, start, site=site, prefer=prefer)
    radiant: Radiant | None
    if sky_radiant is not None and sky is not None and sky_scene == scene:
        radiant = radiant_in_picture(sky_radiant, sky, sky.wcs.scale_deg_per_px)
    else:
        radiant = find_radiant([m for m in found.meteors if m.scene == scene], scanned.width)

    composite_image = _measure_meteors(scanned, found.meteors, sky, sky_scene, scene, composite, report)
    rank_meteors(found.meteors)

    files: list[Path] = []
    images: dict[str, np.ndarray] = {}
    note = f": {radiant_shower[0].name}" if radiant_shower else ""
    if found.meteors:
        images["trajectories.png"] = trajectories(scanned, found.meteors, radiant, note)
    if composite_image is not None:
        images["composite.png"] = composite_image
    if found.meteors:
        images["top.png"] = top_five(found.meteors, timeline, scanned.width, scanned.height)
        images["heatmap.png"] = heatmap(scanned, found.meteors)
    if found.others:
        images["satellites.png"] = satellites_and_planes(scanned, found.others)
    for name, image in images.items():
        cv2.imwrite(str(out / name), image)
        files.append(out / name)
    # The video ends on the pictures of what it showed.
    stills = [image for name, image in images.items() if name != "satellites.png" or style.show - {Kind.METEOR}]

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
            timeline=timeline,
            progress=lambda f: report("Writing the highlight video", f),
        )
        if clips:
            video_path = out / "meteors.mp4"
            files.append(video_path)
    short_paths: list[Path] = []
    if shorts > 0 and found.meteors:
        short_paths = write_shorts(
            scanned.path,
            found.meteors,
            out / "shorts",
            count=shorts,
            width=scanned.width,
            height=scanned.height,
            timeline=timeline,
            progress=lambda f: report("Writing vertical clips", f),
        )
        files.extend(short_paths)

    tables: dict[str, Callable[[Path], None]] = {
        "meteors.csv": lambda p: write_meteors(p, found.meteors, timeline),
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
        "site": asdict(site) if site else None,
    }
    data = summary(scanned, found, camera, radiant, scene, clips, settings)
    data.update(_night_summary(scanned, found.meteors, sky, sky_radiant, radiant_shower, timeline, start_from_file))
    if data["sky"] is not None:
        data["sky"]["scene"] = sky_scene
    data["shorts"] = [p.name for p in short_paths]
    write_summary(out / "summary.json", data)
    files.append(out / "summary.json")
    write_html_report(
        out / "report.html",
        scanned,
        found,
        data,
        images,
        video_path.name if video_path else None,
        timeline=timeline,
        shorts=[f"shorts/{p.name}" for p in short_paths],
    )
    files.append(out / "report.html")
    report("Done", 1.0)
    return Result(out, found, video_path, files, data, short_paths)


def _measure_meteors(
    scanned: Scan,
    meteors: list[SkyObject],
    sky: SkyCalibration | None,
    sky_scene: int,
    scene: int,
    make_composite: bool,
    report: Progress,
) -> np.ndarray | None:
    """Brightness of every meteor, and the composite picture (of `scene`), from one more pass over the meteors'
    frames. Magnitudes and star alignment need the plate solution, which holds for `sky_scene` only."""
    if not meteors:
        return None
    report("Measuring the meteors", 0.0)
    try:
        with FrameReader(scanned.path) as reader:
            stacker = None
            if make_composite:
                lo, hi = scanned.scene_bounds(scene)
                aligned = sky if sky is not None and sky_scene == scene else None
                base_s = aligned.at_s if aligned is not None else (lo + hi) / 2 / scanned.fps
                middle = round(base_s * scanned.fps)
                base = median_frame(reader.read(max(lo, middle - 30), min(hi, middle + 30))[::4])
                stacker = Composite(base, base_s, aligned)
            for done, (meteor, window) in enumerate(windows(reader, meteors), start=1):
                meteor.brightness = measure(meteor, window, sky if meteor.scene == sky_scene else None)
                if stacker is not None and meteor.scene == scene:
                    stacker.add(meteor, window)
                report("Measuring the meteors", done / len(meteors))
    except MeteorCutError as error:
        log.warning("could not measure the meteors: %s", error)
        return None
    return stacker.image() if stacker is not None and stacker.count else None


def _night_summary(
    scanned: Scan,
    meteors: list[SkyObject],
    sky: SkyCalibration | None,
    sky_radiant: SkyRadiant | None,
    radiant_shower: Any,
    timeline: Timeline,
    start_from_file: bool,
) -> dict[str, Any]:
    """The summary's sky, time, shower and ranking sections."""
    counts: dict[str, int] = {}
    for m in meteors:
        if m.shower_code:
            counts[m.shower_code] = counts.get(m.shower_code, 0) + 1
    data: dict[str, Any] = {
        "start_time": timeline.iso(0),
        "start_time_from": ("file" if start_from_file else "settings") if timeline.start else None,
        "showers": dict(sorted(counts.items(), key=lambda kv: -kv[1])) or None,
        "shower_names": {code: (s.name if (s := by_code(code)) else "sporadic") for code in counts} or None,
        "fireballs": [m.label for m in meteors if m.brightness and m.brightness.fireball],
        "top_longest": [m.label for m in top(meteors, "length")],
        "top_brightest": [m.label for m in top(meteors, "brightness")],
        "sky": None,
    }
    if sky is not None:
        hfov, vfov = sky.field_deg(scanned.width, scanned.height)
        ra, dec = sky.centre(scanned.width, scanned.height)
        data["sky"] = {
            "solved_at_s": round(sky.at_s, 2),
            "field_deg": [round(hfov, 2), round(vfov, 2)],
            "centre_ra_dec_deg": [round(ra, 3), round(dec, 3)],
            "scale_arcsec_per_px": round(sky.wcs.scale_deg_per_px * 3600, 2),
            "radiant_ra_dec_deg": [round(sky_radiant.ra_deg, 2), round(sky_radiant.dec_deg, 2)]
            if sky_radiant
            else None,
            "radiant_meteors": sky_radiant.members if sky_radiant else None,
            "radiant_spread_deg": round(sky_radiant.spread_deg, 2) if sky_radiant else None,
            "radiant_shower": radiant_shower[0].code if radiant_shower else None,
            "radiant_shower_distance_deg": round(radiant_shower[1], 2) if radiant_shower else None,
        }
    return data
