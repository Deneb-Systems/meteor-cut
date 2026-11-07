"""CSV and JSON output: one row per object, one row per tracked position, what was rejected, and a summary."""

from __future__ import annotations

import csv
import json
import math
from collections import Counter
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

from .classify import Classification
from .model import Rejection, Scan, SkyObject
from .optics import Camera
from .radiant import Radiant


def _num(value: float | None, digits: int = 2) -> float | str:
    return "" if value is None or not math.isfinite(value) else round(value, digits)


def _write(path: Path, header: Sequence[str], rows: Iterable[Sequence[Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        out = csv.writer(f)
        out.writerow(header)
        out.writerows(rows)


METEOR_COLUMNS = (
    "number",
    "speed_rank",
    "track_id",
    "scene",
    "start_time_s",
    "end_time_s",
    "duration_s",
    "start_frame",
    "end_frame",
    "start_x",
    "start_y",
    "end_x",
    "end_y",
    "length_px",
    "speed_px_s",
    "direction_deg",
    "straightness",
    "angular_length_deg",
    "angular_speed_deg_s",
    "elevation_start_deg",
    "elevation_end_deg",
    "range_km",
    "sky_speed_km_s",
)


def write_meteors(path: Path, meteors: Sequence[SkyObject]) -> None:
    def row(m: SkyObject) -> list[Any]:
        mo = m.motion
        return [
            m.number,
            m.rank,
            m.track_id,
            m.scene,
            _num(m.start_s, 3),
            _num(m.end_s, 3),
            _num(m.duration_s, 3),
            m.start_frame,
            m.end_frame,
            _num(m.start[0], 1),
            _num(m.start[1], 1),
            _num(m.end[0], 1),
            _num(m.end[1], 1),
            _num(m.length_px, 1),
            _num(m.speed_px_s, 1),
            _num(m.direction_deg, 1),
            _num(m.straightness, 3),
            _num(mo.angular_length_deg if mo else None),
            _num(mo.angular_speed_deg_s if mo else None),
            _num(mo.elevation_start_deg if mo else None, 1),
            _num(mo.elevation_end_deg if mo else None, 1),
            _num(mo.range_km if mo else None, 0),
            _num(mo.sky_speed_km_s if mo else None, 1),
        ]

    _write(path, METEOR_COLUMNS, map(row, meteors))


OBJECT_COLUMNS = (
    "object",
    "kind",
    "speed_rank",
    "start_time_s",
    "end_time_s",
    "duration_s",
    "start_x",
    "start_y",
    "end_x",
    "end_y",
    "length_px",
    "speed_px_s",
    "direction_deg",
    "angular_speed_deg_s",
    "brightness_variation",
    "detected_share",
    "travels_with_another_light",
)


def write_satellites_and_planes(path: Path, objects: Sequence[SkyObject]) -> None:
    def row(o: SkyObject) -> list[Any]:
        d = o.diagnostics
        return [
            o.label,
            o.kind.value,
            o.rank,
            _num(o.start_s),
            _num(o.end_s),
            _num(o.duration_s),
            _num(o.start[0], 1),
            _num(o.start[1], 1),
            _num(o.end[0], 1),
            _num(o.end[1], 1),
            _num(o.length_px, 1),
            _num(o.speed_px_s, 1),
            _num(o.direction_deg, 1),
            _num(o.motion.angular_speed_deg_s if o.motion else None),
            d.get("brightness_variation", ""),
            d.get("detected_share", ""),
            d.get("travels_with_another_light", ""),
        ]

    _write(path, OBJECT_COLUMNS, map(row, objects))


def write_points(path: Path, objects: Sequence[SkyObject], camera: Camera | None) -> None:
    def rows() -> Iterable[list[Any]]:
        for o in objects:
            for p in o.points:
                elevation = camera.elevation_deg(p.cx, p.cy) if camera else None
                yield [
                    o.label,
                    o.kind.value,
                    p.frame,
                    _num(p.frame / o.fps, 3),
                    _num(p.cx, 1),
                    _num(p.cy, 1),
                    _num(p.w, 1),
                    _num(p.h, 1),
                    p.peak,
                    _num(elevation, 2),
                ]

    _write(
        path, ("object", "kind", "frame", "time_s", "x", "y", "box_w", "box_h", "brightness", "elevation_deg"), rows()
    )


def write_rejected(path: Path, rejected: Sequence[Rejection], fps: float) -> None:
    _write(
        path,
        ("track_id", "scene", "start_time_s", "end_time_s", "reason", "description"),
        (
            [
                r.track_id,
                r.scene,
                _num(r.start_frame / fps, 3),
                _num(r.end_frame / fps, 3),
                r.reason.value,
                r.reason.description,
            ]
            for r in rejected
        ),
    )


def _stats(values: Sequence[float], digits: int) -> dict[str, float] | None:
    if not values:
        return None
    return {
        "mean": round(sum(values) / len(values), digits),
        "min": round(min(values), digits),
        "max": round(max(values), digits),
    }


def summary(
    scan: Scan,
    found: Classification,
    camera: Camera | None,
    radiant: Radiant | None,
    radiant_scene: int,
    clips: Sequence[tuple[int, int]],
    settings: dict[str, Any],
) -> dict[str, Any]:
    from . import __version__
    from .tracking import sdk_version

    meteors = found.meteors
    hours = scan.duration_s / 3600
    fastest = min(meteors, key=lambda m: m.rank) if meteors else None
    return {
        "meteor_cut_version": __version__,
        "sky_tracker_version": sdk_version(),
        "video": scan.path,
        "duration_s": round(scan.duration_s, 2),
        "fps": round(scan.fps, 3),
        "size": [scan.width, scan.height],
        "scenes": len(scan.cuts) + 1,
        "meteors": len(meteors),
        # A rate from a short clip means little; it is given from 10 minutes of video.
        "meteors_per_hour": round(len(meteors) / hours, 1) if hours >= 1 / 6 else None,
        "fastest_meteor": fastest.label if fastest else None,
        "angular_speed_deg_s": _stats([m.motion.angular_speed_deg_s for m in meteors if m.motion], 2),
        "sky_speed_km_s": _stats([m.motion.sky_speed_km_s for m in meteors if m.motion and m.motion.sky_speed_km_s], 1),
        "satellites": len(found.satellites),
        "planes": len(found.planes),
        "radiant": None
        if radiant is None
        else {
            "x_px": round(radiant.x, 1),
            "y_px": round(radiant.y, 1),
            "meteors": radiant.members,
            "spread_px": round(radiant.spread_px, 1),
            "scene": radiant_scene,
            "in_frame": radiant.in_frame(scan.width, scan.height),
            "elevation_deg": _num(camera.elevation_deg(radiant.x, radiant.y), 1) if camera else None,
        },
        "camera": None
        if camera is None
        else {
            "horizontal_fov_deg": round(camera.horizontal_fov_deg, 2),
            "tilt_deg": camera.tilt_deg,
            "roll_deg": camera.roll_deg,
        },
        "highlight_clips_s": [[round(a / scan.fps, 2), round(b / scan.fps, 2)] for a, b in clips],
        "rejected": dict(sorted(Counter(r.reason.value for r in found.rejected).items())),
        "settings": settings,
    }


def write_summary(path: Path, data: dict[str, Any]) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
