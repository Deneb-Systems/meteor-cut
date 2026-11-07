"""CSV and JSON output: one row per meteor, one row per tracked position, and a summary."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

from .meteors import Meteor, Rejected
from .optics import Camera
from .scan import Scan


def _num(value: float | None, digits: int = 2):
    return "" if value is None or not math.isfinite(value) else round(value, digits)


def write_meteors_csv(path: Path, meteors: list[Meteor]) -> None:
    columns = ["number", "speed_rank", "track_id", "scene", "start_time_s", "end_time_s", "duration_s",
               "start_frame", "end_frame", "start_x", "start_y", "end_x", "end_y", "length_px", "speed_px_s",
               "direction_deg", "straightness", "angular_length_deg", "angular_speed_deg_s",
               "elevation_start_deg", "elevation_end_deg", "range_km", "sky_speed_km_s"]
    with path.open("w", newline="") as f:
        out = csv.writer(f)
        out.writerow(columns)
        for m in meteors:
            sp = m.speed
            out.writerow([
                m.extra["number"], m.rank, m.id, m.scene, _num(m.start_s, 3), _num(m.end_s, 3), _num(m.duration_s, 3),
                m.start_frame, m.end_frame, _num(m.start[0], 1), _num(m.start[1], 1), _num(m.end[0], 1),
                _num(m.end[1], 1), _num(m.length_px, 1), _num(m.speed_px_s, 1), _num(m.direction_deg, 1),
                _num(m.straightness, 3),
                _num(sp.angular_length_deg if sp else None), _num(sp.angular_speed_deg_s if sp else None),
                _num(sp.elevation_start_deg if sp else None, 1), _num(sp.elevation_end_deg if sp else None, 1),
                _num(sp.range_km if sp else None, 0), _num(sp.sky_speed_km_s if sp else None, 1),
            ])


def write_points_csv(path: Path, meteors: list[Meteor], fps: float, camera: Camera | None) -> None:
    with path.open("w", newline="") as f:
        out = csv.writer(f)
        out.writerow(["object", "kind", "frame", "time_s", "x", "y", "box_w", "box_h", "brightness",
                      "elevation_deg"])
        for m in meteors:
            for p in m.points:
                elevation = camera.elevation_deg(p.cx, p.cy) if camera else None
                out.writerow([m.label, m.kind, p.frame, _num(p.frame / fps, 3), _num(p.cx, 1), _num(p.cy, 1),
                              _num(p.w, 1), _num(p.h, 1), p.peak, _num(elevation, 2)])


def write_objects_csv(path: Path, objects: list[Meteor]) -> None:
    """Satellites and planes: one row each."""
    with path.open("w", newline="") as f:
        out = csv.writer(f)
        out.writerow(["object", "kind", "start_time_s", "end_time_s", "duration_s", "start_x", "start_y", "end_x",
                      "end_y", "length_px", "speed_px_s", "direction_deg", "angular_speed_deg_s",
                      "brightness_variation", "detected_share", "travels_with_another_light"])
        for o in objects:
            out.writerow([o.label, o.kind, _num(o.start_s, 2), _num(o.end_s, 2), _num(o.duration_s, 2),
                          _num(o.start[0], 1), _num(o.start[1], 1), _num(o.end[0], 1), _num(o.end[1], 1),
                          _num(o.length_px, 1), _num(o.speed_px_s, 1), _num(o.direction_deg, 1),
                          _num(o.speed.angular_speed_deg_s if o.speed else None),
                          o.extra.get("brightness_variation"), o.extra.get("detected_share"),
                          o.extra.get("companion")])


def write_rejected_csv(path: Path, rejected: list[Rejected], fps: float) -> None:
    with path.open("w", newline="") as f:
        out = csv.writer(f)
        out.writerow(["track_id", "scene", "start_time_s", "end_time_s", "reason"])
        for r in sorted(rejected, key=lambda r: r.start_frame):
            out.writerow([r.id, r.scene, _num(r.start_frame / fps, 3), _num(r.end_frame / fps, 3), r.reason])


def summary(scan: Scan, meteors: list[Meteor], rejected: list[Rejected], camera: Camera | None,
            clips: list[tuple[int, int]], settings: dict, others: list[Meteor] = (), radiant=None,
            radiant_scene: int = 0) -> dict:
    hours = scan.frames / scan.fps / 3600 if scan.fps else 0
    speeds = [m.speed.angular_speed_deg_s for m in meteors if m.speed]
    sky = [m.speed.sky_speed_km_s for m in meteors if m.speed and m.speed.sky_speed_km_s]
    fastest = min(meteors, key=lambda m: m.rank) if meteors else None
    return {
        "video": scan.path,
        "duration_s": round(scan.frames / scan.fps, 2) if scan.fps else None,
        "fps": round(scan.fps, 3),
        "size": [scan.width, scan.height],
        "scenes": len(scan.cuts) + 1,
        "meteors": len(meteors),
        # A rate from a short clip is meaningless; give it from 10 minutes of video.
        "meteors_per_hour": round(len(meteors) / hours, 1) if hours >= 1 / 6 else None,
        "fastest": fastest.extra["number"] if fastest else None,
        "satellites": sum(1 for o in others if o.kind == "satellite"),
        "planes": sum(1 for o in others if o.kind == "plane"),
        "radiant": {"x_px": round(radiant.x, 1), "y_px": round(radiant.y, 1), "meteors": radiant.members,
                    "spread_px": round(radiant.spread_px, 1), "scene": radiant_scene,
                    "in_frame": 0 <= radiant.x < scan.width and 0 <= radiant.y < scan.height,
                    "elevation_deg": _num(camera.elevation_deg(radiant.x, radiant.y), 1) if camera else None}
        if radiant else None,
        "angular_speed_deg_s": {"mean": round(sum(speeds) / len(speeds), 2), "max": round(max(speeds), 2),
                                "min": round(min(speeds), 2)} if speeds else None,
        "sky_speed_km_s": {"mean": round(sum(sky) / len(sky), 1), "max": round(max(sky), 1)} if sky else None,
        "camera": {"horizontal_fov_deg": round(camera.horizontal_fov_deg, 2), "tilt_deg": camera.tilt_deg,
                   "roll_deg": camera.roll_deg} if camera else None,
        "highlight_clips": [[round(a / scan.fps, 2), round(b / scan.fps, 2)] for a, b in clips],
        "rejected": {reason: sum(1 for r in rejected if r.reason == reason)
                     for reason in sorted({r.reason for r in rejected})},
        "settings": settings,
    }


def write_summary(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")
