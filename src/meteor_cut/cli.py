"""meteor-cut VIDEO [options]   or   meteor-cut serve"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from .licence import SkyTrackerUnavailable
from .meteors import Rules
from .optics import SENSORS
from .pipeline import Lens, run


def _region(value: str) -> tuple[int, int, int, int]:
    try:
        x, y, w, h = (int(v) for v in value.split(","))
    except ValueError:
        raise argparse.ArgumentTypeError("use X,Y,W,H in pixels, for example 0,0,1920,700") from None
    return x, y, w, h


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="meteor-cut",
        description="Find the meteors in a video from a camera on a tripod, and make a highlight video with only "
                    "them, their speeds and a CSV of every meteor. Run 'meteor-cut serve' for the upload page.")
    p.add_argument("video", help="video file, or 'serve' to start the upload page")
    p.add_argument("-o", "--out", help="output folder (default: <video name>-meteors beside the video)")
    lens = p.add_argument_group("camera, for speeds in degrees per second and km/s")
    lens.add_argument("--focal", type=float, metavar="MM", help="lens focal length in mm, for example 14")
    lens.add_argument("--sensor", metavar="SIZE",
                      help=f"sensor: {', '.join(SENSORS)}, or WIDTHxHEIGHT in mm (default full-frame)")
    lens.add_argument("--fov", type=float, metavar="DEG", help="horizontal field of view instead of --focal")
    lens.add_argument("--tilt", type=float, metavar="DEG",
                      help="how far above the horizon the camera points (0 level, 90 straight up); needed for km/s")
    lens.add_argument("--roll", type=float, default=0.0, metavar="DEG", help="camera roll (default 0)")
    lens.add_argument("--altitude", type=float, default=95.0, metavar="KM",
                      help="height meteors burn at, for km/s (default 95)")
    p.add_argument("--sky", type=_region, metavar="X,Y,W,H",
                   help="look for meteors only in this part of the frame, for example above the horizon")
    p.add_argument("--min-speed", type=float, default=Rules.min_speed, metavar="F",
                   help=f"slowest meteor, in frame widths per second (default {Rules.min_speed})")
    p.add_argument("--max-duration", type=float, default=Rules.max_duration_s, metavar="S",
                   help=f"longest meteor in seconds (default {Rules.max_duration_s})")
    p.add_argument("--no-video", action="store_true", help="write only the CSV and JSON files")
    p.add_argument("--host", default="127.0.0.1", help="serve: address to listen on")
    p.add_argument("--port", type=int, default=8765, help="serve: port (default 8765)")
    return p


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.video == "serve":
        from .web import serve
        serve(args.host, args.port)
        return 0

    video = Path(args.video)
    if not video.is_file():
        print(f"meteor-cut: no such video: {video}", file=sys.stderr)
        return 2
    out = Path(args.out) if args.out else video.with_name(video.stem + "-meteors")
    lens = Lens(focal_mm=args.focal, sensor=args.sensor, hfov_deg=args.fov, tilt_deg=args.tilt, roll_deg=args.roll,
                altitude_km=args.altitude)
    rules = Rules(min_speed=args.min_speed, max_duration_s=args.max_duration)

    last = [""]
    started = time.time()

    def progress(stage: str, fraction: float) -> None:
        line = f"\r{stage}: {fraction * 100:5.1f}%"
        if stage != last[0] and last[0]:
            print(file=sys.stderr)
        last[0] = stage
        print(line, end="", file=sys.stderr, flush=True)

    try:
        result = run(str(video), out, lens, sky_region=args.sky, rules=rules, make_video=not args.no_video,
                     progress=progress)
    except (SkyTrackerUnavailable, ValueError) as error:
        print(f"\nmeteor-cut: {error}", file=sys.stderr)
        return 1
    print(file=sys.stderr)
    s = result.summary
    rate = f" ({s['meteors_per_hour']} per hour)" if s["meteors_per_hour"] else ""
    print(f"{result.meteors} meteors in {s['duration_s']:.0f} s of video{rate}, {time.time() - started:.0f} s to process.")
    if s["angular_speed_deg_s"]:
        a = s["angular_speed_deg_s"]
        print(f"Angular speed: mean {a['mean']} deg/s, fastest {a['max']} deg/s (meteor #{s['fastest']}).")
    if s["sky_speed_km_s"]:
        print(f"Speed across the sky: at least {s['sky_speed_km_s']['mean']} km/s on average.")
    if result.video:
        print(f"Highlights: {result.video}")
    print(f"Tables: {result.out_dir / 'meteors.csv'}, meteor_points.csv, rejected.csv, summary.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
