"""Command line: meteor-cut VIDEO [options], or meteor-cut serve."""

from __future__ import annotations

import argparse
import logging
import sys
import time
from collections.abc import Sequence
from pathlib import Path

from . import __version__
from .classify import Rules
from .errors import MeteorCutError
from .optics import SENSORS
from .pipeline import Lens, Observation, Result, parse_show, process
from .render import Style
from .sky import Site, parse_start
from .text import duration_text, length_text, speed_text, time_left_s

EXIT_OK, EXIT_ERROR, EXIT_USAGE = 0, 1, 2


def _region(value: str) -> tuple[int, int, int, int]:
    try:
        x, y, w, h = (int(v) for v in value.split(","))
    except ValueError:
        raise argparse.ArgumentTypeError("use X,Y,WIDTH,HEIGHT in pixels, for example 0,0,1920,700") from None
    if w <= 0 or h <= 0:
        raise argparse.ArgumentTypeError("width and height must be positive")
    return x, y, w, h


def _site(value: str) -> Site:
    try:
        return Site.parse(value)
    except MeteorCutError as error:
        raise argparse.ArgumentTypeError(str(error)) from None


def _start(value: str) -> object:
    try:
        return parse_start(value)
    except MeteorCutError as error:
        raise argparse.ArgumentTypeError(str(error)) from None


def _lens_arguments(parser: argparse.ArgumentParser) -> None:
    lens = parser.add_argument_group("camera (for speeds in degrees per second, and km/s)")
    lens.add_argument("--focal", type=float, metavar="MM", help="lens focal length in mm, for example 14")
    lens.add_argument(
        "--sensor", metavar="SIZE", help=f"sensor: {', '.join(SENSORS)}, or WIDTHxHEIGHT in mm (default full-frame)"
    )
    lens.add_argument("--fov", type=float, metavar="DEG", help="horizontal field of view, instead of --focal")
    lens.add_argument(
        "--tilt",
        type=float,
        metavar="DEG",
        help="how far above the horizon the camera points (0 level, 90 straight up); needed for km/s",
    )
    lens.add_argument("--roll", type=float, default=0.0, metavar="DEG", help="camera roll (default 0)")
    lens.add_argument(
        "--altitude", type=float, default=95.0, metavar="KM", help="height meteors burn at, for km/s (default 95)"
    )


def _process_parser(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("video", type=Path, help="the video file")
    parser.add_argument("-o", "--out", type=Path, help="output folder (default: <video name>-meteors beside it)")
    _lens_arguments(parser)
    night = parser.add_argument_group("time, place and stars (for showers, magnitudes and exact speeds)")
    night.add_argument(
        "--start",
        type=_start,
        metavar="TIME",
        help="when the video started, such as 2026-08-12T22:30+02:00 (default: the file's creation time, if any)",
    )
    night.add_argument("--no-file-time", action="store_true", help="do not use the file's creation time")
    night.add_argument("--site", type=_site, metavar="LAT,LON", help="where the camera stood, such as 50.45,30.52")
    night.add_argument(
        "--solve", action="store_true", help="identify the stars with astrometry.net's solve-field (if installed)"
    )
    night.add_argument(
        "--wcs",
        type=Path,
        metavar="FILE",
        help="a plate solution (wcs.fits) of sky-frame.png from nova.astrometry.net, instead of --solve",
    )
    night.add_argument(
        "--wcs-at",
        type=float,
        metavar="S",
        help="seconds into the video of the frame the --wcs solution is for (default: the middle, as sky-frame.png)",
    )
    find = parser.add_argument_group("what to find")
    find.add_argument(
        "--sky",
        type=_region,
        metavar="X,Y,W,H",
        help="look only in this part of the frame, for example the sky above the horizon",
    )
    find.add_argument(
        "--min-speed",
        type=float,
        default=Rules.min_speed,
        metavar="F",
        help=f"slowest meteor, in frame widths per second (default {Rules.min_speed})",
    )
    find.add_argument(
        "--max-duration",
        type=float,
        default=Rules.max_duration_s,
        metavar="S",
        help=f"longest meteor, in seconds (default {Rules.max_duration_s})",
    )
    video = parser.add_argument_group("the highlight video")
    video.add_argument(
        "--show",
        default="meteors",
        metavar="WHAT",
        help="meteors (default), satellites, planes, a comma-separated list, or all",
    )
    video.add_argument("--no-trails", action="store_true", help="do not keep past trails faintly on screen")
    video.add_argument("--no-video", action="store_true", help="write only the images and tables")
    video.add_argument(
        "--shorts",
        type=int,
        default=5,
        metavar="N",
        help="vertical 9:16 clips of the N fastest, N longest and N brightest meteors (default 5, 0 for none)",
    )
    video.add_argument("--no-composite", action="store_true", help="skip the picture of every meteor on one sky")


def _watch_parser(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("source", help="camera number (0 = first camera), stream URL (rtsp://...) or a video file")
    parser.add_argument("-o", "--out", type=Path, required=True, help="folder for the clips and tables")
    parser.add_argument("--minutes", type=float, metavar="M", help="stop after M minutes (default: until Ctrl+C)")
    parser.add_argument("--sky", type=_region, metavar="X,Y,W,H", help="look only in this part of the frame")
    _lens_arguments(parser)


def _sky_frame_parser(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("video", type=Path, help="the video file")
    parser.add_argument("-o", "--out", type=Path, default=Path("sky-frame.png"), help="picture to write")
    parser.add_argument("--at", type=float, metavar="S", help="seconds into the video (default: the middle)")


def _serve_parser(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--host", default="127.0.0.1", help="address to listen on (default 127.0.0.1, this computer)")
    parser.add_argument("--port", type=int, default=8765, help="port (default 8765)")
    parser.add_argument("--max-upload-gb", type=float, default=20.0, help="largest upload accepted (default 20)")
    parser.add_argument(
        "--results", type=Path, metavar="DIR", help="where each video's results are kept (default: ~/Meteor Cut)"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="meteor-cut",
        description="Find the meteors in a night-sky video from a camera on a tripod, and make a highlight video, "
        "pictures of every trajectory and a heat map, and tables of every meteor, satellite and plane.",
        epilog="Run 'meteor-cut VIDEO --help' for the options, or 'meteor-cut serve' for the upload page.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    verbosity = parser.add_mutually_exclusive_group()
    verbosity.add_argument("-v", "--verbose", action="store_true", help="explain each step")
    verbosity.add_argument("-q", "--quiet", action="store_true", help="print only errors")
    commands = parser.add_subparsers(dest="command", metavar="COMMAND")
    _process_parser(commands.add_parser("process", help="process a video (the default command)"))
    _serve_parser(commands.add_parser("serve", help="start the upload page on this computer"))
    _watch_parser(commands.add_parser("watch", help="watch a camera live and keep only the meteors"))
    _sky_frame_parser(
        commands.add_parser("sky-frame", help="a clean star picture of a video, to plate-solve at nova.astrometry.net")
    )
    commands.add_parser("star-index", help="download the star index files solve-field needs for wide fields (1.6 MB)")
    return parser


class _Progress:
    """One updating line on stderr per stage, with the time left, when stderr is a terminal."""

    def __init__(self, enabled: bool):
        self.enabled = enabled and sys.stderr.isatty()
        self.stage = ""
        self.started = 0.0
        self.start_fraction = 0.0

    def __call__(self, stage: str, fraction: float) -> None:
        if not self.enabled:
            return
        if stage != self.stage:
            if self.stage:
                print(file=sys.stderr)
            self.stage, self.started, self.start_fraction = stage, time.monotonic(), fraction
        left = time_left_s(fraction, self.start_fraction, time.monotonic() - self.started)
        eta = f"  ({duration_text(left)} left)" if left else ""
        line = f"\r{stage}: {int(fraction * 10000) / 100:6.2f}%{eta}"
        print(f"{line:<70}", end="", file=sys.stderr, flush=True)

    def close(self) -> None:
        if self.enabled and self.stage:
            print(file=sys.stderr)


def _print_result(result: Result, seconds: float) -> None:
    s = result.summary
    rate = f" ({s['meteors_per_hour']} per hour)" if s["meteors_per_hour"] else ""
    print(f"{s['meteors']} meteors in {s['duration_s']:.0f} s of video{rate}; processed in {seconds:.0f} s.")
    if s["angular_speed_deg_s"]:
        a = s["angular_speed_deg_s"]
        print(f"Angular speed: mean {a['mean']} deg/s, fastest {a['max']} deg/s ({s['fastest_meteor']}).")
    if s["sky_speed_km_s"]:
        print(f"Speed across the view: at least {s['sky_speed_km_s']['mean']} km/s on average.")
    if s["satellites"] or s["planes"]:
        print(f"Also {s['satellites']} satellites and {s['planes']} planes.")
    if s["radiant"]:
        r = s["radiant"]
        where = "" if r["in_frame"] else " (outside the frame)"
        print(f"Radiant: {r['meteors']} meteors trace back to x={r['x_px']:.0f}, y={r['y_px']:.0f}{where}.")
    sky = s.get("sky")
    if sky:
        print(f"Stars identified: {sky['field_deg'][0]:.1f} x {sky['field_deg'][1]:.1f} degrees of sky.")
        if sky["radiant_ra_dec_deg"]:
            ra, dec = sky["radiant_ra_dec_deg"]
            named = f" ({sky['radiant_shower']})" if sky["radiant_shower"] else ""
            print(f"Radiant among the stars: RA {ra:.1f}, Dec {dec:+.1f}{named}.")
    if s.get("showers"):
        print("Showers: " + ", ".join(f"{n} {code}" for code, n in s["showers"].items()) + ".")
    if s.get("fireballs"):
        print(f"Fireballs: {', '.join(s['fireballs'])}.")
    if s.get("top_longest"):
        print(f"Longest trails: {', '.join(s['top_longest'])}; brightest: {', '.join(s['top_brightest']) or '-'}.")
    print(f"Results in {result.out_dir}:")
    for path in result.files:
        print(f"  {path.relative_to(result.out_dir)}")


def _process(args: argparse.Namespace) -> int:
    if not args.video.is_file():
        print(f"meteor-cut: no such video: {args.video}", file=sys.stderr)
        return EXIT_USAGE
    out = args.out or args.video.with_name(args.video.stem + "-meteors")
    progress = _Progress(enabled=not args.quiet)
    started = time.monotonic()
    try:
        lens = _lens(args)
        style = Style(show=parse_show(args.show), trails=not args.no_trails)
        rules = Rules(min_speed=args.min_speed, max_duration_s=args.max_duration)
        observation = Observation(
            start=args.start,
            site=args.site,
            wcs=args.wcs,
            wcs_at_s=args.wcs_at,
            solve=args.solve,
            use_file_time=not args.no_file_time,
        )
        result = process(
            args.video,
            out,
            lens=lens,
            rules=rules,
            style=style,
            sky_region=args.sky,
            make_video=not args.no_video,
            observation=observation,
            composite=not args.no_composite,
            shorts=max(0, args.shorts),
            progress=progress,
        )
    except MeteorCutError as error:
        progress.close()
        print(f"meteor-cut: {error}", file=sys.stderr)
        return EXIT_ERROR
    progress.close()
    if not args.quiet:
        _print_result(result, time.monotonic() - started)
    return EXIT_OK


def _lens(args: argparse.Namespace) -> Lens:
    return Lens(
        focal_mm=args.focal,
        sensor=args.sensor,
        hfov_deg=args.fov,
        tilt_deg=args.tilt,
        roll_deg=args.roll,
        altitude_km=args.altitude,
    )


def _watch(args: argparse.Namespace) -> int:
    import threading

    from .live import LiveMeteor, Watcher

    stop = threading.Event()

    def announce(found: LiveMeteor) -> None:
        if not args.quiet:
            m = found.meteor
            saved = f", saved {found.clip.name}" if found.clip else ""
            print(f"{found.time:%H:%M:%S}  meteor {m.number}: {speed_text(m)[0]}, {length_text(m)}{saved}")

    try:
        lens = _lens(args)
        watcher = Watcher(
            args.source,
            args.out,
            camera_lens=lens.camera,
            altitude_km=args.altitude,
            sky_region=args.sky,
            on_meteor=announce,
        )
        if not args.quiet:
            print(f"Watching {args.source}; press Ctrl+C to stop.", file=sys.stderr)
        try:
            result = watcher.run(stop=stop, max_seconds=args.minutes * 60 if args.minutes else None)
        except KeyboardInterrupt:  # pragma: no cover - interactive
            stop.set()
            return EXIT_OK
    except MeteorCutError as error:
        print(f"meteor-cut: {error}", file=sys.stderr)
        return EXIT_ERROR
    if not args.quiet:
        print(f"{len(result.meteors)} meteors in {result.frames} frames. Results in {result.out_dir}.")
    return EXIT_OK


def _sky_frame(args: argparse.Namespace) -> int:
    import cv2

    from .scan import probe
    from .sky import sky_frame

    try:
        info = probe(args.video)
        at = args.at if args.at is not None else info.frames / info.fps / 2
        cv2.imwrite(str(args.out), sky_frame(args.video, at))
    except MeteorCutError as error:
        print(f"meteor-cut: {error}", file=sys.stderr)
        return EXIT_ERROR
    print(f"Wrote {args.out} ({at:.1f} s into the video). Upload it at https://nova.astrometry.net/upload, download")
    print(f"wcs.fits from the result page, and run: meteor-cut {args.video} --wcs wcs.fits --wcs-at {at:.1f}")
    return EXIT_OK


def _star_index(_: argparse.Namespace) -> int:
    from .sky import download_indexes, find_solver

    try:
        folder = download_indexes()
    except OSError as error:
        print(f"meteor-cut: could not download the index files: {error}", file=sys.stderr)
        return EXIT_ERROR
    print(f"Star index files in {folder}.")
    if find_solver() is None:
        print("solve-field is not installed yet: brew install astrometry-net (macOS) or apt install astrometry.net.")
    return EXIT_OK


def _serve(args: argparse.Namespace) -> int:
    from .web import serve

    serve(args.host, args.port, max_upload_bytes=int(args.max_upload_gb * 1024**3), results_dir=args.results)
    return EXIT_OK


def main(argv: Sequence[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    # "process" is the default command: meteor-cut night.mp4 is meteor-cut process night.mp4.
    at = 0
    while at < len(argv) and argv[at] in ("-v", "--verbose", "-q", "--quiet"):
        at += 1
    if at < len(argv) and argv[at] not in (*_COMMANDS, "-h", "--help", "--version"):
        argv.insert(at, "process")
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help()
        return EXIT_USAGE
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING, format="%(levelname)s %(name)s: %(message)s"
    )
    return int(_COMMANDS[args.command](args))


_COMMANDS = {
    "process": _process,
    "serve": _serve,
    "watch": _watch,
    "sky-frame": _sky_frame,
    "star-index": _star_index,
}


if __name__ == "__main__":
    sys.exit(main())
