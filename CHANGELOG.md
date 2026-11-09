# Changelog

All notable changes to this project are listed here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow [Semantic Versioning](https://semver.org/).

## [Unreleased]

## [0.1.1] - 2026-11-09

### Fixed

- Videos that show every picture twice (a 15 fps camera in a 30 fps stream, common on livestreams) lost every meteor
  shorter than about 0.2 s: the fast tracker sees a meteor only on new pictures, so it was detected on half the frames
  and failed the 4-detection rule. Meteor Cut now notices the repeated pictures and asks for as many detections as
  cover the same time, at least 3 (`Rules.min_repeated_detections`). On an 8-hour Perseid livestream this finds 34
  more real meteors.
- Two stars twinkling near each other no longer pass for a short meteor. A track has to be detected in at least 70 % of
  the pictures between its first and last detection (`Rules.min_seen_share`); otherwise it is rejected as
  `seen_on_and_off`.

## [0.1.0] - 2026-11-08

First release.

### Added

- Meteor detection with Sky Tracker's `MotionTracker` in `bright-diff` mode, which needs no licence key, with
  scene-cut handling for compilations.
- Rules that leave out people, lamps and phones, slow movers and anything that is not a straight line, each with a
  reason code in `rejected.csv`. They apply only to tracks slower than 15 % of the frame width per second
  (`Rules.crowd_max_speed`), so fast meteors near tree branches or noise are kept. A meteor that breaks up counts
  once; its other pieces are rejected as `meteor_fragment`.
- Satellites and planes from a second, slower tracker. Planes are told apart by strobe flashes (brief, sharp,
  repeated: `Rules.flash_ratio`, `flash_margin`, `min_flashes`) or lights flying side by side, so flaring satellites,
  faint ones and twinkling are not counted as planes.
- Angular speed from the lens (focal length and sensor, or field of view), and an estimated km/s from the tilt, or
  from the stars with `--site LAT,LON`.
- Plate solving (`--solve` with astrometry.net's solve-field, or `--wcs` with a solution from nova.astrometry.net;
  `meteor-cut sky-frame` and `meteor-cut star-index` help with both): exact angular speeds, sky coordinates, and the
  radiant among the stars, which does not smear as the sky turns through the night.
- The shower's radiant, fitted from the meteors' paths, and each meteor's shower from the IAU Meteor Data Center's
  established showers, given the start time (`--start`, or the file's creation time); the night's radiant is named
  after the closest shower.
- Brightness of every meteor, and with a plate solution an approximate magnitude from the catalogued stars in the same
  frames; fireballs (magnitude -4 or brighter) are flagged.
- Top five fastest, longest trails and brightest meteors: `top.png`, the report, the video's closing cards and
  `summary.json`, with `speed_rank`, `length_rank` and `brightness_rank` in `meteors.csv`.
- A 1920x1080 highlight video with boxes, trails, a panel of meteors sorted by speed, and closing pictures; clock
  times in the video, tables and report when the start time is known.
- `composite.png`: every meteor of the night on one clean picture of the sky, lined up on the stars when the video is
  plate-solved.
- Vertical 9:16 clips in `shorts/fastest`, `shorts/longest` and `shorts/brightest` (the top N of each list,
  `--shorts N`): a close-up over the whole frame, titled with the meteor's place on the list, then a quarter-speed
  replay.
- `report.html`, a self-contained page with the totals, meteors per hour, the pictures and sortable tables.
- `trajectories.png`, `satellites.png`, `heatmap.png`, `meteors.csv`, `satellites_planes.csv`, `meteor_points.csv`,
  `rejected.csv` and `summary.json`.
- `meteor-cut watch`: live mode for a camera, stream or file, saving each meteor's clip as it is confirmed.
- The `meteor-cut` command, a Python API (`meteor_cut.process`) and a local upload page (`meteor-cut serve`) in the
  Sky Tracker website's style, which keeps each video's results in `~/Meteor Cut/<video> <date time>/`.

[Unreleased]: https://github.com/Deneb-Systems/meteor-cut/compare/v0.1.1...HEAD
[0.1.1]: https://github.com/Deneb-Systems/meteor-cut/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/Deneb-Systems/meteor-cut/releases/tag/v0.1.0
