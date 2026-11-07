# Changelog

All notable changes to this project are listed here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added

- `composite.png`: every meteor of the night on one clean picture of the sky, lined up on the stars when the video is
  plate-solved, as astrophotographers make by hand.
- Vertical 9:16 clips in `shorts/fastest`, `shorts/longest` and `shorts/brightest` (the top N of each list,
  `--shorts N`): a close-up over the whole frame, titled with the meteor's place on the list, then a quarter-speed
  replay.
- Brightness of every meteor, and with a plate solution an approximate magnitude from the catalogued stars in the same
  frames; fireballs (magnitude -4 or brighter) are flagged.
- Top five fastest, longest trails and brightest meteors: `top.png`, the report, the video's closing cards,
  `summary.json`, and `length_rank` / `brightness_rank` in `meteors.csv` beside `speed_rank`.
- Plate solving (`--solve` with astrometry.net's solve-field, or `--wcs` with a solution from nova.astrometry.net;
  `meteor-cut sky-frame` and `meteor-cut star-index` help with both): exact angular speeds, sky coordinates, and the
  radiant among the stars, which no longer smears as the sky turns through the night.
- Each meteor's shower from the IAU Meteor Data Center's established showers, given the start time (`--start`, or the
  file's creation time), and the night's radiant named after the closest shower.
- `--site LAT,LON` for elevations and km/s from the stars, without a camera tilt.
- Clock times in the video, tables and report when the start time is known.
- `meteor-cut watch`: live mode for a camera, stream or file, saving each meteor's clip as it is confirmed.
- The upload page takes the start time, the site, a plate solution or a request to solve, the composite and the
  number of vertical clips, and shows the new results.

### Changed

- The upload page keeps each video's results in `~/Meteor Cut/<video> <date time>/` (`meteor-cut serve --results DIR`)
  instead of a temporary folder deleted when the server stopped, and shows where they are.

- Planes are told from satellites by strobe flashes (brief, sharp, repeated) instead of any brightness spread. Flaring
  satellites, faint ones the tracker briefly loses, and twinkling no longer count as planes: on an 8-hour night 81
  "planes" became 1. `Rules.blink_variation` is replaced by `flash_ratio`, `flash_margin` and `min_flashes`;
  `satellites_planes.csv` gains a `flashes` column.

- The upload page and report match the Sky Tracker website: Manrope (bundled, works offline), large type, rounded
  cards, drag-and-drop upload with progress, and the report titled with the video's own name.

## [0.1.0] - 2026-11-07

First release.

### Added

- Meteor detection with Sky Tracker's `MotionTracker` (`bright-diff`), with scene-cut handling for compilations.
- Rules that leave out people, lamps and phones, slow movers and anything that is not a straight line, each with a
  reason code in `rejected.csv`.
- Satellites and planes from a second, slower tracker; planes told apart by blinking or lights flying side by side.
- Angular speed from the lens (focal length and sensor, or field of view), and an estimated km/s from the tilt.
- The shower's radiant, fitted from the meteors' paths.
- A 1920x1080 highlight video with boxes, trails, a panel of meteors sorted by speed, and closing pictures.
- `report.html`, a self-contained page with the totals, meteors per hour, the pictures and sortable tables.
- `trajectories.png`, `satellites.png`, `heatmap.png`, `meteors.csv`, `satellites_planes.csv`, `meteor_points.csv`,
  `rejected.csv` and `summary.json`.
- The `meteor-cut` command, a local upload page (`meteor-cut serve`) and a Python API (`meteor_cut.process`).

[Unreleased]: https://github.com/Deneb-Systems/meteor-cut/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/Deneb-Systems/meteor-cut/releases/tag/v0.1.0
