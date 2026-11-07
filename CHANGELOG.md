# Changelog

All notable changes to this project are listed here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Changed

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
