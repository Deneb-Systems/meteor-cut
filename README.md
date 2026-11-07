# Meteor Cut

[![CI](https://github.com/Deneb-Systems/meteor-cut/actions/workflows/ci.yml/badge.svg)](https://github.com/Deneb-Systems/meteor-cut/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/meteor-cut)](https://pypi.org/project/meteor-cut/)
[![Python](https://img.shields.io/pypi/pyversions/meteor-cut)](https://pypi.org/project/meteor-cut/)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue)](LICENSE)

Point a camera at the night sky, record for an hour or a whole night, and give Meteor Cut the video. It finds every
meteor and cuts a highlight video of just those moments, with each meteor boxed, its trail drawn and its speed shown,
and a panel listing them all from fastest to slowest. It also picks out satellites and planes, draws every trajectory
of the night on one picture with the shower's radiant, maps where meteors appeared, and writes it all to CSV.

Meteors are found and followed by [Sky Tracker](https://sky-tracker.deneb-systems.com)'s `MotionTracker`, which looks
for what got brighter since the previous frame: the head of a meteor rather than the trail it leaves.

## Install

```bash
pip install meteor-cut
```

Meteor Cut needs Python 3.10 or later and a Sky Tracker licence key. A free 7-day trial key is one short form away at
[sky-tracker.deneb-systems.com/trial](https://sky-tracker.deneb-systems.com/trial/); then set it in your shell:

```bash
export SKY_TRACKER_LICENSE_KEY=SKT2....        # Windows PowerShell: $env:SKY_TRACKER_LICENSE_KEY = "SKT2...."
```

## Quick start

```bash
meteor-cut night.mp4
meteor-cut night.mp4 --focal 14 --sensor full-frame --tilt 40 --show all
```

Or start the upload page and use it from your browser. It runs on your computer; videos never leave it.

```bash
meteor-cut serve            # then open http://127.0.0.1:8765
```

From Python:

```python
from meteor_cut import Lens, process

result = process("night.mp4", "night-meteors", lens=Lens(focal_mm=14, tilt_deg=40))
for meteor in result.found.meteors:
    print(meteor.label, f"{meteor.start_s:.1f} s", f"{meteor.motion.angular_speed_deg_s:.1f} deg/s")
```

## What you get

Results go to `night-meteors/` beside the video (choose with `-o`):

| File | What is in it |
|---|---|
| `report.html` | Everything on one page to read or share: totals, meteors per hour, the pictures, sortable tables |
| `meteors.mp4` | The highlight video, 1920x1080, ending on the pictures below |
| `trajectories.png` | Every meteor of the night, coloured by speed, and the radiant they trace back to |
| `satellites.png` | Every satellite and plane |
| `heatmap.png` | Where in the frame meteors appeared |
| `meteors.csv` | One row per meteor: time, position, length, direction, speed and rank by speed |
| `satellites_planes.csv` | One row per satellite or plane |
| `meteor_points.csv` | Every tracked position of every object, frame by frame |
| `rejected.csv` | Everything that moved but was not kept, with the reason (people, lamps, too slow, ...) |
| `summary.json` | Totals, rates, the camera, the radiant, versions and every setting used |

## Speeds

| You give | You get |
|---|---|
| Nothing | Speed on screen, in pixels per second |
| The lens: `--focal` with `--sensor`, or `--fov` | Angular speed in degrees per second, which is what meteor observers compare |
| The lens and `--tilt` (degrees above the horizon) | Also an estimate in km/s |

One camera cannot see a meteor move towards or away from it, so the km/s figure is the speed across your view, at the
height meteors burn (95 km by default, `--altitude`): the real speed is at least that. Values well above 72 km/s, the
fastest a meteor can enter the atmosphere, mean the lens or tilt is wrong. The lens is assumed to be rectilinear (a
normal wide-angle lens, not a fisheye), and the video to show the sensor's full width.

## Options worth knowing

| Option | Use it when |
|---|---|
| `--sky X,Y,W,H` | Trees, people or lights on the ground get in the way: look only at the sky |
| `--show meteors,satellites,planes` or `--show all` | You want satellites and planes in the video too (they are always in the tables) |
| `--min-speed F` | Slow meteors are missed (default 0.04 frame widths per second) |
| `--no-trails` | You prefer a clean picture without the paths of earlier meteors |
| `--no-video` | You only need the tables and pictures |

`meteor-cut --help` and `meteor-cut process --help` list everything.

## How it works

In short: Sky Tracker follows everything that moves, twice (every frame for meteors, about five frames a second for
satellites and planes). Meteor Cut then keeps what moves like a meteor: straight, fast, over in a few seconds, and
never twice in the same place. Lights that flare in one spot, people, and groups of things moving together are left
out, and scene cuts in compilations are handled. [docs/how-it-works.md](docs/how-it-works.md) explains each step and
its limits.

## Contributing

Bug reports with a sample video are the most useful thing you can send. See [CONTRIBUTING.md](CONTRIBUTING.md) to set
up a development copy; the tests run without a Sky Tracker licence.

## Licence

Meteor Cut is MIT-licensed: see [LICENSE](LICENSE). It uses the Sky Tracker SDK, which has its
[own licence](https://sky-tracker.deneb-systems.com/terms/). The upload page and report use the Manrope typeface,
included under the SIL Open Font License ([src/meteor_cut/web/static/fonts/Manrope-OFL.txt](src/meteor_cut/web/static/fonts/Manrope-OFL.txt)).
