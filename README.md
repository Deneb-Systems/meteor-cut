# Meteor Cut

Give it a night-sky video from a camera on a tripod. Meteor Cut finds every meteor, cuts a highlight video with only
the moments that have one, and measures how fast each one crossed the sky. Every meteor gets a box, its trail and its
speed; a panel on the right lists them all, fastest first. It also picks out satellites and planes, draws every
trajectory of the night on one picture with the shower's radiant, and makes a heat map of where meteors appeared.
The numbers go into CSV files.

Meteors are found and followed with [Sky Tracker](https://sky-tracker.deneb-systems.com)'s `MotionTracker`, which
spots what got brighter since the previous frame: the head of a meteor rather than the trail it leaves.

## Install

```bash
pip install meteor-cut
```

Sky Tracker needs a licence key. A free 7-day trial key is one form away:
[sky-tracker.deneb-systems.com/trial](https://sky-tracker.deneb-systems.com/trial/). Then:

```bash
export SKY_TRACKER_LICENSE_KEY=SKT2....
```

## Use it

From the command line:

```bash
meteor-cut night.mp4
meteor-cut night.mp4 --focal 14 --sensor full-frame --tilt 40
```

Or open the page, upload the video there and download the results:

```bash
meteor-cut serve          # then open http://127.0.0.1:8765
```

The page runs on your computer; videos never leave it.

Results go to `night-meteors/` beside the video (`-o` to choose):

| File | What is in it |
|---|---|
| `meteors.mp4` | The highlight video, 1920x1080, ending on the two images below |
| `trajectories.png` | Every trajectory on one picture: meteors coloured by speed, satellites and planes, and the radiant |
| `heatmap.png` | Where in the frame the meteors appeared |
| `satellites_planes.csv` | One row per satellite or plane |
| `meteors.csv` | One row per meteor: time, position, length, direction, speed, rank by speed |
| `meteor_points.csv` | Every tracked position of every meteor, satellite and plane, frame by frame |
| `rejected.csv` | Moving things that were not meteors, and why |
| `summary.json` | Totals, rates, the camera and the settings used |

## Satellites, planes and the radiant

Satellites and planes move too slowly to stand out from one frame to the next, so a second tracker looks at about
five frames a second. Slow, straight and long tracks are satellites when they are one steady point, and planes when
they blink, drop out, or travel as several lights together. That split is a rule of thumb: a flaring satellite can pass
for a plane. They are always in the CSV and on the trajectories picture; to put them in the video too, use
`--show all` (or `--show meteors,satellites`).

Meteors of a shower all fly away from one point in the sky, the radiant. With a normal (rectilinear) lens their paths
are straight lines in the picture, so Meteor Cut finds the point they trace back to, leaving out sporadic meteors from
other directions. It is often outside the frame.

Past trails stay faintly on screen while the video plays; `--no-trails` turns that off.

## Speeds

- **No lens given:** speed on screen, in pixels per second.
- **Lens given** (`--focal` with `--sensor`, or `--fov`): angular speed, in degrees per second. This is what meteor
  observers compare, and it needs nothing else.
- **Lens and tilt given** (`--tilt`, degrees above the horizon the camera points): also an estimate in km/s. One camera
  cannot see how fast a meteor moves towards or away from it, so this is the speed across your view at the height
  meteors burn (95 km, `--altitude`). The real speed is at least that. Values far above 72 km/s, the fastest a meteor
  can enter, mean the lens or tilt is off.

The lens is assumed to be rectilinear (a normal wide-angle lens, not a fisheye), and the video to show the sensor's
full width.

## What counts as a meteor

A meteor moves in a straight line, fast, for under 3 seconds, and never lights up the same place twice. So Meteor Cut
leaves out things that wander or move slowly (people, planes, satellites), lights that flare in the same spot again and
again (torches, phones), and groups of things moving together (a person walking). Cuts between scenes, common in
compilations, are found and handled.

If the ground is busy, keep the search to the sky: `--sky 0,0,1920,700` (x, y, width, height in pixels). To catch
slower meteors, lower `--min-speed` (frame widths per second, default 0.04).

## Develop

```bash
pip install -e '.[test]'
pytest
```
