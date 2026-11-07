# How Meteor Cut works

Meteor Cut reads the video twice. The first pass finds everything that moves; then each moving thing is classified;
the second pass draws the highlight video. This page walks through each step, the numbers behind it, and its limits.

```
video ──► scan (Sky Tracker, two trackers, scene cuts) ──► tracks
tracks ──► classify ──► meteors, satellites, planes, rejected (with reasons)
          └► optics (lens, tilt) ──► angular speed, km/s
meteors ──► radiant, trajectories.png, heatmap.png
all of it ──► render ──► meteors.mp4      report ──► CSV + summary.json
```

| Module | Job |
|---|---|
| `scan.py` | Decode the video once; run the trackers; find scene cuts; keep a dark frame per scene |
| `tracking.py` | The only code that talks to the Sky Tracker SDK (tests swap in a stand-in) |
| `classify.py` | Meteor, satellite, plane or rejected, with a reason code |
| `optics.py` | Pixels to directions in the sky; angular speed; km/s |
| `radiant.py` | The point a shower's meteors trace back to |
| `images.py`, `render.py` | The pictures and the highlight video |
| `report.py` | CSV and JSON |
| `pipeline.py` | `process()`: the whole run |
| `cli.py`, `web/` | The command line and the upload page |

## 1. Finding what moves

Sky Tracker's `MotionTracker` runs in `bright-diff` mode: it looks for pixels that got brighter than in the previous
frame. A meteor's head is always newly bright, while its trail is fading, so the box sits on the head. Each detection
is linked to the tracks so far by a Kalman filter and an optimal assignment, which gives every moving thing a stable
ID. `moving_only` drops anything that has not moved 3 pixels from where it appeared, which removes twinkling stars and
sensor noise.

Satellites and planes move a pixel or less per frame, too little to stand out from one frame to the next. A second
tracker therefore sees only every few frames (about five a second), so they have moved several pixels between its
frames.

Videos wider than 1920 pixels are scaled down for detection and the positions scaled back.

**Scene cuts.** Compilations cut between cameras. A cut changes the whole picture at once: the mean change of a
160x90 thumbnail jumps from about 0.2 (a still night sky) to 3-9 (out of 255), so anything above 2.5 counts as a cut.
Both trackers restart there, and every track belongs to one scene.

## 2. Meteor or not

A track is a meteor when it is:

| Test | Default | Rejection code |
|---|---|---|
| Detected in enough frames | 4 | `too_brief` |
| Long enough | 1.5 % of the frame width | `moved_too_little` |
| Short enough in time | 3 s | `too_long` |
| Fast enough | 4 % of the frame width per second | `too_slow` |
| Straight (distance / path length, and distance from a fitted line) | 0.9, 4 % | `not_straight` |
| Not in a spot where other things keep moving, within 10 s and 8 % of the frame width | fewer than 2 | `repeats_in_place` |
| Not moving at the same time as other things nearby | fewer than 2 | `moves_in_group` |

The last two rules are what remove people and lights: a phone screen or a head torch flares in the same place again
and again, and a person shows up as several moving parts at once. Meteors in a shower can arrive close together, but
almost never in the same place within seconds.

All distances are fractions of the frame width, so the same rules suit 720p and 4K. They live in `Rules` and can be
changed from Python; the command line exposes the two most useful ones.

## 3. Satellites and planes

The slow tracker's tracks that last at least 2 seconds, move in a straight line, and move at 0.2-12 % of the frame
width per second are satellites or planes. Each one's brightness over time decides which:

- **Plane:** blinking (brightness spread above 25 % of its mean), gaps (detected in fewer than 70 % of the slow
  tracker's frames), or travelling together with another light. Lights of one plane travelling together are merged.
- **Satellite:** one steady point.

This is a rule of thumb. A satellite that flares (an Iridium-style glint, a tumbling rocket body) can be called a
plane, and a distant plane with dim lights can be called a satellite. Short tracks that the slow tracker sees during
a meteor are dropped as `part_of_meteor`.

## 4. Speed

With the lens known, each pixel maps to a direction in the camera's view (a pinhole model: `focal_px = focal_mm /
sensor_width_mm × image width`). The angle between the first and last detection, divided by the time between them, is
the angular speed in degrees per second. It needs nothing about where the camera points.

With the tilt also known, each direction has an elevation above the horizon. Assuming the meteor burned at 95 km, the
distance to it follows from the elevation on a curved Earth, and angular speed times distance gives the speed across
the line of sight in km/s. The part of the motion along the line of sight is invisible to one camera, so the true speed
is at least this. Satellites and planes get angular speeds only, because their height is unknown.

## 5. The radiant

With a rectilinear lens a meteor's path is a straight line in the image, and the meteors of a shower all fly away from
one point, the radiant. Meteor Cut finds the point closest to all the paths by least squares, then refits twice with
tighter tolerances (25 %, 12 % and 6 % of the frame width), keeping only meteors whose path passes near the point and
that fly away from it. Sporadic meteors from other directions drop out. At least five meteors must agree. The radiant
is often outside the frame.

## 6. The highlight video

Every kept object becomes a clip from 0.7 s before it appears to 1 s after it ends, clipped to its scene. Clips closer
than 0.6 s are joined. For long recordings the second pass jumps straight to each clip instead of decoding the hours
between; it lands two seconds early and reads forward, so frame numbers stay exact even where seeking is approximate,
and it falls back to reading straight through when a video cannot seek at all.

## Limits

- **Fisheye lenses:** paths are curves, not lines. Detection still works, but straightness, speeds and the radiant are
  off. Use `--sky` to stay away from the edges, or treat the speeds as rough.
- **Moving cameras:** `bright-diff` assumes a still camera. Tracking mounts and handheld video do not work.
- **Very faint meteors in heavily compressed video:** they may never get bright enough between frames to be detected.
- **Twilight:** a brightening or darkening sky is fine, but clouds lit by passing car headlights can cause false
  detections; `rejected.csv` shows what was left out and why.
