# How Meteor Cut works

Meteor Cut reads the whole video once to find everything that moves; then each moving thing is classified, and short
stretches around each meteor are read again to measure it, stack it onto the composite, and draw the videos. This page walks through each step, the numbers behind it, and its limits.

```
video ──► sky frame ──► plate solution (astrometry.net, optional)
video ──► scan (Sky Tracker, two trackers, scene cuts) ──► tracks
tracks ──► classify ──► meteors, satellites, planes, rejected (with reasons)
          └► optics (lens, tilt) or the stars ──► angular speed, km/s, sky paths ──► showers, radiant
meteors' frames ──► brightness and magnitude, composite.png ──► rankings, top.png
all of it ──► render ──► meteors.mp4, shorts/      report ──► CSV + summary.json + report.html
```

| Module | Job |
|---|---|
| `scan.py` | Decode the video once; run the trackers; find scene cuts; keep a dark frame per scene |
| `tracking.py` | The only code that talks to the Sky Tracker SDK (tests swap in a stand-in) |
| `classify.py` | Meteor, satellite, plane or rejected, with a reason code |
| `optics.py` | Pixels to directions in the sky; angular speed; km/s |
| `radiant.py` | The point a shower's meteors trace back to, in the picture or among the stars |
| `astro.py`, `wcs.py`, `sky.py` | Sidereal time and the Sun's longitude; plate solutions; the camera against the stars |
| `showers.py`, `data/` | The IAU's established showers and the bright-star catalogue |
| `frames.py`, `photometry.py` | Reading stretches of the video again; brightness and magnitude |
| `composite.py`, `shorts.py` | The composite picture; the vertical clips |
| `enrich.py` | Sky measurements, showers and rankings, added after classification |
| `live.py` | Watching a camera live |
| `images.py`, `render.py` | The pictures and the highlight video |
| `report.py`, `html_report.py` | CSV and JSON; the report page |
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

- **Plane:** strobe flashes, or flying with another light beside it (within 2 % of the frame width, offset across the
  direction of travel, like wing-tip lights). Lights of one plane are merged. A flash is a single sample at least 1.8
  times, and 50 grey levels above, the median of the five samples on each side, with both neighbours below three
  quarters of it; it takes three. At five samples a second a strobe (a tenth of a second) shows as exactly that: one
  sample, alone. Samples where the tracker briefly lost the light (below a quarter of its usual level) are skipped.
- **Satellite:** everything else: one point, steady, flaring (brightening and fading over seconds as it catches the
  sun), twinkling, or dropping in and out of detection when faint. Satellites of a train (Starlink) fly one behind
  another, not side by side.

On a night of Perseids (8 hours, 1,551 satellites and planes) an earlier rule, any brightness spread above 25 % of
the mean, called 81 lights planes; checked frame by frame, they were satellites, mostly flares and faint ones losing
detection. The flash rule calls one light a plane: navigation lights and strobes over the trees at dusk. A plane whose
strobes face away, or a distant one with dim lights, still passes for a satellite. Short tracks that the slow tracker sees during
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

Every meteor becomes a clip from 0.7 s before it appears to 1 s after it ends, clipped to its scene. Satellites and
planes, when shown, take half a minute or more to cross, so only the middle 3 seconds of each pass are kept
(`Style.pass_clip_s`); their whole paths are in the tables and on the trajectories picture. Clips closer
than 0.6 s are joined. For long recordings the second pass jumps straight to each clip instead of decoding the hours
between; it lands two seconds early and reads forward, so frame numbers stay exact even where seeking is approximate,
and it falls back to reading straight through when a video cannot seek at all.

## 7. The stars

A frame from the middle of the video (the median of fifteen frames, so meteors, planes and noise drop out, with the
sky glow flattened) is plate-solved by astrometry.net: its star patterns are matched against an index of the sky, which
gives a TAN projection with SIP distortion terms, so lens distortion is measured rather than assumed. `wcs.py` reads
that answer without an astronomy library; its results match astropy to a nanodegree.

A tripod turns with the Earth, so every pixel sweeps through right ascension at the sidereal rate, 15.04 degrees an
hour, at a fixed declination. One solution therefore describes the whole night: a meteor's start and end become
directions among the stars at the moments it was there. Angular lengths and speeds are measured along the sky, and
with the site and time the elevation of each point comes from the stars too, so no camera tilt is needed for km/s.

The radiant is then fitted among the stars, where it stays put all night, instead of in the picture, where it drifts
by 15 degrees an hour. Each meteor's path is a great circle; the radiant is the direction closest to all of their
poles (the smallest eigenvector of the sum of their outer products), refitted with tightening tolerances (15, 8 and 4
degrees) as in the picture version.

## 8. Showers

Given the date (the start time), the Sun's ecliptic longitude picks the showers that are active, from the IAU Meteor
Data Center's established showers (171 of them, one row per shower: the median of the published parameter sets). Each
shower's radiant is moved by its daily drift to that night. A meteor belongs to a shower when its great circle passes
within 4 degrees of the radiant, it flies away from the radiant and starts within 90 degrees of it, and, when the
distance to the meteor is known, its angular speed is between 0.35 and 2 times what the shower's entry speed gives at
that distance. With the site known, showers whose radiant was below the horizon are left out. The closest fit wins;
the rest are sporadic. A sporadic meteor that happens to line up is counted as a member, so a shower count of a few
is uncertain.

## 9. Brightness and magnitude

Video stores light through a gamma curve, so pixels are made linear first (power 2.2). A meteor's brightness in a frame
is the light it added inside its tracked box (plus 4 pixels), above the median of the nine frames before it appeared;
its peak is the brightest frame. With a plate solution, catalogued stars of magnitude 1 to 5 in that background are
measured the same way (3-pixel aperture, sky from a 6-9 pixel ring, centred on the brightest pixel near the catalogue
position); their median zero point needs at least eight stars and turns the meteor's peak into a magnitude. Clipped
pixels (250 or more) make the value a lower bound. Video compression, noise reduction and the true camera curve make
this rough: treat it as plus or minus half a magnitude or more. Magnitude -4 or brighter is a fireball.

## 10. Composite and vertical clips

The composite starts from a median of frames at the plate-solved moment (or the middle of the busiest scene). Each
meteor's streak is the most it added to its background in any of its frames, inside a soft corridor along its path,
minus a small noise floor so hundreds of streaks do not add up their noise. With a plate solution each streak is
remapped from where its piece of sky was then to where it was at the base picture's moment, so the shower radiates
from one point among the stars while the landscape stays sharp.

Vertical clips go to the top meteors of each list, in a folder per list: the fastest (by angular speed when the lens or
the stars are known), the longest trails and the brightest. A meteor on several lists gets a clip in each, titled with
its place ("2nd fastest"). Each is 1080x1920: a square close-up holding the
whole path with a margin (never closer than 300 source pixels), the whole frame below with the close-up outlined, and
the meteor's details as titles. It plays from 1.2 s before the meteor to 1.5 s after, then replays the meteor at a
quarter speed.

## 11. Live mode

`meteor-cut watch` runs the same tracker and the same meteor rules on a live source. The last 16 seconds or so are kept
as JPEG in memory. Lamps and people are recognised by what else moves near them within ten seconds, so each candidate
is decided ten and a half seconds after it ends; a confirmed meteor's clip is cut from memory, from 1.5 s before it to
1.5 s after, its brightness measured and its streak added to the composite. When the source ends or is stopped,
anything still moving is left undecided, because it was cut short. Satellites and planes are not looked for live.

## Limits

- **Fisheye lenses:** paths are curves, not lines. Detection still works, but straightness, speeds and the radiant are
  off. Use `--sky` to stay away from the edges, or treat the speeds as rough.
- **Moving cameras:** `bright-diff` assumes a still camera. Tracking mounts and handheld video do not work.
- **Very faint meteors in heavily compressed video:** they may never get bright enough between frames to be detected.
- **Plate solving:** needs a clear part of the sky in the middle of the video and a camera that did not move. A
  compilation is solved for the scene of that frame only.
- **Magnitudes:** rough, see above; meteors outside the solved scene or with too few stars around get none.
- **Twilight:** a brightening or darkening sky is fine, but clouds lit by passing car headlights can cause false
  detections; `rejected.csv` shows what was left out and why.
