"""report.html: one self-contained page to read and share.

The pictures are embedded (JPEG), the tables sort when a heading is clicked, and the highlight video plays from
meteors.mp4 beside the page. No external files, fonts or scripts.
"""

from __future__ import annotations

import base64
import html
import math
from collections import Counter
from collections.abc import Mapping, Sequence
from importlib.resources import files
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .classify import Classification
from .enrich import top
from .model import Scan, SkyObject
from .showers import by_code
from .text import Timeline, clock, length_text, magnitude_text, speed_text


def _jpeg(image: np.ndarray, width: int = 1600) -> str:
    if image.shape[1] > width:
        image = cv2.resize(image, (width, round(image.shape[0] * width / image.shape[1])), interpolation=cv2.INTER_AREA)
    ok, data = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 85])
    if not ok:  # pragma: no cover - OpenCV always encodes a valid BGR image
        return ""
    return "data:image/jpeg;base64," + base64.b64encode(data.tobytes()).decode("ascii")


def _font_data() -> str:
    """Manrope, the Sky Tracker website's font, embedded so the page looks the same offline (SIL OFL)."""
    data = files("meteor_cut.web").joinpath("static/fonts/manrope-latin-wght-normal.woff2").read_bytes()
    return "data:font/woff2;base64," + base64.b64encode(data).decode("ascii")


def _describe_settings(settings: Mapping[str, Any]) -> str:
    """The settings that shaped the results, in words."""
    lens = settings.get("lens") or {}
    parts = []
    if lens.get("hfov_deg"):
        parts.append(f"{lens['hfov_deg']:g}° field of view")
    elif lens.get("focal_mm"):
        parts.append(f"{lens['focal_mm']:g} mm lens, {lens.get('sensor') or 'full-frame'} sensor")
    else:
        parts.append("lens not given")
    if lens.get("tilt_deg") is not None:
        parts.append(f"tilt {lens['tilt_deg']:g}°, meteors at {lens.get('altitude_km', 95):g} km")
    if settings.get("sky_region"):
        x, y, w, h = settings["sky_region"]
        parts.append(f"sky region {w}x{h} at {x},{y}")
    shown = settings.get("show") or ["meteor"]
    parts.append("video shows " + ", ".join(f"{k}s" for k in shown))
    parts.append("trails " + ("on" if settings.get("trails", True) else "off"))
    return "; ".join(parts)


def _describe_sky(s: Mapping[str, Any]) -> str:
    """The camera's place among the stars and the clock, in words."""
    parts = []
    sky = s.get("sky")
    if sky:
        w, h = sky["field_deg"]
        ra, dec = sky["centre_ra_dec_deg"]
        parts.append(f"Plate-solved: {w:.1f}° x {h:.1f}° of sky centred on RA {ra:.1f}°, Dec {dec:+.1f}°")
        if sky.get("radiant_ra_dec_deg"):
            rra, rdec = sky["radiant_ra_dec_deg"]
            named = by_code(sky["radiant_shower"]) if sky.get("radiant_shower") else None
            parts.append(
                f"radiant at RA {rra:.1f}°, Dec {rdec:+.1f}° from {sky['radiant_meteors']} meteors"
                + (
                    f" ({named.name}, {sky['radiant_shower_distance_deg']:.1f}° from the catalogue radiant)"
                    if named
                    else ""
                )
            )
    if s.get("start_time"):
        source = (
            " (from the file's creation time; pass the start time if it is wrong)"
            if s.get("start_time_from") == "file"
            else ""
        )
        parts.append(f"recording started {s['start_time'][:19].replace('T', ' ')}{source}")
    return "; ".join(parts) + ("." if parts else "")


def _logo_svg() -> str:
    """The Deneb Systems logo, inline so the page stays one file."""
    svg = files("meteor_cut.web").joinpath("static/deneb-systems-logo.svg").read_text(encoding="utf-8")
    return svg.replace("<svg ", '<svg role="img" aria-label="Deneb Systems" ', 1)


def _num(value: float | None, digits: int = 1) -> str:
    return "" if value is None or not math.isfinite(value) else f"{value:.{digits}f}"


def _per_hour_chart(meteors: Sequence[SkyObject], duration_s: float) -> str:
    """Meteors in each hour of the recording, as an inline SVG bar chart."""
    hours = max(1, math.ceil(duration_s / 3600))
    counts = Counter(int(m.start_s // 3600) for m in meteors)
    top = max(counts.values(), default=1)
    bar, gap, height = 56, 14, 140
    width = hours * (bar + gap) + gap
    parts = [f'<svg viewBox="0 0 {width} {height + 44}" role="img" aria-label="Meteors per hour">']
    for hour in range(hours):
        n = counts.get(hour, 0)
        h = round(height * n / top) if top else 0
        x = gap + hour * (bar + gap)
        parts.append(f'<rect x="{x}" y="{height - h + 18}" width="{bar}" height="{h}" rx="4" class="bar"/>')
        parts.append(f'<text x="{x + bar / 2}" y="{height - h + 12}" class="value">{n}</text>')
        parts.append(f'<text x="{x + bar / 2}" y="{height + 36}" class="axis">{hour}h</text>')
    parts.append("</svg>")
    return "".join(parts)


def _table(table_id: str, headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    head = "".join(f"<th>{html.escape(h)}</th>" for h in headers)
    body = "".join("<tr>" + "".join(f"<td>{html.escape(str(cell))}</td>" for cell in row) + "</tr>" for row in rows)
    return f'<table id="{table_id}" class="sortable"><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>'


def _meteor_rows(meteors: Sequence[SkyObject], timeline: Timeline) -> list[list[Any]]:
    rows = []
    for m in meteors:
        mo = m.motion
        b = m.brightness
        rows.append(
            [
                m.label,
                m.rank,
                timeline.label(m.start_s),
                _num(m.duration_s, 2),
                _num(m.speed_px_s, 0),
                _num(mo.angular_speed_deg_s if mo else None),
                _num(mo.sky_speed_km_s if mo else None, 0),
                _num(mo.angular_length_deg if mo else None),
                _num(m.length_px, 0),
                _num(b.magnitude if b else None),
                m.brightness_rank or "",
                m.shower_code,
                _num(m.direction_deg, 0),
            ]
        )
    return rows


def _top_table(meteors: Sequence[SkyObject], by: str, timeline: Timeline) -> str:
    chosen = top(meteors, by)
    if not chosen:
        return '<p class="lead">Not measured.</p>'
    value = (
        length_text
        if by == "length"
        else (lambda m: magnitude_text(m) or f"{m.brightness.peak:.3g}" if m.brightness else "")
    )
    rows = [
        [f"{i}. {m.label}", value(m), timeline.label(m.start_s), speed_text(m)[0], m.shower_code]
        for i, m in enumerate(chosen, 1)
    ]
    head = "Length" if by == "length" else "Brightness"
    return _table(f"top-{by}", ["Meteor", head, "Time", "Speed", "Shower"], rows)


def _other_rows(others: Sequence[SkyObject], timeline: Timeline) -> list[list[Any]]:
    return [
        [
            o.label,
            o.kind.value,
            timeline.label(o.start_s),
            _num(o.duration_s, 0),
            _num(o.speed_px_s, 1),
            _num(o.motion.angular_speed_deg_s if o.motion else None, 2),
            _num(o.direction_deg, 0),
        ]
        for o in others
    ]


def write_html_report(
    path: Path,
    scan: Scan,
    found: Classification,
    summary: Mapping[str, Any],
    images: Mapping[str, np.ndarray],
    video_name: str | None,
    *,
    timeline: Timeline | None = None,
    shorts: Sequence[str] = (),
) -> None:
    timeline = timeline or Timeline(None, scan.duration_s >= 3600)
    s = summary
    name = html.escape(Path(scan.path).name)
    cards = [
        (clock(scan.duration_s), "of video"),
        (str(s["meteors"]), "meteors"),
        (str(s["meteors_per_hour"] or "-"), "meteors per hour"),
    ]
    if s["angular_speed_deg_s"]:
        cards.append((f"{s['angular_speed_deg_s']['max']}°/s", f"fastest ({s['fastest_meteor']})"))
    elif found.meteors:
        fastest = min(found.meteors, key=lambda m: m.rank)
        cards.append((f"{fastest.speed_px_s:.0f} px/s", f"fastest ({fastest.label})"))
    if s["sky_speed_km_s"]:
        cards.append((f"≥{s['sky_speed_km_s']['mean']} km/s", "average across the view"))
    for code, n in list((s.get("showers") or {}).items())[:3]:
        shower = by_code(code)
        cards.append((str(n), shower.name if shower else "sporadic" if code == "SPO" else code))
    if s.get("fireballs"):
        cards.append((str(len(s["fireballs"])), "fireballs (mag -4 or brighter)"))
    cards += [(str(s["satellites"]), "satellites"), (str(s["planes"]), "planes")]
    if s["radiant"]:
        r = s["radiant"]
        cards.append(
            (str(r["meteors"]), "meteors from the radiant" + ("" if r["in_frame"] else " (outside the frame)"))
        )
    card_html = "".join(
        f'<div class="card"><b>{html.escape(v)}</b><span>{html.escape(t)}</span></div>' for v, t in cards
    )

    pictures = {
        "composite.png": "The night on one picture: every meteor"
        + (", aligned on the stars" if s.get("sky") else ", where the camera saw it"),
        "trajectories.png": "Every meteor",
        "satellites.png": "Satellites and planes",
        "heatmap.png": "Heat map",
    }
    figures = "".join(
        f'<figure><img src="{_jpeg(images[file])}" alt="{html.escape(title)}"><figcaption>{html.escape(title)}'
        f"</figcaption></figure>"
        for file, title in pictures.items()
        if file in images
    )
    video = f'<video controls preload="metadata" src="{html.escape(video_name)}"></video>' if video_name else ""
    short_html = "".join(
        f'<video controls preload="metadata" playsinline src="{html.escape(name)}"></video>' for name in shorts
    )
    sky_line = _describe_sky(s)
    rejected = "".join(
        f"<li><b>{n}</b> {html.escape(r.reason.description)} <code>{html.escape(r.reason.value)}</code></li>"
        for r, n in sorted(
            ((next(x for x in found.rejected if x.reason.value == code), n) for code, n in s["rejected"].items()),
            key=lambda pair: -pair[1],
        )
    )
    settings = html.escape(_describe_settings(s["settings"]))

    page = TEMPLATE.format(
        font=_font_data(),
        logo=_logo_svg(),
        name=name,
        cards=card_html,
        chart=_per_hour_chart(found.meteors, scan.duration_s),
        video=video,
        figures=figures,
        top_length=_top_table(found.meteors, "length", timeline),
        top_brightness=_top_table(found.meteors, "brightness", timeline),
        shorts=f'<h2>Vertical clips</h2><p class="lead">The best meteors in 9:16, for Shorts, Reels and TikTok. '
        f'Each plays once at normal speed, then at a quarter speed.</p><div class="shorts">{short_html}</div>'
        if short_html
        else "",
        sky_line=f'<p class="lead">{html.escape(sky_line)}</p>' if sky_line else "",
        meteors=_table(
            "meteors",
            [
                "Meteor",
                "Speed rank",
                "Time",
                "Seconds",
                "px/s",
                "°/s",
                "km/s ≥",
                "Length °",
                "Length px",
                "Mag",
                "Brightness rank",
                "Shower",
                "Direction °",
            ],
            _meteor_rows(found.meteors, timeline),
        ),
        others=_table(
            "others",
            ["Object", "Kind", "Time", "Seconds", "px/s", "°/s", "Direction °"],
            _other_rows(found.others, timeline),
        ),
        others_count=len(found.others),
        rejected=rejected or "<li>Nothing</li>",
        versions=html.escape(
            f"Meteor Cut {s['meteor_cut_version']}"
            + (f", Sky Tracker {s['sky_tracker_version']}" if s["sky_tracker_version"] else "")
        ),
        settings=settings,
    )
    path.write_text(page, encoding="utf-8")


TEMPLATE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Meteor Cut report: {name}</title>
<style>
  @font-face {{ font-family:"Manrope Variable"; font-weight:200 800; font-display:swap;
               src:url("{font}") format("woff2-variations") }}
  :root {{ --ink:#030507; --ink-soft:#070a0d; --graphite:#101519; --graphite-light:#182126; --silver:#d5dcdf;
          --silver-bright:#f2f5f5; --muted:#7b878c; --signal:#83e8fa;
          --line:color-mix(in srgb, var(--silver) 16%, transparent);
          --body:"Manrope Variable",-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif }}
  * {{ box-sizing:border-box }}
  html {{ background:var(--ink); color-scheme:dark }}
  body {{ margin:0; background:var(--ink); color:var(--silver); font:16px/1.6 var(--body);
         -webkit-font-smoothing:antialiased }}
  main {{ width:min(100%,72rem); margin:0 auto; padding:clamp(4rem,9vw,6.5rem) clamp(1.125rem,4vw,3rem) 6rem }}
  a {{ color:var(--signal) }}
  .kicker {{ margin:0 0 .75rem; color:var(--signal); font-size:1.125rem; font-weight:600 }}
  .brand {{ display:flex; align-items:center; gap:.75rem; margin:0 0 3rem }}
  .brand svg {{ width:2.5rem; height:auto }}
  .brand b {{ color:var(--silver-bright); font-size:1.125rem; letter-spacing:-.01em }}
  .brand span {{ color:var(--muted); font-size:.9375rem }}
  .brand a {{ color:var(--silver); font-weight:600; text-decoration:none }}
  h1 {{ margin:0; color:var(--silver-bright); font-size:clamp(2.75rem,7vw,5rem); font-weight:700; line-height:1.02;
       letter-spacing:-.045em; overflow-wrap:anywhere }}
  h2 {{ margin:clamp(3.5rem,8vw,5.5rem) 0 1.25rem; color:var(--silver-bright); font-size:clamp(1.75rem,3.5vw,2.5rem);
       font-weight:700; letter-spacing:-.035em }}
  .lead {{ margin:-.5rem 0 1.5rem; color:#a3abae; font-size:1.0625rem }}
  .cards {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(11rem,1fr)); gap:1rem; margin-top:2.5rem }}
  .card {{ padding:1.5rem; border-radius:1.5rem; background:var(--graphite) }}
  .card b {{ display:block; color:var(--silver-bright); font-size:clamp(2rem,4vw,2.75rem); font-weight:700;
            line-height:1; letter-spacing:-.035em; font-variant-numeric:tabular-nums }}
  .card span {{ display:block; margin-top:.5rem; color:var(--muted); font-size:.9375rem }}
  .panel {{ padding:clamp(1.25rem,3vw,2rem); border-radius:1.75rem; background:var(--graphite) }}
  svg {{ display:block; width:100%; height:auto }} .bar {{ fill:var(--signal) }}
  .value, .axis {{ fill:var(--muted); font:600 12px var(--body); text-anchor:middle }}
  video {{ display:block; width:100%; border-radius:1.5rem; background:#000 }}
  figure {{ margin:0 0 1.5rem }} figure img {{ display:block; width:100%; border-radius:1.5rem }}
  figcaption {{ margin-top:.625rem; color:var(--muted); font-size:.9375rem }}
  .scroll {{ max-height:34rem; overflow:auto; border-radius:1.5rem; background:var(--graphite) }}
  table {{ width:100%; border-collapse:collapse; font-size:.9375rem; font-variant-numeric:tabular-nums }}
  th, td {{ padding:.625rem 1rem; text-align:right; border-bottom:1px solid var(--line); white-space:nowrap }}
  th:first-child, td:first-child {{ text-align:left }}
  th {{ position:sticky; top:0; background:var(--graphite-light); color:var(--silver-bright); font-weight:600;
       cursor:pointer; user-select:none }}
  th:hover {{ color:var(--signal) }} tbody tr:hover {{ background:var(--ink-soft) }}
  td:first-child {{ color:var(--silver-bright); font-weight:600 }}
  ul.reasons {{ margin:0; padding:0; list-style:none }}
  ul.reasons li {{ display:flex; gap:1rem; align-items:baseline; padding:.75rem 0;
                  border-bottom:1px solid var(--line) }}
  ul.reasons b {{ min-width:3.5rem; color:var(--silver-bright); font-size:1.25rem; text-align:right;
                 font-variant-numeric:tabular-nums }}
  code {{ margin-left:auto; color:var(--muted); font-size:.8125rem }}
  footer {{ margin-top:4rem; color:var(--muted); font-size:.875rem }}
  h3 {{ margin:0 0 .75rem; color:var(--signal); font-size:1.125rem; font-weight:600 }}
  .two {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(20rem,1fr)); gap:1.5rem }}
  .shorts {{ display:grid; grid-template-columns:repeat(auto-fill,minmax(12rem,1fr)); gap:1rem }}
  .shorts video {{ aspect-ratio:9/16; border-radius:1rem }}
</style></head><body><main>
<header class="brand"><a href="https://deneb-systems.com" aria-label="Deneb Systems">{logo}</a>
<span><b>Meteor Cut</b> <span>by <a href="https://deneb-systems.com">Deneb Systems</a></span></span></header>
<p class="kicker">Meteor Cut report</p>
<h1>{name}</h1>
<div class="cards">{cards}</div>
<h2>Meteors per hour</h2>
<div class="panel">{chart}</div>
<h2>Highlights</h2>
{sky_line}
{video}
<h2>Pictures</h2>
{figures}
<h2>Top meteors</h2>
<div class="two"><div><h3>Longest trails</h3><div class="scroll">{top_length}</div></div>
<div><h3>Brightest</h3><div class="scroll">{top_brightness}</div></div></div>
{shorts}
<h2>Meteors</h2>
<p class="lead">Click a heading to sort. Speed rank 1 is the fastest.</p>
<div class="scroll">{meteors}</div>
<h2>Satellites and planes ({others_count})</h2>
<div class="scroll">{others}</div>
<h2>Left out</h2>
<p class="lead">Everything that moved but was not kept, by reason. rejected.csv lists each one.</p>
<ul class="reasons">{rejected}</ul>
<footer>Made with Meteor Cut by <a href="https://deneb-systems.com">Deneb Systems</a>. {versions}. {settings}.</footer>
</main>
<script>
  for (const table of document.querySelectorAll('table.sortable')) {{
    table.querySelectorAll('th').forEach((th, column) => th.addEventListener('click', () => {{
      const body = table.tBodies[0], rows = [...body.rows];
      const descending = th.dataset.order !== 'desc';
      th.dataset.order = descending ? 'desc' : 'asc';
      const key = (row) => {{
        const text = row.cells[column].textContent.replace(/^[#SP]/, '');
        const parts = text.split(':').map(Number);
        if (parts.every((n) => !Number.isNaN(n)) && text !== '') return parts.reduce((a, n) => a * 60 + n, 0);
        return text;
      }};
      rows.sort((a, b) => {{
        const x = key(a), y = key(b);
        const order = typeof x === 'number' && typeof y === 'number' ? x - y : String(x).localeCompare(String(y));
        return descending ? -order : order;
      }});
      body.append(...rows);
    }}));
  }}
</script>
</body></html>
"""
