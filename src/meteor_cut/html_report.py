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
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .classify import Classification
from .model import Scan, SkyObject
from .text import clock


def _jpeg(image: np.ndarray, width: int = 1600) -> str:
    if image.shape[1] > width:
        image = cv2.resize(image, (width, round(image.shape[0] * width / image.shape[1])), interpolation=cv2.INTER_AREA)
    ok, data = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 85])
    if not ok:  # pragma: no cover - OpenCV always encodes a valid BGR image
        return ""
    return "data:image/jpeg;base64," + base64.b64encode(data.tobytes()).decode("ascii")


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


def _meteor_rows(meteors: Sequence[SkyObject], long: bool) -> list[list[Any]]:
    rows = []
    for m in meteors:
        mo = m.motion
        rows.append(
            [
                m.label,
                m.rank,
                clock(m.start_s, hours_always=long),
                _num(m.duration_s, 2),
                _num(m.speed_px_s, 0),
                _num(mo.angular_speed_deg_s if mo else None),
                _num(mo.sky_speed_km_s if mo else None, 0),
                _num(m.length_px, 0),
                _num(m.direction_deg, 0),
            ]
        )
    return rows


def _other_rows(others: Sequence[SkyObject], long: bool) -> list[list[Any]]:
    return [
        [
            o.label,
            o.kind.value,
            clock(o.start_s, hours_always=long),
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
) -> None:
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
    rejected = "".join(
        f"<li><b>{n}</b> {html.escape(r.reason.description)} <code>{html.escape(r.reason.value)}</code></li>"
        for r, n in sorted(
            ((next(x for x in found.rejected if x.reason.value == code), n) for code, n in s["rejected"].items()),
            key=lambda pair: -pair[1],
        )
    )
    settings = html.escape(str({k: v for k, v in s["settings"].items() if k != "rules"}))

    page = TEMPLATE.format(
        name=name,
        cards=card_html,
        chart=_per_hour_chart(found.meteors, scan.duration_s),
        video=video,
        figures=figures,
        meteors=_table(
            "meteors",
            ["Meteor", "Speed rank", "Time", "Seconds", "px/s", "°/s", "km/s ≥", "Length px", "Direction °"],
            _meteor_rows(found.meteors, scan.duration_s >= 3600),
        ),
        others=_table(
            "others",
            ["Object", "Kind", "Time", "Seconds", "px/s", "°/s", "Direction °"],
            _other_rows(found.others, scan.duration_s >= 3600),
        ),
        others_count=len(found.others),
        rejected=rejected or "<li>Nothing</li>",
        versions=html.escape(f"Meteor Cut {s['meteor_cut_version']}, Sky Tracker {s['sky_tracker_version'] or '?'}"),
        settings=settings,
    )
    path.write_text(page, encoding="utf-8")


TEMPLATE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Meteor Cut report: {name}</title>
<style>
  :root {{ --bg:#05080b; --panel:#0d1418; --line:#1f2b31; --text:#e6edf0; --muted:#8c989e; --signal:#83e8fa; }}
  * {{ box-sizing:border-box }}
  body {{ margin:0; background:var(--bg); color:var(--text);
         font:15px/1.55 system-ui,-apple-system,"Segoe UI",sans-serif }}
  main {{ max-width:1180px; margin:0 auto; padding:40px 20px 80px }}
  h1 {{ font-size:34px; margin:0 0 4px; letter-spacing:-.02em }} h2 {{ margin:44px 0 14px; font-size:22px }}
  .lead {{ color:var(--muted); margin:0 0 26px }}
  .cards {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(160px,1fr)); gap:12px }}
  .card {{ background:var(--panel); border:1px solid var(--line); border-radius:12px; padding:16px }}
  .card b {{ display:block; font-size:26px }} .card span {{ color:var(--muted); font-size:13px }}
  svg {{ width:100%; max-width:760px; height:auto }} .bar {{ fill:var(--signal) }}
  .value, .axis {{ fill:var(--muted); font-size:12px; text-anchor:middle }}
  video {{ width:100%; border-radius:12px; background:#000 }}
  figure {{ margin:0 0 22px }} figure img {{ width:100%; border-radius:12px; display:block }}
  figcaption {{ color:var(--muted); font-size:13px; margin-top:6px }}
  .scroll {{ max-height:520px; overflow:auto; border:1px solid var(--line); border-radius:12px }}
  table {{ border-collapse:collapse; width:100%; font-size:14px; font-variant-numeric:tabular-nums }}
  th, td {{ padding:7px 12px; text-align:right; border-bottom:1px solid var(--line); white-space:nowrap }}
  th:first-child, td:first-child {{ text-align:left }}
  th {{ position:sticky; top:0; background:var(--panel); cursor:pointer; user-select:none }}
  th:hover {{ color:var(--signal) }} tbody tr:hover {{ background:#0b1a20 }}
  ul {{ padding-left:20px }} li {{ margin:4px 0 }} code {{ color:var(--muted); font-size:12px }}
  footer {{ color:var(--muted); font-size:13px; margin-top:44px }}
</style></head><body><main>
<h1>Meteor Cut report</h1>
<p class="lead">{name}</p>
<div class="cards">{cards}</div>
<h2>Meteors per hour of recording</h2>
{chart}
<h2>Highlights</h2>
{video}
<h2>Pictures</h2>
{figures}
<h2>Meteors</h2>
<p class="lead">Click a heading to sort. Speed rank 1 is the fastest.</p>
<div class="scroll">{meteors}</div>
<h2>Satellites and planes ({others_count})</h2>
<div class="scroll">{others}</div>
<h2>Left out</h2>
<p class="lead">Everything that moved but was not kept, by reason. rejected.csv lists each one.</p>
<ul>{rejected}</ul>
<footer>{versions}. Settings: {settings}</footer>
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
