"""A small local web page: upload a video, enter the lens if you know it, get the highlight video and the tables.

Runs on this computer only (127.0.0.1 by default); videos are processed here and never leave it.
"""

from __future__ import annotations

import shutil
import tempfile
import threading
import uuid
from dataclasses import dataclass, field
from pathlib import Path

from flask import Flask, abort, jsonify, request, send_from_directory

from .licence import SkyTrackerUnavailable
from .optics import SENSORS
from .pipeline import Lens, run

OUTPUTS = ("meteors.mp4", "meteors.csv", "meteor_points.csv", "rejected.csv", "summary.json")


@dataclass
class Job:
    id: str
    folder: Path
    name: str
    stage: str = "Waiting"
    fraction: float = 0.0
    done: bool = False
    error: str | None = None
    summary: dict = field(default_factory=dict)


def _float(form, key: str) -> float | None:
    value = (form.get(key) or "").strip()
    return float(value) if value else None


def create_app(work_dir: Path | None = None) -> Flask:
    app = Flask(__name__)
    root = work_dir or Path(tempfile.mkdtemp(prefix="meteor-cut-"))
    jobs: dict[str, Job] = {}

    def process(job: Job, video: Path, lens: Lens) -> None:
        def progress(stage: str, fraction: float) -> None:
            job.stage, job.fraction = stage, fraction
        try:
            result = run(str(video), job.folder, lens, progress=progress)
            job.summary = result.summary
        except (SkyTrackerUnavailable, ValueError) as error:
            job.error = str(error)
        except Exception as error:  # shown to the user, who is the only one using this page
            job.error = f"{type(error).__name__}: {error}"
        finally:
            video.unlink(missing_ok=True)
            job.done = True

    @app.get("/")
    def index():
        options = "".join(f'<option value="{k}">{k} ({w:g} x {h:g} mm)</option>' for k, (w, h) in SENSORS.items())
        return PAGE.replace("{{SENSORS}}", options)

    @app.post("/jobs")
    def start():
        upload = request.files.get("video")
        if not upload or not upload.filename:
            return jsonify(error="Choose a video file."), 400
        try:
            lens = Lens(focal_mm=_float(request.form, "focal"), sensor=request.form.get("sensor") or None,
                        hfov_deg=_float(request.form, "fov"), tilt_deg=_float(request.form, "tilt"),
                        roll_deg=_float(request.form, "roll") or 0.0,
                        altitude_km=_float(request.form, "altitude") or 95.0)
            lens.camera(1920, 1080)  # check the combination now, not after the upload is processed
        except ValueError as error:
            return jsonify(error=str(error)), 400
        job = Job(uuid.uuid4().hex[:12], root / uuid.uuid4().hex[:12], Path(upload.filename).name)
        job.folder.mkdir(parents=True)
        video = job.folder / ("upload" + Path(upload.filename).suffix.lower())
        upload.save(video)
        jobs[job.id] = job
        threading.Thread(target=process, args=(job, video, lens), daemon=True).start()
        return jsonify(id=job.id)

    @app.get("/jobs/<job_id>")
    def status(job_id: str):
        job = jobs.get(job_id) or abort(404)
        files = [f for f in OUTPUTS if (job.folder / f).exists()] if job.done else []
        return jsonify(stage=job.stage, fraction=job.fraction, done=job.done, error=job.error,
                       summary=job.summary, files=files, name=job.name)

    @app.get("/jobs/<job_id>/<name>")
    def file(job_id: str, name: str):
        job = jobs.get(job_id) or abort(404)
        if name not in OUTPUTS:
            abort(404)
        return send_from_directory(job.folder, name, as_attachment=request.args.get("download") == "1")

    @app.post("/jobs/<job_id>/delete")
    def delete(job_id: str):
        job = jobs.pop(job_id, None) or abort(404)
        shutil.rmtree(job.folder, ignore_errors=True)
        return jsonify(ok=True)

    return app


def serve(host: str = "127.0.0.1", port: int = 8765) -> None:
    print(f"Meteor Cut: open http://{host}:{port} in your browser (Ctrl+C to stop)")
    create_app().run(host=host, port=port, threaded=True)


PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Meteor Cut</title>
<style>
  :root { --bg:#05080b; --panel:#0d1418; --line:#1f2b31; --text:#e6edf0; --muted:#8c989e; --signal:#83e8fa; }
  * { box-sizing:border-box } body { margin:0; background:var(--bg); color:var(--text);
    font:16px/1.55 system-ui,-apple-system,Segoe UI,sans-serif }
  main { max-width:880px; margin:0 auto; padding:48px 20px 80px }
  h1 { font-size:40px; letter-spacing:-.02em; margin:0 0 6px } p.lead { color:var(--muted); margin:0 0 32px }
  form, .card { background:var(--panel); border:1px solid var(--line); border-radius:14px; padding:24px; margin-bottom:24px }
  label { display:block; font-size:14px; color:var(--muted); margin-bottom:6px }
  input, select { width:100%; padding:10px 12px; border-radius:8px; border:1px solid var(--line); background:#071014;
    color:var(--text); font:inherit }
  .grid { display:grid; grid-template-columns:repeat(auto-fit,minmax(180px,1fr)); gap:16px; margin:18px 0 }
  .hint { color:var(--muted); font-size:13px; margin-top:4px }
  button { background:var(--signal); color:#031015; border:0; border-radius:999px; padding:12px 22px; font:600 16px system-ui;
    cursor:pointer } button:disabled { opacity:.5; cursor:wait }
  .bar { height:8px; background:#14232a; border-radius:99px; overflow:hidden; margin:12px 0 }
  .bar div { height:100%; width:0; background:var(--signal); transition:width .3s }
  video { width:100%; border-radius:10px; background:#000; margin:12px 0 }
  a { color:var(--signal) } .files a { margin-right:16px } .error { color:#ff9d8a }
  .stats { display:grid; grid-template-columns:repeat(auto-fit,minmax(150px,1fr)); gap:12px; margin:12px 0 }
  .stat b { display:block; font-size:26px } .stat span { color:var(--muted); font-size:13px }
</style></head><body><main>
<h1>Meteor Cut</h1>
<p class="lead">Upload a night-sky video from a camera on a tripod. Meteor Cut finds every meteor with Sky Tracker,
makes a highlight video with only them, sorted by speed, and gives you the tables. Everything runs on this computer.</p>
<form id="form">
  <label for="video">Video</label><input id="video" name="video" type="file" accept="video/*" required>
  <div class="grid">
    <div><label for="focal">Focal length (mm)</label><input id="focal" name="focal" type="number" step="0.1" min="1" placeholder="14"></div>
    <div><label for="sensor">Sensor</label><select id="sensor" name="sensor">{{SENSORS}}</select></div>
    <div><label for="fov">or field of view (°)</label><input id="fov" name="fov" type="number" step="0.1" min="1" max="178" placeholder="horizontal"></div>
    <div><label for="tilt">Camera tilt (°)</label><input id="tilt" name="tilt" type="number" step="1" min="-10" max="90" placeholder="above horizon">
      <div class="hint">0 level, 90 straight up. Needed for km/s.</div></div>
    <div><label for="altitude">Meteor height (km)</label><input id="altitude" name="altitude" type="number" step="1" min="40" max="200" value="95"></div>
  </div>
  <p class="hint">Without a lens, speeds are in pixels per second. With the lens, in degrees per second;
    with the tilt too, an estimate in km/s (the part of the speed across your view, so the real speed is at least that).</p>
  <button id="go" type="submit">Find meteors</button>
</form>
<div id="job" class="card" hidden>
  <div id="stage">Uploading…</div><div class="bar"><div id="fill"></div></div>
  <div id="result"></div>
</div>
<script>
const form = document.getElementById('form'), go = document.getElementById('go');
const box = document.getElementById('job'), stage = document.getElementById('stage'), fill = document.getElementById('fill');
const result = document.getElementById('result');
form.addEventListener('submit', async (event) => {
  event.preventDefault();
  go.disabled = true; box.hidden = false; result.innerHTML = ''; stage.textContent = 'Uploading…'; fill.style.width = '0';
  const response = await fetch('/jobs', { method: 'POST', body: new FormData(form) }).catch(() => null);
  const body = response ? await response.json().catch(() => ({})) : {};
  if (!response || !response.ok) { stage.innerHTML = `<span class="error">${body.error || 'Upload failed.'}</span>`; go.disabled = false; return; }
  poll(body.id);
});
async function poll(id) {
  const s = await fetch(`/jobs/${id}`).then(r => r.json());
  stage.textContent = `${s.stage} ${Math.round(s.fraction * 100)}%`; fill.style.width = `${s.fraction * 100}%`;
  if (!s.done) return setTimeout(() => poll(id), 700);
  go.disabled = false;
  if (s.error) { stage.innerHTML = `<span class="error">${s.error.replace(/</g, '&lt;')}</span>`; return; }
  const m = s.summary, a = m.angular_speed_deg_s, k = m.sky_speed_km_s;
  stage.textContent = `${m.meteors} meteors in ${s.name}`;
  result.innerHTML = `
    <div class="stats">
      <div class="stat"><b>${m.meteors}</b><span>meteors</span></div>
      <div class="stat"><b>${m.meteors_per_hour ?? '–'}</b><span>per hour</span></div>
      ${a ? `<div class="stat"><b>${a.max}°/s</b><span>fastest (#${m.fastest})</span></div>
             <div class="stat"><b>${a.mean}°/s</b><span>average</span></div>` : ''}
      ${k ? `<div class="stat"><b>≥${k.mean} km/s</b><span>average across the view</span></div>` : ''}
    </div>
    ${s.files.includes('meteors.mp4') ? `<video controls src="/jobs/${id}/meteors.mp4"></video>` : '<p>No meteors found, so no video.</p>'}
    <p class="files">${s.files.map(f => `<a href="/jobs/${id}/${f}?download=1">${f}</a>`).join('')}</p>`;
}
</script>
</main></body></html>
"""
