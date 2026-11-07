"""A small web page on this computer: upload a video, enter the lens if known, get the results.

Videos are processed here and never leave the machine. The server binds to 127.0.0.1 unless told otherwise, runs one
job at a time, and deletes its working folder when it stops.
"""

from __future__ import annotations

import atexit
import ipaddress
import logging
import shutil
import tempfile
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from flask import Flask, abort, jsonify, render_template, request, send_from_directory
from werkzeug.datastructures import MultiDict
from werkzeug.utils import secure_filename

from ..errors import MeteorCutError
from ..optics import SENSORS
from ..pipeline import Lens, parse_show, process
from ..render import Style
from ..tracking import TrackerFactory, sky_tracker_factory

log = logging.getLogger(__name__)

OUTPUTS = (
    "report.html",
    "meteors.mp4",
    "trajectories.png",
    "satellites.png",
    "heatmap.png",
    "meteors.csv",
    "satellites_planes.csv",
    "meteor_points.csv",
    "rejected.csv",
    "summary.json",
)
VIDEO_SUFFIXES = {".mp4", ".mov", ".m4v", ".mkv", ".avi", ".webm", ".mts", ".m2ts"}


@dataclass
class Job:
    id: str
    folder: Path
    name: str
    stage: str = "Waiting"
    fraction: float = 0.0
    done: bool = False
    error: str | None = None
    summary: dict[str, Any] = field(default_factory=dict)

    def status(self) -> dict[str, Any]:
        files = [f for f in OUTPUTS if (self.folder / f).exists()] if self.done else []
        return {
            "stage": self.stage,
            "fraction": self.fraction,
            "done": self.done,
            "error": self.error,
            "summary": self.summary,
            "files": files,
            "name": self.name,
        }


def _number(form: MultiDict[str, str], key: str) -> float | None:
    value = (form.get(key) or "").strip()
    if not value:
        return None
    try:
        return float(value)
    except ValueError:
        raise MeteorCutError(f"{key} must be a number") from None


def create_app(
    work_dir: Path | None = None,
    *,
    tracker_factory: TrackerFactory = sky_tracker_factory,
    max_upload_bytes: int = 20 * 1024**3,
) -> Flask:
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = max_upload_bytes
    root = work_dir or Path(tempfile.mkdtemp(prefix="meteor-cut-"))
    if work_dir is None:
        atexit.register(shutil.rmtree, root, ignore_errors=True)
    jobs: dict[str, Job] = {}
    lock = threading.Lock()
    worker = ThreadPoolExecutor(max_workers=1, thread_name_prefix="meteor-cut")  # one video at a time

    def run(job: Job, video: Path, lens: Lens, style: Style) -> None:
        def progress(stage: str, fraction: float) -> None:
            job.stage, job.fraction = stage, fraction

        try:
            job.summary = process(
                video, job.folder, lens=lens, style=style, tracker_factory=tracker_factory, progress=progress
            ).summary
        except MeteorCutError as error:
            job.error = str(error)
        except Exception:
            log.exception("job %s failed", job.id)
            job.error = "Something went wrong while processing this video; the server log has the details."
        finally:
            video.unlink(missing_ok=True)
            job.done = True

    def get_job(job_id: str) -> Job:
        with lock:
            job = jobs.get(job_id)
        if job is None:
            abort(404)
        return job

    @app.get("/")
    def index() -> str:
        return render_template("index.html", sensors=SENSORS)

    @app.post("/jobs")
    def start() -> Any:
        upload = request.files.get("video")
        if upload is None or not upload.filename:
            return jsonify(error="Choose a video file."), 400
        suffix = Path(upload.filename).suffix.lower()
        if suffix not in VIDEO_SUFFIXES:
            return jsonify(error=f"{suffix or 'That file'} is not a video format Meteor Cut reads."), 400
        try:
            lens = Lens(
                focal_mm=_number(request.form, "focal"),
                sensor=request.form.get("sensor") or None,
                hfov_deg=_number(request.form, "fov"),
                tilt_deg=_number(request.form, "tilt"),
                roll_deg=_number(request.form, "roll") or 0.0,
                altitude_km=_number(request.form, "altitude") or 95.0,
            )
            lens.camera(1920, 1080)  # check the combination before accepting the upload
            style = Style(
                show=parse_show(request.form.getlist("show") or ["meteors"]), trails=request.form.get("trails") == "on"
            )
        except MeteorCutError as error:
            return jsonify(error=str(error)), 400
        job = Job(uuid.uuid4().hex, root / uuid.uuid4().hex, Path(upload.filename).name)
        job.folder.mkdir(parents=True)
        # Keep the uploader's file name (made safe) so the report is titled with it, in a folder of its own so it can
        # never collide with an output such as meteors.mp4.
        (job.folder / "upload").mkdir()
        video = job.folder / "upload" / (secure_filename(upload.filename) or f"video{suffix}")
        upload.save(video)
        with lock:
            jobs[job.id] = job
        worker.submit(run, job, video, lens, style)
        return jsonify(id=job.id), 202

    @app.get("/jobs/<job_id>")
    def status(job_id: str) -> Any:
        return jsonify(get_job(job_id).status())

    @app.get("/jobs/<job_id>/<name>")
    def file(job_id: str, name: str) -> Any:
        job = get_job(job_id)
        if name not in OUTPUTS:
            abort(404)
        return send_from_directory(job.folder, name, as_attachment=request.args.get("download") == "1")

    @app.delete("/jobs/<job_id>")
    def delete(job_id: str) -> Any:
        job = get_job(job_id)
        if not job.done:
            return jsonify(error="This video is still being processed."), 409
        with lock:
            jobs.pop(job_id, None)
        shutil.rmtree(job.folder, ignore_errors=True)
        return "", 204

    @app.errorhandler(413)
    def too_large(_: Exception) -> Any:
        return jsonify(error=f"The video is larger than {max_upload_bytes / 1024**3:.0f} GB."), 413

    return app


def _is_loopback(host: str) -> bool:
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return host == "localhost"


def serve(host: str = "127.0.0.1", port: int = 8765, *, max_upload_bytes: int = 20 * 1024**3) -> None:
    if not _is_loopback(host):
        log.warning("listening on %s: anyone who can reach this address can upload videos and read the results", host)
    print(f"Meteor Cut: open http://{host}:{port} in your browser (Ctrl+C to stop)")
    create_app(max_upload_bytes=max_upload_bytes).run(host=host, port=port, threaded=True)
