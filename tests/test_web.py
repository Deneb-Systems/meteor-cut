import io
import time

import pytest

from meteor_cut.web import create_app
from meteor_cut.web.app import _is_loopback


@pytest.fixture
def client(tmp_path, fake_tracker):
    return create_app(tmp_path, tracker_factory=fake_tracker, max_upload_bytes=50 * 1024**2).test_client()


def upload(client, data: bytes, name: str = "night.avi", **form):
    return client.post("/jobs", data={"video": (io.BytesIO(data), name), **form}, content_type="multipart/form-data")


def test_the_page_lists_the_sensors(client):
    page = client.get("/")
    assert page.status_code == 200 and b"Meteor Cut" in page.data and b"full-frame" in page.data
    assert b"by <a" in page.data and b"Deneb Systems" in page.data
    with client.get("/static/deneb-systems-logo.svg") as logo:
        assert logo.status_code == 200 and b"<svg" in logo.data


@pytest.mark.parametrize(
    ("data", "error"),
    [
        ({}, "Choose a video file"),
        ({"name": "notes.txt"}, "not a video format"),
        ({"tilt": "30"}, "needs the lens"),
        ({"focal": "fourteen"}, "must be a number"),
        ({"show": "ufos"}, "show takes"),
        ({"start": "yesterday"}, "start takes"),
        ({"latitude": "50"}, "both latitude and longitude"),
    ],
)
def test_bad_requests_are_refused_before_processing(client, fake_tracker, data, error):
    if data == {}:
        response = client.post("/jobs", data={})
    else:
        name = data.pop("name", "night.avi")
        response = upload(client, b"x", name, **data)
    assert response.status_code == 400 and error in response.get_json()["error"]
    assert not fake_tracker.created


def test_an_upload_is_processed_and_its_results_served(client, night_video):
    response = upload(
        client, night_video.path.read_bytes(), fov="90", tilt="40", show=["meteors", "satellites"], trails="on"
    )
    assert response.status_code == 202
    job = response.get_json()["id"]
    for _ in range(600):
        status = client.get(f"/jobs/{job}").get_json()
        if status["done"]:
            break
        time.sleep(0.05)
    assert status["error"] is None, status["error"]
    assert status["summary"]["meteors"] == 1 and "meteors.mp4" in status["files"]
    with client.get(f"/jobs/{job}/report.html") as report:
        assert b"Meteor Cut report" in report.data
        assert b"night.avi" in report.data  # titled with the uploaded file's name
        assert b"90\xc2\xb0 field of view" in report.data
    with client.get(f"/jobs/{job}/meteors.csv") as table:
        assert table.data.startswith(b"number,")
    with client.get(f"/jobs/{job}/heatmap.png?download=1") as image:
        assert "attachment" in image.headers["Content-Disposition"]
    assert client.get(f"/jobs/{job}/upload.avi").status_code == 404  # only the results are served
    assert client.delete(f"/jobs/{job}").status_code == 204
    assert client.get(f"/jobs/{job}").status_code == 404


def test_uploads_over_the_limit_are_refused(tmp_path, fake_tracker):
    small = create_app(tmp_path, tracker_factory=fake_tracker, max_upload_bytes=1024).test_client()
    response = upload(small, b"x" * 4096)
    assert response.status_code == 413 and "larger than" in response.get_json()["error"]


def test_unknown_jobs_are_404(client):
    assert client.get("/jobs/nope").status_code == 404
    assert client.get("/jobs/nope/meteors.csv").status_code == 404


@pytest.mark.parametrize(
    ("host", "local"),
    [("127.0.0.1", True), ("::1", True), ("localhost", True), ("0.0.0.0", False), ("192.168.1.5", False)],
)
def test_loopback_detection(host, local):
    assert _is_loopback(host) is local


def test_an_upload_named_like_an_output_does_not_clash(client, night_video):
    response = upload(client, night_video.path.read_bytes(), "meteors.mp4")
    job = response.get_json()["id"]
    for _ in range(600):
        status = client.get(f"/jobs/{job}").get_json()
        if status["done"]:
            break
        time.sleep(0.05)
    assert status["error"] is None and status["summary"]["meteors"] == 1 and "meteors.mp4" in status["files"]


def test_progress_checks_are_not_logged_but_everything_else_is():
    import logging

    from meteor_cut.web.app import _QuietPolling

    def line(message):
        return logging.LogRecord("werkzeug", logging.INFO, __file__, 1, message, None, None)

    quiet = _QuietPolling()
    assert not quiet.filter(line('127.0.0.1 - - [07/Nov/2026 13:53:06] "GET /jobs/cbbdc65308 HTTP/1.1" 200 -'))
    assert quiet.filter(line('127.0.0.1 - - [07/Nov/2026 13:53:06] "POST /jobs HTTP/1.1" 202 -'))
    assert quiet.filter(line('127.0.0.1 - - [07/Nov/2026 13:53:06] "GET /jobs/cbbdc65308/meteors.mp4 HTTP/1.1" 200 -'))
    assert quiet.filter(line('127.0.0.1 - - [07/Nov/2026 13:53:06] "GET /jobs/nope HTTP/1.1" 404 -'))
