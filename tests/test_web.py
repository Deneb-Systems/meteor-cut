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


@pytest.mark.parametrize(
    ("data", "error"),
    [
        ({}, "Choose a video file"),
        ({"name": "notes.txt"}, "not a video format"),
        ({"tilt": "30"}, "needs the lens"),
        ({"focal": "fourteen"}, "must be a number"),
        ({"show": "ufos"}, "show takes"),
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
