import numpy as np
import pytest

from conftest import line, make_scan, make_track
from meteor_cut.classify import classify, find_meteors
from meteor_cut.model import Kind
from meteor_cut.render import Renderer, Style, clips
from meteor_cut.text import ascii_text, clock


def test_clips_pad_merge_and_stop_at_scene_cuts():
    scan = make_scan(
        make_track(1, line(100, 100, 20, 0, 6), start=100),
        make_track(2, line(900, 100, 20, 0, 6), start=120),
        make_track(3, line(100, 600, 20, 0, 6), scene=1, start=310),
        cuts=(300,),
        frames=1000,
    )
    meteors, _ = find_meteors(scan)
    style = Style(before_s=1.0, after_s=1.0, merge_gap_s=0.5)
    # The first two overlap once padded and merge; the third may not start before its scene does.
    assert clips(scan, meteors, style) == [(75, 150), (300, 340)]


@pytest.mark.parametrize(("seconds", "text"), [(0, "00:00.0"), (75.25, "01:15.2"), (4 * 3600 + 62, "4:01:02")])
def test_clock(seconds, text):
    assert clock(seconds) == text


def test_ascii_text_for_opencv():
    assert ascii_text("12.5°/s ≥30 km/s") == "12.5 deg/s >30 km/s"


def make_renderer(show=frozenset({Kind.METEOR}), trails=True):
    scan = make_scan(
        make_track(1, line(400, 200, 20, 12, 10), start=10),
        make_track(2, line(1200, 300, 25, 5, 10), start=80),
        frames=200,
    )
    found = classify(scan)
    return Renderer(scan, found.meteors, found.others, Style(show=show, trails=trails)), found


def test_frames_have_the_output_size_and_show_the_meteor():
    renderer, _ = make_renderer()
    image = np.zeros((1080, 1920, 3), np.uint8)
    quiet = renderer.frame(image, 0)
    during = renderer.frame(image, 15)
    assert quiet.shape == during.shape == (1080, 1920, 3)
    video = (
        slice(renderer.video_y, renderer.video_y + renderer.video_h),
        slice(renderer.video_x, renderer.video_x + renderer.video_w),
    )
    assert during[video].sum() > quiet[video].sum()  # the box, trail and label
    assert [o.label for o in renderer.active(15)] == ["#1"]
    assert renderer.footer.startswith("2 meteors")


def test_trails_stay_after_a_meteor_unless_turned_off():
    image = np.zeros((1080, 1920, 3), np.uint8)
    with_trails, _ = make_renderer()
    without, _ = make_renderer(trails=False)
    for renderer in (with_trails, without):
        for index in range(60):
            renderer.frame(image, index)
    video = (
        slice(with_trails.video_y, with_trails.video_y + with_trails.video_h),
        slice(with_trails.video_x, with_trails.video_x + with_trails.video_w),
    )
    assert with_trails.frame(image, 60)[video].sum() > without.frame(image, 60)[video].sum()


def test_end_cards_are_letterboxed():
    renderer, _ = make_renderer()
    card = renderer.card(np.full((500, 500, 3), 200, np.uint8))
    assert card.shape == (1080, 1920, 3)
    assert card[540, 960].tolist() == [200, 200, 200] and card[540, 10].tolist() != [200, 200, 200]
