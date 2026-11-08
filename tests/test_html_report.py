from types import SimpleNamespace

from meteor_cut.html_report import _per_hour_chart


def bars(svg: str) -> list[str]:
    """The hour labels under the bars."""
    return [part.split(">")[1].split("<")[0] for part in svg.split('class="axis"')[1:]]


def at(*seconds: float) -> list[SimpleNamespace]:
    return [SimpleNamespace(start_s=s) for s in seconds]


def test_an_hour_long_video_has_one_bar():
    # 3600.03 s used to give a second, empty hour for its last 0.03 s.
    svg = _per_hour_chart(at(10, 1800, 3599), 3600.03)
    assert bars(svg) == ["0h"]
    assert 'class="value">3</text>' in svg


def test_a_short_last_stretch_counts_in_the_hour_before():
    svg = _per_hour_chart(at(100, 3700), 3900)
    assert bars(svg) == ["0h"]
    assert ">2</text>" in svg


def test_a_long_last_stretch_is_an_hour_of_its_own():
    assert bars(_per_hour_chart(at(100, 7300), 8000)) == ["0h", "1h", "2h"]


def test_a_short_video_has_one_bar_and_a_size_limit():
    svg = _per_hour_chart(at(5), 120)
    assert bars(svg) == ["0h"]
    assert 'style="max-width:252px"' in svg
