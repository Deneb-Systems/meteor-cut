import pytest

from meteor_cut import __version__, cli


def run(capsys, *argv):
    try:
        code = cli.main(list(argv))
    except SystemExit as exit_:
        code = exit_.code
    out = capsys.readouterr()
    return code, out.out, out.err


def test_version_and_help(capsys):
    code, out, _ = run(capsys, "--version")
    assert code == 0 and __version__ in out
    code, out, _ = run(capsys)
    assert code == cli.EXIT_USAGE and "serve" in out


def test_process_is_the_default_command(capsys, monkeypatch, night_video, fake_tracker, tmp_path):
    seen = {}

    def fake_process(video, out, **options):
        seen.update(video=video, out=out, **options)
        raise cli.MeteorCutError("stop here")

    monkeypatch.setattr(cli, "process", fake_process)
    code, _, err = run(
        capsys,
        "-q",
        str(night_video.path),
        "-o",
        str(tmp_path),
        "--fov",
        "90",
        "--tilt",
        "30",
        "--show",
        "all",
        "--sky",
        "0,0,640,200",
    )
    assert code == cli.EXIT_ERROR and "stop here" in err
    assert seen["out"] == tmp_path and seen["lens"].hfov_deg == 90 and seen["sky_region"] == (0, 0, 640, 200)
    assert len(seen["style"].show) == 3


def test_missing_video_and_bad_options(capsys, tmp_path):
    code, _, err = run(capsys, str(tmp_path / "missing.mp4"))
    assert code == cli.EXIT_USAGE and "no such video" in err
    code, _, err = run(capsys, "process", "night.mp4", "--sky", "nope")
    assert code == cli.EXIT_USAGE and "X,Y,WIDTH,HEIGHT" in err


def test_a_settings_error_is_reported_not_raised(capsys, night_video):
    code, _, err = run(capsys, str(night_video.path), "--tilt", "30")
    assert code == cli.EXIT_ERROR and "needs the lens" in err


def test_a_full_run_prints_a_summary(capsys, monkeypatch, night_video, fake_tracker, tmp_path):
    original = cli.process
    monkeypatch.setattr(cli, "process", lambda *a, **k: original(*a, **k, tracker_factory=fake_tracker))
    code, out, _ = run(capsys, str(night_video.path), "-o", str(tmp_path), "--no-video")
    assert code == 0
    assert "1 meteors" in out and "satellites" in out and "meteors.csv" in out


@pytest.mark.parametrize("argv", [["serve", "--help"], ["process", "--help"]])
def test_subcommand_help(capsys, argv):
    code, out, _ = run(capsys, *argv)
    assert code == 0 and "usage" in out
