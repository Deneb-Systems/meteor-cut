import sys
import types

import pytest

from meteor_cut.errors import SkyTrackerError
from meteor_cut.tracking import TRIAL_URL, sky_tracker_factory


def install_fake_sdk(monkeypatch, motion_tracker=None):
    module = types.ModuleType("sky_tracker")
    if motion_tracker is not None:
        module.MotionTracker = motion_tracker
    monkeypatch.setitem(sys.modules, "sky_tracker", module)


def test_missing_sdk(monkeypatch):
    monkeypatch.setitem(sys.modules, "sky_tracker", None)
    with pytest.raises(SkyTrackerError, match="pip install"):
        sky_tracker_factory()


def test_sdk_without_motion_tracker(monkeypatch):
    install_fake_sdk(monkeypatch)
    with pytest.raises(SkyTrackerError, match="too old"):
        sky_tracker_factory()


def test_missing_licence_points_to_the_trial(monkeypatch):
    def refuse(**options):
        raise RuntimeError("no license key found")

    install_fake_sdk(monkeypatch, refuse)
    with pytest.raises(SkyTrackerError) as error:
        sky_tracker_factory()
    assert TRIAL_URL in str(error.value)


def test_options_reach_the_sdk_and_other_errors_pass_through(monkeypatch):
    seen = {}

    def tracker(**options):
        seen.update(options)
        if options.get("mode") == "broken":
            raise RuntimeError("something else")
        return "tracker"

    install_fake_sdk(monkeypatch, tracker)
    assert sky_tracker_factory(mode="bright-diff", moving_only=True) == "tracker"
    assert seen == {"mode": "bright-diff", "moving_only": True}
    with pytest.raises(RuntimeError, match="something else"):
        sky_tracker_factory(mode="broken")
