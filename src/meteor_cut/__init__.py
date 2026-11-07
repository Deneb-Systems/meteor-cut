"""Meteor Cut: find the meteors (and satellites and planes) in a night-sky video, tracked with Sky Tracker.

The whole run is one call::

    from meteor_cut import Lens, process

    result = process("night.mp4", "night-meteors", lens=Lens(focal_mm=14, tilt_deg=40))
    for meteor in result.found.meteors:
        print(meteor.label, meteor.start_s, meteor.speed_value)
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

from .classify import Classification, Rules
from .errors import MeteorCutError, SettingsError, SkyTrackerError, VideoError
from .model import Kind, Reason, Rejection, SkyObject
from .optics import AngularMotion, Camera
from .pipeline import Lens, Result, analyse, parse_show, process
from .radiant import Radiant
from .render import Style
from .scan import scan

try:
    __version__ = version("meteor-cut")
except PackageNotFoundError:  # running from a source tree that was never installed
    __version__ = "0.0.0+unknown"

__all__ = [
    "AngularMotion",
    "Camera",
    "Classification",
    "Kind",
    "Lens",
    "MeteorCutError",
    "Radiant",
    "Reason",
    "Rejection",
    "Result",
    "Rules",
    "SettingsError",
    "SkyObject",
    "SkyTrackerError",
    "Style",
    "VideoError",
    "__version__",
    "analyse",
    "parse_show",
    "process",
    "scan",
]
