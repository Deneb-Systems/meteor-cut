"""Errors Meteor Cut raises on purpose. Anything else is a bug."""

from __future__ import annotations


class MeteorCutError(Exception):
    """Base class: a problem the user can fix, with a message meant for them."""


class VideoError(MeteorCutError):
    """The video cannot be opened or read."""


class SettingsError(MeteorCutError, ValueError):
    """Settings that cannot work, such as a camera tilt without a lens."""


class SkyTrackerError(MeteorCutError):
    """The Sky Tracker SDK is missing, too old, or has no valid licence key."""
