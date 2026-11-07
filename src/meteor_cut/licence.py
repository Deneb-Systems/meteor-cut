"""Sky Tracker: import it and create the MotionTracker, with a clear message when either is missing."""

from __future__ import annotations

TRIAL_URL = "https://sky-tracker.deneb-systems.com/trial/"


class SkyTrackerUnavailable(RuntimeError):
    pass


def motion_tracker(**options):
    try:
        import sky_tracker
    except ImportError:
        raise SkyTrackerUnavailable(
            "Meteor Cut needs the Sky Tracker SDK: pip install 'sky-tracker-sdk>=1.2'") from None
    if not hasattr(sky_tracker, "MotionTracker"):
        raise SkyTrackerUnavailable(
            "This Sky Tracker SDK is too old for Meteor Cut (needs 1.2 or later): pip install -U sky-tracker-sdk")
    try:
        return sky_tracker.MotionTracker(**options)
    except ValueError:
        raise
    except RuntimeError as error:
        message = str(error)
        if "licen" in message.lower():
            raise SkyTrackerUnavailable(
                f"Sky Tracker needs a licence key: {message}\n"
                f"Get a free 7-day trial key at {TRIAL_URL} and set SKY_TRACKER_LICENSE_KEY to it.") from None
        raise
