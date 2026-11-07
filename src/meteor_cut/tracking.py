"""The one place Meteor Cut talks to Sky Tracker.

Everything else depends on the small `MotionTracker` protocol below, so tests (and anyone experimenting) can pass a
stand-in tracker instead of the SDK.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol

from .errors import SkyTrackerError

if TYPE_CHECKING:
    from collections.abc import Sequence

    import numpy as np

TRIAL_URL = "https://sky-tracker.deneb-systems.com/trial/"
MIN_SDK = "1.2"


class MotionTrack(Protocol):
    """What Meteor Cut reads from each track Sky Tracker reports."""

    id: int
    cx: float
    cy: float
    bbox_x: float
    bbox_y: float
    bbox_w: float
    bbox_h: float
    misses: int


class MotionTracker(Protocol):
    def update(self, frame: np.ndarray, dt: float = ...) -> Sequence[MotionTrack]: ...

    def reset(self) -> None: ...


class TrackerFactory(Protocol):
    def __call__(self, **options: Any) -> MotionTracker: ...


def sky_tracker_factory(**options: Any) -> MotionTracker:
    """Creates Sky Tracker's MotionTracker, turning install and licence problems into a clear message."""
    try:
        import sky_tracker
    except ImportError:
        raise SkyTrackerError(
            f"Meteor Cut needs the Sky Tracker SDK: pip install 'sky-tracker-sdk>={MIN_SDK}'"
        ) from None
    if not hasattr(sky_tracker, "MotionTracker"):
        raise SkyTrackerError(
            f"This Sky Tracker SDK is too old for Meteor Cut (it needs {MIN_SDK} or later): "
            "pip install --upgrade sky-tracker-sdk"
        )
    try:
        tracker: MotionTracker = sky_tracker.MotionTracker(**options)
    except ValueError:
        raise
    except RuntimeError as error:
        if "licen" in str(error).lower():
            # Since 1.2.0 the bright-diff mode Meteor Cut uses is free; other modes need a key.
            raise SkyTrackerError(
                f"Sky Tracker needs a licence key for this mode ({error}). Meteor Cut's own bright-diff mode needs "
                f"none with sky-tracker-sdk {MIN_SDK} or later; for the others, a free 7-day trial key is at "
                f"{TRIAL_URL}."
            ) from None
        raise
    return tracker


def sdk_version() -> str | None:
    """The installed Sky Tracker SDK's version, or None when it is not installed."""
    from importlib.metadata import PackageNotFoundError, version

    try:
        return version("sky-tracker-sdk")
    except PackageNotFoundError:
        return None
