"""What is added to the found objects after classification: measurements against the stars, showers, rankings."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime, timedelta

from .astro import alt_az, solar_longitude_deg
from .model import SkyObject
from .radiant import Radiant, SkyRadiant
from .showers import ShowerRules, SkyPath, assign
from .sky import Site, SkyCalibration, sky_motion


def apply_sky(
    objects: Sequence[SkyObject],
    sky: SkyCalibration,
    *,
    start: datetime | None,
    site: Site | None,
    altitude_km: float,
) -> None:
    """Angular motion and sky paths from the plate solution; meteors also get a km/s estimate when the site and
    time are known."""
    for obj in objects:
        meteor = obj.kind.value == "meteor"
        obj.motion = sky_motion(
            sky,
            obj.start,
            obj.end,
            obj.start_s,
            obj.end_s,
            when=start,
            site=site,
            altitude_km=altitude_km if meteor else None,
        )
        if meteor:
            obj.sky_path = SkyPath(sky.vector(*obj.start, obj.start_s), sky.vector(*obj.end, obj.end_s))


def rank_by_speed(objects: Sequence[SkyObject]) -> None:
    for rank, obj in enumerate(sorted(objects, key=lambda o: o.speed_value, reverse=True), start=1):
        obj.rank = rank


def rank_meteors(meteors: Sequence[SkyObject]) -> None:
    """Longest trails and brightest meteors, 1 first. Brightness ranks by magnitude when every measured meteor has
    one, else by the light measured."""
    for rank, m in enumerate(sorted(meteors, key=lambda o: o.length_value, reverse=True), start=1):
        m.length_rank = rank
    measured = [m for m in meteors if m.brightness is not None and m.brightness.peak > 0]
    if measured and all(m.brightness and m.brightness.magnitude is not None for m in measured):
        order = sorted(measured, key=_magnitude)
    else:
        order = sorted(measured, key=lambda o: o.brightness.peak if o.brightness else 0.0, reverse=True)
    for rank, m in enumerate(order, start=1):
        m.brightness_rank = rank


def _magnitude(obj: SkyObject) -> float:
    b = obj.brightness
    return b.magnitude if b is not None and b.magnitude is not None else float("inf")


def top(meteors: Sequence[SkyObject], by: str, count: int = 5) -> list[SkyObject]:
    """The top `count` meteors by "speed", "length" or "brightness" rank."""
    attribute = {"speed": "rank", "length": "length_rank", "brightness": "brightness_rank"}[by]
    ranked = [m for m in meteors if getattr(m, attribute)]
    return sorted(ranked, key=lambda m: getattr(m, attribute))[:count]


def assign_showers(
    meteors: Sequence[SkyObject],
    start: datetime,
    *,
    site: Site | None = None,
    rules: ShowerRules | None = None,
    prefer: str | None = None,
) -> None:
    """Shower membership for meteors with a sky path, on the night of `start`; `prefer` is the shower the night's
    radiant matched."""
    for m in meteors:
        if m.sky_path is None:
            continue
        when = start + timedelta(seconds=m.start_s)
        m.diagnostics["shower_checked"] = True
        membership = assign(
            m.sky_path,
            solar_longitude_deg(when),
            angular_speed_deg_s=m.motion.angular_speed_deg_s if m.motion else None,
            range_km=m.motion.range_km if m.motion else None,
            rules=rules,
            prefer=prefer,
        )
        if membership is not None and site is not None:
            # A radiant below the horizon sends no meteors.
            ra, dec = membership.shower.radiant(solar_longitude_deg(when))
            if alt_az(ra, dec, when, site.latitude_deg, site.longitude_deg)[0] < -5:
                membership = None
        m.shower = membership


def radiant_in_picture(radiant: SkyRadiant, sky: SkyCalibration, scale_deg_px: float) -> Radiant:
    """The sky radiant where it was in the solved frame, for the pictures."""
    x, y = sky.pixel(radiant.ra_deg, radiant.dec_deg, sky.at_s)
    return Radiant(float(x), float(y), radiant.members, radiant.spread_deg / scale_deg_px, radiant.numbers)
