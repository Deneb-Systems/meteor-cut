"""Where each pixel points in the sky, from a plate solution (a FITS World Coordinate System header).

astrometry.net (solve-field, or nova.astrometry.net in a browser) writes its answer as wcs.fits: a gnomonic (TAN)
projection with SIP polynomials for the lens distortion. This module reads that file and maps pixels to right
ascension and declination and back, without the weight of a full astronomy library. Pixel coordinates are 0-based
(the first pixel's centre is 0, 0), as in OpenCV; FITS counts from 1, which is handled here.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .errors import SettingsError

_CARD = 80
_BLOCK = 2880


def read_header(path: str | Path) -> dict[str, str | float | bool]:
    """The primary header's keywords of a FITS file (a wcs.fits from astrometry.net, or any FITS image)."""
    data = Path(path).read_bytes()
    cards: dict[str, str | float | bool] = {}
    for start in range(0, len(data), _CARD):
        card = data[start : start + _CARD].decode("ascii", errors="replace")
        key = card[:8].strip()
        if key == "END":
            return cards
        if card[8:10] != "= " or not key:
            continue
        value = card[10:].strip()
        if value.startswith("'"):
            cards[key] = value[1:].split("'", 1)[0].rstrip()
            continue
        value = value.split("/", 1)[0].strip()
        if value in {"T", "F"}:
            cards[key] = value == "T"
            continue
        try:
            cards[key] = float(value.replace("D", "E"))
        except ValueError:
            cards[key] = value
    raise SettingsError(f"{path} is not a FITS header: no END card")


def _sip(cards: dict[str, str | float | bool], name: str) -> dict[tuple[int, int], float]:
    order = int(float(cards.get(f"{name}_ORDER", 0) or 0))
    terms = {}
    for p in range(order + 1):
        for q in range(order + 1 - p):
            value = cards.get(f"{name}_{p}_{q}")
            if isinstance(value, (int, float)) and not isinstance(value, bool) and value:
                terms[(p, q)] = float(value)
    return terms


def _poly(terms: dict[tuple[int, int], float], u: np.ndarray, v: np.ndarray) -> np.ndarray:
    total = np.zeros_like(u)
    for (p, q), coefficient in terms.items():
        total += coefficient * u**p * v**q
    return total


@dataclass(frozen=True)
class Wcs:
    crpix: tuple[float, float]  # FITS (1-based) reference pixel
    crval: tuple[float, float]  # RA, Dec at the reference pixel, degrees
    cd: np.ndarray  # 2x2, degrees per pixel
    a: dict[tuple[int, int], float] = field(default_factory=dict)  # SIP forward distortion
    b: dict[tuple[int, int], float] = field(default_factory=dict)
    ap: dict[tuple[int, int], float] = field(default_factory=dict)  # SIP inverse distortion
    bp: dict[tuple[int, int], float] = field(default_factory=dict)
    width: int | None = None  # image size the solution was made for, when the header says
    height: int | None = None

    @classmethod
    def from_header(cls, cards: dict[str, str | float | bool]) -> Wcs:
        ctype = (str(cards.get("CTYPE1", "")), str(cards.get("CTYPE2", "")))
        if not (ctype[0].startswith("RA---TAN") and ctype[1].startswith("DEC--TAN")):
            raise SettingsError(f"only TAN (gnomonic) plate solutions are supported, not {ctype[0]} / {ctype[1]}")

        def number(key: str, default: float | None = None) -> float:
            value = cards.get(key, default)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise SettingsError(f"the plate solution has no {key}")
            return float(value)

        if "CD1_1" in cards:
            cd = np.array([[number("CD1_1"), number("CD1_2", 0.0)], [number("CD2_1", 0.0), number("CD2_2")]])
        else:
            pc = np.array([[number("PC1_1", 1.0), number("PC1_2", 0.0)], [number("PC2_1", 0.0), number("PC2_2", 1.0)]])
            cd = np.diag([number("CDELT1"), number("CDELT2")]) @ pc
        size = cards.get("IMAGEW"), cards.get("IMAGEH")
        return cls(
            crpix=(number("CRPIX1"), number("CRPIX2")),
            crval=(number("CRVAL1"), number("CRVAL2")),
            cd=cd,
            a=_sip(cards, "A"),
            b=_sip(cards, "B"),
            ap=_sip(cards, "AP"),
            bp=_sip(cards, "BP"),
            width=int(size[0]) if isinstance(size[0], (int, float)) else None,
            height=int(size[1]) if isinstance(size[1], (int, float)) else None,
        )

    @classmethod
    def read(cls, path: str | Path) -> Wcs:
        return cls.from_header(read_header(path))

    @property
    def scale_deg_per_px(self) -> float:
        return math.sqrt(abs(float(np.linalg.det(self.cd))))

    def pixel_to_world(self, x: np.ndarray | float, y: np.ndarray | float) -> tuple[np.ndarray, np.ndarray]:
        """Right ascension and declination (degrees) of 0-based pixels."""
        u = np.asarray(x, float) + 1.0 - self.crpix[0]
        v = np.asarray(y, float) + 1.0 - self.crpix[1]
        u, v = u + _poly(self.a, u, v), v + _poly(self.b, u, v)
        xi = self.cd[0, 0] * u + self.cd[0, 1] * v
        eta = self.cd[1, 0] * u + self.cd[1, 1] * v
        # Gnomonic deprojection: native spherical coordinates, then rotate the reference point to CRVAL.
        phi = np.arctan2(np.radians(xi), -np.radians(eta))
        theta = np.arctan2(1.0, np.hypot(np.radians(xi), np.radians(eta)))
        ra0, dec0 = np.radians(self.crval[0]), np.radians(self.crval[1])
        dphi = phi - np.pi  # LONPOLE 180
        dec = np.arcsin(np.sin(theta) * np.sin(dec0) + np.cos(theta) * np.cos(dec0) * np.cos(dphi))
        ra = ra0 + np.arctan2(
            -np.cos(theta) * np.sin(dphi), np.sin(theta) * np.cos(dec0) - np.cos(theta) * np.sin(dec0) * np.cos(dphi)
        )
        return np.degrees(ra) % 360.0, np.degrees(dec)

    def world_to_pixel(self, ra: np.ndarray | float, dec: np.ndarray | float) -> tuple[np.ndarray, np.ndarray]:
        """0-based pixels of sky positions; NaN for directions more than 90 degrees from the field's centre."""
        ra_r, dec_r = np.radians(np.asarray(ra, float)), np.radians(np.asarray(dec, float))
        ra0, dec0 = np.radians(self.crval[0]), np.radians(self.crval[1])
        dra = ra_r - ra0
        theta = np.arcsin(np.sin(dec_r) * np.sin(dec0) + np.cos(dec_r) * np.cos(dec0) * np.cos(dra))
        phi = np.pi + np.arctan2(
            -np.cos(dec_r) * np.sin(dra), np.sin(dec_r) * np.cos(dec0) - np.cos(dec_r) * np.sin(dec0) * np.cos(dra)
        )
        with np.errstate(divide="ignore", invalid="ignore"):
            r = np.degrees(1.0 / np.tan(theta))
            r = np.where(theta > 0, r, np.nan)
        xi, eta = r * np.sin(phi), -r * np.cos(phi)
        u, v = np.linalg.solve(self.cd, np.vstack([np.ravel(xi), np.ravel(eta)]))
        u, v = u.reshape(np.shape(xi)), v.reshape(np.shape(eta))
        if self.ap or self.bp:
            u, v = u + _poly(self.ap, u, v), v + _poly(self.bp, u, v)
        elif self.a or self.b:
            u, v = self._invert_sip(u, v)
        return u + self.crpix[0] - 1.0, v + self.crpix[1] - 1.0

    def _invert_sip(self, u: np.ndarray, v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Undistorted to distorted offsets by fixed-point iteration, when the header has no inverse polynomials."""
        pu, pv = u.copy(), v.copy()
        for _ in range(20):
            pu, pv = u - _poly(self.a, pu, pv), v - _poly(self.b, pu, pv)
        return pu, pv
