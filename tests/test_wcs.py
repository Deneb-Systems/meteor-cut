import numpy as np
import pytest

from conftest import tan_header, write_fits_header
from meteor_cut.errors import SettingsError
from meteor_cut.wcs import Wcs, read_header


def test_the_reference_pixel_is_the_reference_direction():
    wcs = Wcs.from_header(tan_header(ra=45, dec=50))
    ra, dec = wcs.pixel_to_world(959.5, 539.5)
    assert (float(ra), float(dec)) == pytest.approx((45.0, 50.0))
    assert wcs.scale_deg_per_px == pytest.approx(53 / 1920)


def test_pixels_round_trip_across_the_frame():
    wcs = Wcs.from_header(tan_header())
    xs, ys = np.meshgrid(np.linspace(0, 1919, 7), np.linspace(0, 1079, 5))
    back = wcs.world_to_pixel(*wcs.pixel_to_world(xs, ys))
    assert np.allclose(back[0], xs, atol=1e-6) and np.allclose(back[1], ys, atol=1e-6)


def test_sip_distortion_round_trips_with_and_without_inverse_terms():
    cards = {**tan_header(), "CTYPE1": "RA---TAN-SIP", "CTYPE2": "DEC--TAN-SIP", "A_ORDER": 2.0, "B_ORDER": 2.0}
    cards |= {"A_2_0": 2e-6, "A_0_2": -1e-6, "B_1_1": 1.5e-6}
    wcs = Wcs.from_header(cards)  # no AP/BP: the inverse is found by iteration
    x, y = np.array([10.0, 1800.0, 960.0]), np.array([20.0, 1000.0, 300.0])
    back = wcs.world_to_pixel(*wcs.pixel_to_world(x, y))
    assert np.allclose(back[0], x, atol=1e-3) and np.allclose(back[1], y, atol=1e-3)


def test_matches_astropy_on_a_distorted_solution():
    astropy_wcs = pytest.importorskip("astropy.wcs")
    cards = {**tan_header(ra=300, dec=-20), "CTYPE1": "RA---TAN-SIP", "CTYPE2": "DEC--TAN-SIP"}
    cards |= {"A_ORDER": 2.0, "B_ORDER": 2.0, "A_2_0": 3e-6, "A_1_1": -1e-6, "B_0_2": 2e-6, "CD1_2": 1e-4}
    reference = astropy_wcs.WCS({k: v for k, v in cards.items() if k not in {"IMAGEW", "IMAGEH"}})
    ours = Wcs.from_header(cards)
    x, y = np.array([0.0, 500.5, 1919.0]), np.array([0.0, 800.25, 1079.0])
    expected = reference.all_pix2world(x, y, 0)
    ra, dec = ours.pixel_to_world(x, y)
    assert np.allclose(ra, expected[0], atol=1e-9) and np.allclose(dec, expected[1], atol=1e-9)


def test_directions_behind_the_camera_have_no_pixel():
    wcs = Wcs.from_header(tan_header(ra=45, dec=50))
    x, y = wcs.world_to_pixel(225.0, -50.0)
    assert np.isnan(x) and np.isnan(y)


def test_reads_a_fits_header_file(tmp_path):
    path = write_fits_header(tmp_path / "wcs.fits", tan_header(ra=12.5))
    assert read_header(path)["CTYPE1"] == "RA---TAN"
    wcs = Wcs.read(path)
    assert wcs.crval == (12.5, 50.0) and (wcs.width, wcs.height) == (1920, 1080)


def test_rejects_what_it_cannot_read(tmp_path):
    with pytest.raises(SettingsError, match="TAN"):
        Wcs.from_header({**tan_header(), "CTYPE1": "RA---ZEA", "CTYPE2": "DEC--ZEA"})
    with pytest.raises(SettingsError, match="CRVAL1"):
        Wcs.from_header({k: v for k, v in tan_header().items() if k != "CRVAL1"})
    bad = tmp_path / "bad.fits"
    bad.write_bytes(b"x" * 2880)
    with pytest.raises(SettingsError, match="END"):
        read_header(bad)
