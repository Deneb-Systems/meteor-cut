"""Regenerates src/meteor_cut/data/bright_stars.csv from the Yale Bright Star Catalogue (5th revised edition).

The catalogue (Hoffleit & Warren 1991, CDS V/50) lists every star to about magnitude 6.5. Meteor Cut uses it to
measure meteors' brightness against the stars around them.

    python scripts/update_stars.py                 # fetch ybsc5.gz from the Harvard mirror
    python scripts/update_stars.py --from ybsc5    # an unpacked local copy
"""

from __future__ import annotations

import argparse
import csv
import gzip
import urllib.request
from pathlib import Path

LIMIT = 6.5  # faintest magnitude kept: about what a dark sky shows to the eye
SOURCE = "http://tdc-www.harvard.edu/catalogs/ybsc5.gz"
OUTPUT = Path(__file__).resolve().parents[1] / "src/meteor_cut/data/bright_stars.csv"


def parse(lines: list[str]) -> list[tuple[int, float, float, float]]:
    """(HR number, RA deg, Dec deg, V magnitude), J2000, from the catalogue's fixed-width records."""
    stars = []
    for raw in lines:
        line = raw.ljust(110)
        if not line[75:77].strip() or not line[102:107].strip():
            continue  # novae and other entries with no position or magnitude
        ra = (int(line[75:77]) + int(line[77:79]) / 60 + float(line[79:83]) / 3600) * 15
        dec = int(line[84:86]) + int(line[86:88]) / 60 + int(line[88:90]) / 3600
        if line[83] == "-":
            dec = -dec
        stars.append((int(line[0:4]), round(ra, 5), round(dec, 5), float(line[102:107])))
    return stars


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--from", dest="source", type=Path, help="an unpacked ybsc5 file instead of downloading")
    args = parser.parse_args()
    if args.source:
        text = args.source.read_text(encoding="latin-1")
    else:
        with urllib.request.urlopen(SOURCE, timeout=60) as response:
            text = gzip.decompress(response.read()).decode("latin-1")
    stars = sorted((s for s in parse(text.splitlines()) if s[3] <= LIMIT), key=lambda s: s[3])
    with OUTPUT.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["hr", "ra_deg", "dec_deg", "vmag"])
        writer.writerows(stars)
    print(f"{len(stars)} stars -> {OUTPUT}")  # noqa: T201


if __name__ == "__main__":
    main()
