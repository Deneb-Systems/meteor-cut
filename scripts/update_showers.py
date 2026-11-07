"""Regenerates src/meteor_cut/data/showers.csv from the IAU Meteor Data Center's list of established showers.

The list gives several sets of parameters for most showers, one per study. Meteor Cut keeps one row per shower: the
median of each value over the sets that give it.

    python scripts/update_showers.py                                       # fetch the current list
    python scripts/update_showers.py --from streamestablisheddata2026.txt  # a local copy

Source: IAU MDC, Jenniskens et al. 2020, Planetary and Space Science 182, 104821; Jopek & Kanuchova 2017,
Planetary and Space Science 143, 3-6.
"""

from __future__ import annotations

import argparse
import csv
import statistics
import urllib.request
from collections import defaultdict
from pathlib import Path

SOURCE = "https://www.ta3.sk/IAUC22DB/MDC2022/Etc/streamestablisheddata2026.txt"
OUTPUT = Path(__file__).resolve().parents[1] / "src/meteor_cut/data/showers.csv"
# Columns of each record (fields separated by |, each in quotes).
CODE, NUMBER, NAME, BEGIN, END, PEAK, RA, DEC, DRA, DDEC, VG, TECHNIQUE = 3, 1, 6, 8, 9, 10, 11, 12, 13, 14, 15, 33


def _number(text: str) -> float | None:
    try:
        return float(text)
    except ValueError:
        return None


def _median(values: list[float | None]) -> float | None:
    known = [v for v in values if v is not None]
    return round(statistics.median(known), 2) if known else None


def parse(lines: list[str]) -> list[dict[str, object]]:
    sets: dict[int, list[list[str]]] = defaultdict(list)
    for line in lines:
        if line.startswith('"'):
            fields = [f.strip().strip('"').strip() for f in line.split("|")]
            if fields[NUMBER].isdigit():  # a few records have no IAU number yet
                sets[int(fields[NUMBER])].append(fields)
    showers = []
    for number, rows in sorted(sets.items()):

        def column(index: int, rows: list[list[str]] = rows) -> list[float | None]:
            return [_number(r[index]) for r in rows]

        techniques = "".join(sorted({c for r in rows for c in r[TECHNIQUE] if c.isalpha()}))
        showers.append(
            {
                "iau": number,
                "code": rows[0][CODE],
                "name": rows[0][NAME],
                "sol_begin": _median(column(BEGIN)),
                "sol_end": _median(column(END)),
                "sol_peak": _median(column(PEAK)),
                "ra_deg": _median(column(RA)),
                "dec_deg": _median(column(DEC)),
                "dra_deg_day": _median(column(DRA)),
                "ddec_deg_day": _median(column(DDEC)),
                "vg_km_s": _median(column(VG)),
                "techniques": techniques,
            }
        )
    return [s for s in showers if s["sol_peak"] is not None and s["ra_deg"] is not None and s["dec_deg"] is not None]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--from", dest="source", type=Path, help="a local copy of the list instead of downloading")
    args = parser.parse_args()
    if args.source:
        text = args.source.read_text(encoding="latin-1")
    else:
        with urllib.request.urlopen(SOURCE, timeout=60) as response:
            text = response.read().decode("latin-1")
    showers = parse(text.splitlines())
    with OUTPUT.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(showers[0]))
        writer.writeheader()
        writer.writerows(showers)
    print(f"{len(showers)} showers -> {OUTPUT}")  # noqa: T201


if __name__ == "__main__":
    main()
