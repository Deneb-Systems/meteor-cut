# Bundled data

| File | What | Source | Refresh with |
|---|---|---|---|
| `bright_stars.csv` | 8,404 stars to magnitude 6.5: HR number, J2000 position, V magnitude | Yale Bright Star Catalogue, 5th revised edition (Hoffleit & Warren 1991; CDS V/50), from the Harvard mirror | `python scripts/update_stars.py` |
| `showers.csv` | 171 established meteor showers: activity window and peak (solar longitude), radiant and its daily drift, geocentric speed; the median of the list's parameter sets for each shower | IAU Meteor Data Center, list of established showers, 2026 edition. Jenniskens et al. 2020, *Planetary and Space Science* 182, 104821; Jopek & Kanuchova 2017, *Planetary and Space Science* 143, 3-6 | `python scripts/update_showers.py` |

Both are scientific reference data published for open use; please cite the sources above in work that relies on them.
