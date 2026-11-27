"""Convert the Serrano et al. 2021 ratings xlsx into the path,gloss CSV that ManifestDataset reads."""
import argparse
import csv
from pathlib import Path

import openpyxl

SHEET = "MedianRating_per_image"
ATTRS = ("glossiness", "refsharp", "contgloss", "metallicness", "lightness", "anisotropy")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--xlsx", default="data/serrano/Serrano_SIGG2021_PerceptualRatings.xlsx")
    p.add_argument("--out", default="data/serrano/manifest.csv")
    p.add_argument("--attr", choices=ATTRS, default="glossiness", help="rating used as the `gloss` column")
    p.add_argument("--image-root", help="if given, drop rows whose image is not under this directory")
    args = p.parse_args()

    wb = openpyxl.load_workbook(args.xlsx, read_only=True, data_only=True)
    rows = wb[SHEET].iter_rows(values_only=True)
    header = next(rows)
    col = {name: i for i, name in enumerate(header)}

    root = Path(args.image_root) if args.image_root else None
    kept, skipped = [], 0
    for r in rows:
        name = r[col["image"]]
        if root is not None and not (root / name).is_file():
            skipped += 1
            continue
        raw = float(r[col[args.attr]])
        # ratings are 1-7; map to 0-1. The raw medians of every attribute are kept as extra columns.
        row = {"path": name, "gloss": (raw - 1) / 6, "gloss_raw": raw}
        row.update({a: float(r[col[a]]) for a in ATTRS})
        kept.append(row)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["path", "gloss", "gloss_raw", *ATTRS])
        w.writeheader()
        w.writerows(kept)
    print(f"wrote {len(kept)} rows to {args.out} (skipped {skipped} without an image)")


if __name__ == "__main__":
    main()
