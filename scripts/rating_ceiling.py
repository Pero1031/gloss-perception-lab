"""Noise ceiling of the median ratings of the Serrano sphere images (split-half reliability of the individual ratings).

Each image has at least five individual ratings. The ratings of an image are split at random into two halves, the medians
of the halves are correlated over all images (the target of the models is the median of all ratings), and the
Spearman-Brown formula gives the reliability of the full set. A model that predicted the true score perfectly would
correlate with the observed median at about sqrt(reliability).
"""
import argparse
from collections import defaultdict

import numpy as np
import openpyxl
from scipy.stats import spearmanr

ATTRS = ["glossiness", "contgloss", "refsharp", "metallicness", "lightness"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--xlsx", default="data/serrano/Serrano_SIGG2021_PerceptualRatings.xlsx")
    p.add_argument("--geometry", default="sphere")
    p.add_argument("--repeats", type=int, default=50)
    args = p.parse_args()

    wb = openpyxl.load_workbook(args.xlsx, read_only=True, data_only=True)
    rows = wb["all_data"].iter_rows(values_only=True)
    col = {n: i for i, n in enumerate(next(rows))}
    ratings = {a: defaultdict(list) for a in ATTRS}
    for r in rows:
        if r[col["geometry"]] == args.geometry:
            for a in ATTRS:
                ratings[a][r[col["image"]]].append(float(r[col[a]]))
    images = sorted(ratings["glossiness"])
    counts = np.array([len(ratings["glossiness"][i]) for i in images])
    print(f"{len(images)} {args.geometry} images; individual ratings per image: min {counts.min()}, "
          f"median {int(np.median(counts))}, max {counts.max()}; images with >= 5: {(counts >= 5).mean():.3f}")

    rng = np.random.default_rng(0)
    sb = lambda r: 2 * r / (1 + r)
    print(f"\n{'attribute':14s}{'split-half r':>13s}{'Spearman':>10s}{'reliability':>13s}{'ceiling r':>11s}"
          f"{'ceiling among median>1 (rho)':>30s}")
    for a in ATTRS:
        lists = [np.array(ratings[a][i]) for i in images]
        full = np.array([np.median(v) for v in lists])
        rs, rhos, rho_gl = [], [], []
        for _ in range(args.repeats):
            h1, h2 = [], []
            for v in lists:
                perm = rng.permutation(len(v))
                k = len(v) // 2
                h1.append(np.median(v[perm[:k]]))
                h2.append(np.median(v[perm[k:]]))
            h1, h2 = np.array(h1), np.array(h2)
            rs.append(np.corrcoef(h1, h2)[0, 1])
            rhos.append(spearmanr(h1, h2)[0])
            g = full > 1  # images the full median calls glossy (selection by the full set slightly flatters this value)
            rho_gl.append(spearmanr(h1[g], h2[g])[0])
        r, rho, rg = np.mean(rs), np.mean(rhos), np.mean(rho_gl)
        print(f"{a:14s}{r:13.3f}{rho:10.3f}{sb(r):13.3f}{np.sqrt(sb(r)):11.3f}{np.sqrt(sb(rg)):30.3f}")


if __name__ == "__main__":
    main()
