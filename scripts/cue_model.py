"""Predict the Serrano attribute ratings from the highlight cues and compare with the CNN results.

1. Spearman correlation of each cue with each attribute, and the same after removing the part explained by glossiness.
2. Prediction of each attribute from cue sets (ridge / gradient boosting), validated with material-group folds and on the
   same train/validation split as the CNN runs (train.py --seed 0).
"""
import argparse
import csv
from pathlib import Path

import numpy as np
from scipy.stats import rankdata, spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import QuantileTransformer, StandardScaler

from src.data.datasets import ManifestDataset, material_groups, material_of
from src.train import split_indices

ATTRS = ["glossiness", "contgloss", "refsharp", "metallicness", "lightness"]
CUES = ["coverage", "contrast", "contrast_log", "contrast_edge", "sharp_grad", "sharp_scale", "body_lum",
        "d_sat", "d_chrom", "P_mean", "P_max", "lum_p99"]
TRIO = ["coverage", "contrast", "sharp_grad", "sharp_scale", "has_hl"]  # the three cues of Marlow et al. + a highlight flag
SETS = {
    "lum_p99 only": ["lum_p99"],
    "P_max only": ["P_max"],
    "trio": TRIO,
    "trio + body lightness": TRIO + ["body_lum"],
    "trio + colour, edge": TRIO + ["body_lum", "d_sat", "d_chrom", "contrast_edge", "contrast_log"],
}


def load(manifest, cues_csv):
    with open(manifest, newline="", encoding="utf-8") as f:
        man = list(csv.DictReader(f))
    with open(cues_csv, newline="") as f:
        cues = {r["path"]: r for r in csv.DictReader(f)}
    paths = [r["path"] for r in man]
    y = {a: (np.array([float(r[a]) for r in man]) - 1) / 6 for a in ATTRS}
    names = list(next(iter(cues.values())))[1:]
    X = np.array([[float(cues[p][n]) for n in names] for p in paths])
    return paths, y, {n: X[:, i] for i, n in enumerate(names)}


def fill(feats):
    """Cues that are undefined without a highlight: 0 (sharp_scale: its maximum), has_hl keeps the information."""
    out = {}
    for k, v in feats.items():
        v = v.copy()
        v[np.isnan(v)] = np.nanmax(v) if k == "sharp_scale" else 0.0
        out[k] = v
    return out


def bins_of(y):
    return np.clip(np.round(y * 6), 0, 6).astype(int)


def balanced_weights(y):
    b = bins_of(y)
    return 1.0 / np.bincount(b, minlength=7)[b]


def bal_mse(pred, y):
    b = bins_of(y)
    return float(np.mean([np.mean((pred[b == k] - y[b == k]) ** 2) for k in range(7) if (b == k).any()]))


def scores(pred, y):
    return {"spearman": spearmanr(pred, y)[0], "pearson": float(np.corrcoef(pred, y)[0, 1]),
            "mse": float(np.mean((pred - y) ** 2)), "bal_mse": bal_mse(pred, y)}


def models():
    return {
        "ridge": make_pipeline(QuantileTransformer(n_quantiles=200, output_distribution="normal"),
                               StandardScaler(), Ridge(alpha=10.0)),
        "boost": HistGradientBoostingRegressor(max_depth=3, learning_rate=0.05, max_iter=300, random_state=0),
    }


def fit_predict(kind, Xtr, ytr, Xte):
    m = models()[kind]
    w = balanced_weights(ytr)
    if kind == "ridge":
        m.fit(Xtr, ytr, ridge__sample_weight=w)
    else:
        m.fit(Xtr, ytr, sample_weight=w)
    return m.predict(Xte)


def partial_spearman(a, b, control):
    ra, rb, rc = rankdata(a), rankdata(b), rankdata(control)
    res = lambda v: v - np.polyval(np.polyfit(rc, v, 1), rc)
    return np.corrcoef(res(ra), res(rb))[0, 1]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", default="data/serrano/manifest_sphere.csv")
    p.add_argument("--cues", default="runs/highlight_cues/cues_q97.csv")
    p.add_argument("--out", default="runs/highlight_cues")
    p.add_argument("--targets", nargs="+", default=["glossiness", "contgloss", "refsharp"])
    args = p.parse_args()
    out = Path(args.out)

    paths, y, feats = load(args.manifest, args.cues)
    F = fill(feats)
    n = len(paths)
    print(f"{n} images, highlight region present in {int(feats['has_hl'].sum())}")

    print("\n== 1. Spearman correlation of each cue with each attribute (cues without a highlight filled as above) ==")
    print(f"{'cue':14s}" + "".join(f"{a:>14s}" for a in ATTRS))
    for c in CUES:
        print(f"{c:14s}" + "".join(f"{spearmanr(F[c], y[a])[0]:14.3f}" for a in ATTRS))
    print("\n== same, after removing the part of both explained by glossiness (partial Spearman) ==")
    print(f"{'cue':14s}" + "".join(f"{a:>14s}" for a in ATTRS[1:]))
    for c in CUES:
        print(f"{c:14s}" + "".join(f"{partial_spearman(F[c], y[a], y['glossiness']):14.3f}" for a in ATTRS[1:]))
    print("\nattributes among themselves (Spearman):")
    for a in ATTRS[1:]:
        print(f"  glossiness vs {a:13s} {spearmanr(y['glossiness'], y[a])[0]:.3f}")

    groups_of = material_groups({material_of(pth) for pth in paths})
    groups = np.array([groups_of[material_of(pth)] for pth in paths])
    ds = ManifestDataset(args.manifest, "unused", size=8)
    tr_idx, va_idx = (np.array(i) for i in split_indices(ds, 0.2, 0))
    rows = []
    for target in args.targets:
        print(f"\n== 2. prediction of {target} ==")
        print(f"{'feature set':26s} {'model':6s} | 5-fold material groups: {'spearman':>8s} {'pearson':>8s} {'bal_mse':>8s} "
              f"| CNN split (val 936): {'spearman':>8s} {'pearson':>8s} {'bal_mse':>8s}")
        for name, cols in SETS.items():
            X = np.stack([F[c] for c in cols], 1)
            for kind in ("ridge", "boost"):
                oof = np.zeros(n)
                for tr, te in GroupKFold(5).split(X, groups=groups):
                    oof[te] = fit_predict(kind, X[tr], y[target][tr], X[te])
                cv = scores(oof, y[target])
                pv = fit_predict(kind, X[tr_idx], y[target][tr_idx], X[va_idx])
                sp = scores(pv, y[target][va_idx])
                print(f"{name:26s} {kind:6s} | {cv['spearman']:33.3f} {cv['pearson']:8.3f} {cv['bal_mse']:8.4f} "
                      f"| {sp['spearman']:30.3f} {sp['pearson']:8.3f} {sp['bal_mse']:8.4f}")
                rows.append({"target": target, "set": name, "model": kind, **{f"cv_{k}": v for k, v in cv.items()},
                             **{f"split_{k}": v for k, v in sp.items()}})
    with open(out / "cue_model_results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print("\nconstant predictor (mean of the training split) on the CNN split, glossiness:")
    c = scores(np.full(len(va_idx), y["glossiness"][tr_idx].mean()), y["glossiness"][va_idx])
    print(f"  bal_mse {c['bal_mse']:.4f}   (CNN: bal_mse 0.036-0.041, Pearson r 0.90-0.93; tiny MLP on kernel stats: "
          f"bal_mse 0.045-0.053, r 0.83-0.86)")


if __name__ == "__main__":
    main()
