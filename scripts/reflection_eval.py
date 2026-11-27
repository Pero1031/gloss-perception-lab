"""Do the reflection cues fix the under-prediction of mirror-like / highly glossy images?

Cue sets are compared under material-group CV, on the train/validation split of the CNN runs, and under the material x
illumination double hold-out. Then the bias for metals and high ratings is compared with and without the reflection
cues, and each reflection cue is checked for specificity (partial Spearman with the attribute ratings after removing
the part explained by glossiness).
"""
import argparse

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

from scripts.cue_analysis import PQ, YR, err_table
from scripts.cue_model import TRIO, bal_mse, fill, fit_predict, load, partial_spearman
from scripts.reflection_cues import RADIAL, REFL, SIM, SPEC, STRUCT
from src.data.datasets import ManifestDataset, material_groups, material_of
from src.train import split_indices

BASE = TRIO + PQ + YR
SETS = {
    "trio + B (reference)": BASE,
    "+ spectrum of structure": BASE + SPEC,
    "+ similarity to background": BASE + SIM,
    "+ structure outside highlights": BASE + STRUCT,
    "+ radial profile (limb)": BASE + RADIAL,
    "+ all reflection cues": BASE + REFL,
    "trio + all reflection cues (no B)": TRIO + REFL,
}


def metrics(pred, y):
    g = y > 0
    return (np.corrcoef(pred, y)[0, 1], bal_mse(pred, y), spearmanr(pred[g], y[g])[0], roc_auc_score(g, pred))


def fmt(m):
    return f"{m[0]:6.3f}{m[1]:8.4f}{m[2]:7.3f}{m[3]:7.3f}"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", default="data/serrano/manifest_sphere.csv")
    p.add_argument("--cues", default="runs/highlight_cues/cues_q95.csv")
    p.add_argument("--refl", default="runs/highlight_cues/reflection_q95.csv")
    args = p.parse_args()

    paths, y, f1 = load(args.manifest, args.cues)
    _, _, f2 = load(args.manifest, args.refl)
    F = {**fill(f1), **fill(f2)}
    yg, n = y["glossiness"], len(paths)
    materials = [material_of(pth) for pth in paths]
    g_of = material_groups(set(materials))
    groups = np.array([g_of[m] for m in materials])
    illum = np.array([pth.split("]")[0][1:] for pth in paths])
    names = sorted(set(illum))
    fold = np.zeros(n, dtype=int)
    for k, (_, te) in enumerate(GroupKFold(5).split(np.zeros(n), groups=groups)):
        fold[te] = k
    ds = ManifestDataset(args.manifest, "unused", size=8)
    tr, va = (np.array(i) for i in split_indices(ds, 0.2, 0))

    print(f"{'set':36s}{'material CV: r':>14s}{'bal':>8s}{'rho>=2':>7s}{'AUC':>7s} | {'CNN split: r':>12s}{'bal':>8s}"
          f"{'rho>=2':>7s}{'AUC':>7s} | {'double hold-out: r':>18s}{'bal':>8s}{'rho>=2':>7s}{'AUC':>7s}", flush=True)
    cv_preds = {}
    for name, cols in SETS.items():
        X = np.stack([F[c] for c in cols], 1)
        cv = np.zeros(n)
        for k in range(5):
            te = fold == k
            cv[te] = fit_predict("boost", X[~te], yg[~te], X[te])
        ho = np.zeros(n)
        for k in range(5):
            for ill in names:
                te, trn = (fold == k) & (illum == ill), (fold != k) & (illum != ill)
                ho[te] = fit_predict("boost", X[trn], yg[trn], X[te])
        pv = fit_predict("boost", X[tr], yg[tr], X[va])
        cv_preds[name] = cv
        print(f"{name:36s}{fmt(metrics(cv, yg)):>35s} | {fmt(metrics(pv, yg[va])):>35s} | {fmt(metrics(ho, yg)):>38s}",
              flush=True)

    print("\n== bias (prediction - rating, rating points, minus the overall shift) for the 5-fold predictions ==")
    met = np.select([y["metallicness"] <= 0, y["metallicness"] <= 2 / 6, y["metallicness"] <= 4 / 6],
                    ["1", "2-3", "4-5"], "6-7")
    rate = np.array([f"{int(round(v * 6)) + 1}" for v in yg])
    ref, allc = cv_preds["trio + B (reference)"], cv_preds["+ all reflection cues"]
    err_table("by metallicness rating", met, ["1", "2-3", "4-5", "6-7"], ref=ref, refl=allc, y=yg)
    err_table("by glossiness rating", rate, [str(i) for i in range(1, 8)], ref=ref, refl=allc, y=yg)

    print("\n== specificity of the reflection cues (Spearman with glossiness; partial Spearman with the others, "
          "glossiness removed) ==")
    print(f"{'cue':14s}{'glossiness':>12s}{'contgloss':>12s}{'refsharp':>12s}{'metallic':>12s}{'lightness':>12s}")
    for c in REFL:
        print(f"{c:14s}{spearmanr(F[c], yg)[0]:12.3f}" + "".join(
            f"{partial_spearman(F[c], y[a], yg):12.3f}" for a in ("contgloss", "refsharp", "metallicness", "lightness")))


if __name__ == "__main__":
    main()
