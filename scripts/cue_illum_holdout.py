"""Do the illumination-discounting cues work for an illumination the model has never seen?

Double hold-out: every image is predicted by a model trained without its material group (5 material folds) AND without
its illumination (9 illuminations). Compare with plain material-group cross-validation, where all 9 illuminations are
seen in training and a model could simply memorise a correction per illumination.
"""
import argparse

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

from scripts.cue_analysis import A_NORM, BG, PQ, YR, illum_effect
from scripts.cue_model import TRIO, bal_mse, fill, fit_predict, load
from src.data.datasets import material_groups, material_of

SETS = {
    "trio": TRIO,
    "trio + B (quantiles)": TRIO + PQ + YR,
    "trio + A: normalised peak": TRIO + A_NORM,
    "trio + A: background descriptors": TRIO + BG,
    "trio + A (both)": TRIO + BG + A_NORM,
    "trio + A + B": TRIO + BG + A_NORM + PQ + YR,
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", default="data/serrano/manifest_sphere.csv")
    p.add_argument("--cues", default="runs/highlight_cues/cues_q95.csv")
    args = p.parse_args()

    paths, y, feats = load(args.manifest, args.cues)
    feats["peak_rel"] = np.log10((feats["peak_lum"] + 1e-3) / (feats["bg_p99"] + 1e-3))
    feats["peak_rel_max"] = np.log10((feats["peak_lum"] + 1e-3) / (feats["bg_max"] + 1e-3))
    F = fill(feats)
    yg, n = y["glossiness"], len(paths)
    materials = [material_of(pth) for pth in paths]
    g_of = material_groups(set(materials))
    groups = np.array([g_of[m] for m in materials])
    illum = np.array([pth.split("]")[0][1:] for pth in paths])
    names = sorted(set(illum))
    fold = np.zeros(n, dtype=int)
    for k, (_, te) in enumerate(GroupKFold(5).split(np.zeros(n), groups=groups)):
        fold[te] = k
    glossy = yg > 0

    print(f"{'set':34s}{'material CV':>34s} | {'material x illumination hold-out':>46s}")
    print(f"{'':34s}{'r':>8s}{'bal':>8s}{'spread':>8s}{'rank':>8s} | {'r':>8s}{'bal':>8s}{'rho>=2':>8s}{'AUC':>8s}"
          f"{'spread':>8s}{'rank':>8s}")
    human, _, _ = illum_effect(np.zeros(n), yg, illum)
    print(f"{'(human spread, 520 materials)':34s}{'':>16s}{human:8.2f}")
    for name, cols in SETS.items():
        X = np.stack([F[c] for c in cols], 1)
        cv = np.zeros(n)
        for k in range(5):
            te = fold == k
            cv[te] = fit_predict("boost", X[~te], yg[~te], X[te])
        ho = np.zeros(n)
        for k in range(5):
            for ill in names:
                te = (fold == k) & (illum == ill)
                tr = (fold != k) & (illum != ill)
                ho[te] = fit_predict("boost", X[tr], yg[tr], X[te])
        _, s1, r1 = illum_effect(cv, yg, illum)
        _, s2, r2 = illum_effect(ho, yg, illum)
        print(f"{name:34s}{np.corrcoef(cv, yg)[0, 1]:8.3f}{bal_mse(cv, yg):8.4f}{s1:8.2f}{r1:8.2f} | "
              f"{np.corrcoef(ho, yg)[0, 1]:8.3f}{bal_mse(ho, yg):8.4f}{spearmanr(ho[glossy], yg[glossy])[0]:8.3f}"
              f"{roc_auc_score(glossy, ho):8.3f}{s2:8.2f}{r2:8.2f}", flush=True)


if __name__ == "__main__":
    main()
