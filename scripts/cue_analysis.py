"""Error analysis of the highlight-cue model and two further cue sets (A, B) on the Serrano sphere images.

0. Error analysis of the cue model (and of the U-Net) by material family, metallicness and rating level.
A. Illumination-discounting cues: background descriptors and the highlight peak relative to the brightest background.
B. Threshold-free cues: quantiles of the gloss map P and of the luminance over the sphere.
Each cue set is judged by prediction quality and by how well it reproduces the effect of the 9 illuminations on the
human ratings.
"""
import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

from scripts.cue_model import TRIO, bal_mse, fill, fit_predict, load
from scripts.gloss_map_baseline import sphere_mask
from scripts.highlight_cues import BG, P_LEVELS, Y_LEVELS, cues_for_batch
from scripts.visualize_maps import build_model
from src.data.datasets import ManifestDataset, material_groups, material_of
from src.models.features import GlossFeatures
from src.train import split_indices

PQ = [f"Pq{round(q * 1000)}" for q in P_LEVELS]
YR = [f"Yr{round(q * 1000)}" for q in Y_LEVELS]
A_NORM = ["peak_rel", "peak_rel_max"]
SETS = {
    "trio": TRIO,
    "B: quantiles only": PQ + YR,
    "trio + B": TRIO + PQ + YR,
    "trio + A: normalised peak": TRIO + A_NORM,
    "trio + A: background descriptors": TRIO + BG,
    "trio + A (both)": TRIO + BG + A_NORM,
    "trio + A + B": TRIO + BG + A_NORM + PQ + YR,
}


def family(material):
    for prefix, name in (("MERL_EDIT", "MERL_EDIT"), ("MERL-", "MERL"), ("RGL", "RGL"), ("UTIA_ISO", "UTIA_ISO")):
        if material.startswith(prefix):
            return name
    return "UTIA"


def illum_effect(pred, y, illum):
    """Spread (max - min, rating points) of the illumination means for humans and for the model, and rank correlation."""
    names = sorted(set(illum))
    h = np.array([y[illum == i].mean() * 6 + 1 for i in names])
    m = np.array([pred[illum == i].mean() * 6 + 1 for i in names])
    return h.max() - h.min(), m.max() - m.min(), spearmanr(h, m)[0]


def err_table(title, labels, order, **preds_y):
    print(title)
    y = preds_y.pop("y")
    cols = {k: ((v - y) * 6) - ((v - y) * 6).mean() for k, v in preds_y.items()}  # bias relative to the overall shift
    print(f"  {'':16s}{'n':>6s}" + "".join(f"{k + ' bias':>18s}{k + ' MAE':>16s}" for k in cols))
    for g in order:
        s = labels == g
        if s.any():
            print(f"  {str(g):16s}{int(s.sum()):6d}" + "".join(
                f"{e[s].mean():+18.2f}{np.abs(e[s]).mean():16.2f}" for e in cols.values()))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", default="data/serrano/manifest_sphere.csv")
    p.add_argument("--root", default="data/serrano/images")
    p.add_argument("--cues", default="runs/highlight_cues/cues_q95.csv")
    p.add_argument("--out", default="runs/highlight_cues")
    args = p.parse_args()
    out = Path(args.out)

    paths, y, feats = load(args.manifest, args.cues)
    n = len(paths)
    feats["peak_rel"] = np.log10((feats["peak_lum"] + 1e-3) / (feats["bg_p99"] + 1e-3))
    feats["peak_rel_max"] = np.log10((feats["peak_lum"] + 1e-3) / (feats["bg_max"] + 1e-3))
    F = fill(feats)
    yg = y["glossiness"]
    materials = [material_of(pth) for pth in paths]
    g_of = material_groups(set(materials))
    groups = np.array([g_of[m] for m in materials])
    illum = np.array([pth.split("]")[0][1:] for pth in paths])
    fam = np.array([family(m) for m in materials])

    ds = ManifestDataset(args.manifest, "unused", size=8)
    tr, va = (np.array(i) for i in split_indices(ds, 0.2, 0))
    yv, glossy_v = yg[va], yg[va] > 0

    # U-Net predictions on the validation split (128 px, as trained)
    img_ds = ManifestDataset(args.manifest, args.root, size=128)
    cnn = {}
    for tag, ck in (("U-Net RGB", "runs/sphere_groupsplit_s0/best.pt"),
                    ("U-Net 21ch", "runs/sphere_multiscale_s0/best.pt")):
        c = torch.load(ck)
        m = build_model(c["args"])
        m.load_state_dict(c["model"])
        m.cuda().eval()
        with torch.no_grad():
            pr = [m(torch.stack([img_ds[i][0] for i in va[s:s + 32]]).cuda()).mean((1, 2, 3)).cpu()
                  for s in range(0, len(va), 32)]
        cnn[tag] = torch.cat(pr).numpy()

    # ---- cue sets: 5-fold material-group predictions and the CNN split --------------------------------------
    oofs, pvs = {}, {}
    for name, cols in SETS.items():
        X = np.stack([F[c] for c in cols], 1)
        oof = np.zeros(n)
        for a, b in GroupKFold(5).split(X, groups=groups):
            oof[b] = fit_predict("boost", X[a], yg[a], X[b])
        oofs[name], pvs[name] = oof, fit_predict("boost", X[tr], yg[tr], X[va])

    print("\n== cue sets (boosted trees); glossiness. 'split' = same train/validation split as the CNN runs ==")
    head = (f"{'set':34s}{'cv r':>7s}{'cv bal':>8s}{'cv rho>=2':>10s}{'cv AUC':>8s}{'cv illum spread':>17s}"
            f"{'rank':>6s} | {'split r':>8s}{'bal':>8s}{'rho>=2':>8s}{'AUC':>6s}{'spread':>8s}{'rank':>6s}")
    print(head)
    gl_all = yg > 0
    hs, ms, rk = illum_effect(oofs["trio"], yg, illum)
    print(f"{'(human, all 520 materials)':34s}{'':>7s}{'':>8s}{'':>10s}{'':>8s}{hs:17.2f}")
    for name in list(SETS) + list(cnn):
        if name in oofs:
            o = oofs[name]
            cv = (f"{np.corrcoef(o, yg)[0, 1]:7.3f}{bal_mse(o, yg):8.4f}{spearmanr(o[gl_all], yg[gl_all])[0]:10.3f}"
                  f"{roc_auc_score(gl_all, o):8.3f}")
            hs, ms, rk = illum_effect(o, yg, illum)
            cv += f"{ms:17.2f}{rk:6.2f}"
            pv = pvs[name]
        else:
            cv, pv = " " * (7 + 8 + 10 + 8 + 17 + 6), cnn[name]
        hs_v, ms_v, rk_v = illum_effect(pv, yv, illum[va])
        print(f"{name:34s}{cv} | {np.corrcoef(pv, yv)[0, 1]:8.3f}{bal_mse(pv, yv):8.4f}"
              f"{spearmanr(pv[glossy_v], yv[glossy_v])[0]:8.3f}{roc_auc_score(glossy_v, pv):6.3f}{ms_v:8.2f}{rk_v:6.2f}")
    print(f"(human spread on the split: {illum_effect(pvs['trio'], yv, illum[va])[0]:.2f})")

    # ---- 0. error analysis ------------------------------------------------------------------------------------
    print("\n== 0. error analysis (bias = mean prediction - rating in rating points, minus the overall shift) ==")
    o = oofs["trio"]
    met = np.select([y["metallicness"] <= 0, y["metallicness"] <= 2 / 6, y["metallicness"] <= 4 / 6],
                    ["1", "2-3", "4-5"], "6-7")
    rate = np.array([f"{int(round(v * 6)) + 1}" for v in yg])
    err_table("cue model 'trio' (5-fold), by material family", fam,
              ["MERL", "MERL_EDIT", "RGL", "UTIA", "UTIA_ISO"], trio=o, y=yg)
    err_table("by metallicness rating", met, ["1", "2-3", "4-5", "6-7"], trio=o, y=yg)
    err_table("by glossiness rating", rate, [str(i) for i in range(1, 8)], trio=o, y=yg)
    print("\nsame groups on the CNN split, cue model vs U-Net (RGB)")
    err_table("by material family", fam[va], ["MERL", "MERL_EDIT", "RGL", "UTIA", "UTIA_ISO"],
              cues=pvs["trio"], unet=cnn["U-Net RGB"], y=yv)
    err_table("by metallicness rating", met[va], ["1", "2-3", "4-5", "6-7"],
              cues=pvs["trio"], unet=cnn["U-Net RGB"], y=yv)

    # montage of the largest errors of the cue model
    e = (o - yg) - (o - yg).mean()
    order = np.argsort(e)
    picks = list(order[-6:][::-1]) + list(order[:6])  # over-predicted, then under-predicted
    thr = json.load(open(args.cues.replace(".csv", ".json")))["threshold"]
    full = ManifestDataset(args.manifest, args.root, size=256)
    dist, radius, _ = sphere_mask(full)
    obj, bg = (dist <= radius - 10).cuda(), (dist >= radius + 8).cuda()
    feat = GlossFeatures(32, sigma_step=1).cuda()
    xs = torch.stack([full[i][0] for i in picks]).cuda()
    with torch.no_grad():
        _, Hm, _ = cues_for_batch(xs, feat, obj, thr, bg)
    fig, axes = plt.subplots(2, 6, figsize=(15, 6.6))
    for k, (ax, i) in enumerate(zip(axes.ravel(), picks)):
        ax.imshow(xs[k].cpu().permute(1, 2, 0))
        if Hm[k].any():
            ax.contour(Hm[k].cpu().numpy(), levels=[0.5], colors="red", linewidths=0.8)
        ax.set_title(f"{'over' if k < 6 else 'under'}: rating {yg[i] * 6 + 1:.0f}, pred {o[i] * 6 + 1:.1f}\n"
                     f"metal {y['metallicness'][i] * 6 + 1:.0f}  {fam[i]}\n{materials[i][:34]}", fontsize=7)
        ax.axis("off")
    fig.savefig(out / "error_montage.png", dpi=120, bbox_inches="tight")
    print("\nsaved", out / "error_montage.png")


if __name__ == "__main__":
    main()
