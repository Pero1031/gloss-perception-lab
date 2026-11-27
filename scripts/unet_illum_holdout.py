"""U-Net and cue models under the same material x illumination hold-out.

Validation materials are those of the usual split (train.py --seed 0). The nine illuminations are split into three folds of
three (names sorted, fold k = k, k+3, k+6). For fold k a model is trained on the training materials under the other six
illuminations and predicts the validation materials under the three held-out ones; the three folds together give one
prediction for each of the 936 validation images. The U-Net models come from
`train.py --illum-fold k --out runs/unet_illum_f{k}` (last epoch, no selection on the held-out data).
"""
import argparse

import numpy as np
import torch

from scripts.cue_analysis import illum_effect
from scripts.cue_model import TRIO, fill, fit_predict, load
from scripts.reflection_cues import REFL
from scripts.reflection_eval import BASE, fmt, metrics
from scripts.visualize_maps import build_model
from src.data.datasets import ManifestDataset
from src.train import split_indices

CUE_SETS = {
    "cues: trio": TRIO,
    "cues: trio + B": BASE,
    "cues: trio + B + reflection": BASE + REFL,
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", default="data/serrano/manifest_sphere.csv")
    p.add_argument("--root", default="data/serrano/images")
    p.add_argument("--cues", default="runs/highlight_cues/cues_q95.csv")
    p.add_argument("--refl", default="runs/highlight_cues/reflection_q95.csv")
    p.add_argument("--unet-dir", default="runs/unet_illum_f{k}")
    args = p.parse_args()

    paths, y, f1 = load(args.manifest, args.cues)
    _, _, f2 = load(args.manifest, args.refl)
    F = {**fill(f1), **fill(f2)}
    yg = y["glossiness"]
    illum = np.array([pth.split("]")[0][1:] for pth in paths])
    names = sorted(set(illum))
    ds = ManifestDataset(args.manifest, "unused", size=8)
    tr_all, va_all = (np.array(i) for i in split_indices(ds, 0.2, 0))
    img_ds = ManifestDataset(args.manifest, args.root, size=128)

    preds = {k: np.full(len(paths), np.nan) for k in list(CUE_SETS) + ["U-Net RGB (illumination hold-out)"]}
    for k in range(3):
        held = {names[i] for i in range(k, len(names), 3)}
        in_held = np.array([i in held for i in illum])
        train = tr_all[~in_held[tr_all]]
        test = va_all[in_held[va_all]]
        for name, cols in CUE_SETS.items():
            X = np.stack([F[c] for c in cols], 1)
            preds[name][test] = fit_predict("boost", X[train], yg[train], X[test])
        ck = torch.load(args.unet_dir.format(k=k) + "/last.pt")
        model = build_model(ck["args"])
        model.load_state_dict(ck["model"])
        model.cuda().eval()
        with torch.no_grad():
            out = [model(torch.stack([img_ds[i][0] for i in test[s:s + 32]]).cuda()).mean((1, 2, 3)).cpu()
                   for s in range(0, len(test), 32)]
        preds["U-Net RGB (illumination hold-out)"][test] = torch.cat(out).numpy()
        print(f"fold {k}: held-out illuminations {sorted(held)}; train {len(train)} images, test {len(test)}", flush=True)

    yv, illum_v = yg[va_all], illum[va_all]
    print(f"\n{'model':36s}{'r':>6s}{'bal':>8s}{'rho>=2':>7s}{'AUC':>7s}{'illum spread':>14s}{'rank corr':>11s}")
    for name, pr in preds.items():
        pv = pr[va_all]
        assert not np.isnan(pv).any()
        hs, ms, rk = illum_effect(pv, yv, illum_v)
        print(f"{name:36s}{fmt(metrics(pv, yv))}{ms:14.2f}{rk:11.2f}")
    print(f"{'(human, 104 validation materials)':36s}{'':>28s}{hs:14.2f}")
    print("\nreference, illuminations seen in training (earlier runs, epoch chosen on the validation data):")
    print("  U-Net RGB seed 0: r 0.903, bal 0.0359, rho>=2 0.794, AUC 0.950, spread 0.55, rank 0.67")
    print("  cues trio + B + reflection (material hold-out only): r 0.915, bal 0.0347, rho>=2 0.816, AUC 0.955")


if __name__ == "__main__":
    main()
