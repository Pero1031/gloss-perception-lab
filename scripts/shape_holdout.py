"""Generalisation to a new shape: cue models and the U-Net, trained without one geometry and tested on it.

Validation materials are those of the usual split (train.py --seed 0). For each geometry G of the manifest a model is
trained on the training materials of the other geometries and predicts the validation materials of G; the materials are
therefore new to the model, the illuminations are not, and the shape is always new.
The U-Net models come from `train.py --holdout-geometry G --out runs/unet_shape_{G}` (last epoch). Also reported:
cue models trained on the sphere only, and how well the mean rating of each geometry (human: blob high, dragon low) is
reproduced.
"""
import argparse
import os

import numpy as np
import torch

from scripts.cue_model import TRIO, fill, fit_predict, load
from scripts.reflection_cues import BG_REFL, REFL
from scripts.reflection_eval import BASE, fmt, metrics
from scripts.shape_masks import geometry_of
from scripts.visualize_maps import build_model
from src.data.datasets import ManifestDataset
from src.train import split_indices

CUE_SETS = {"cues: trio": TRIO, "cues: trio + B": BASE, "cues: trio + B + reflection": BASE + REFL}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", default="data/serrano/manifest_shapes.csv")
    p.add_argument("--root", default="data/serrano/images")
    p.add_argument("--cues", default="runs/highlight_cues/cues_shapes_q95.csv")
    p.add_argument("--refl", default="runs/highlight_cues/reflection_shapes_q95.csv")
    p.add_argument("--unet-dir", default="runs/unet_shape_{g}")
    p.add_argument("--save-preds", default=None, help="npz file for the per-image hold-out predictions")
    args = p.parse_args()

    paths, y, f1 = load(args.manifest, args.cues)
    _, _, f2 = load(args.manifest, args.refl)
    F = {**fill(f1), **fill(f2)}
    # object-vs-background reflection cues are undefined where no background is visible (surface2 fills the frame):
    # cue sets that use them are trained without those images and not tested on them
    no_bg = np.zeros(len(paths), bool)
    for c in BG_REFL:
        no_bg |= np.isnan(f2[c])
    yg = y["glossiness"]
    geo = np.array([geometry_of(pth) for pth in paths])
    geoms = sorted(set(geo))
    ds = ManifestDataset(args.manifest, "unused", size=8)
    tr_all, va_all = (np.array(i) for i in split_indices(ds, 0.2, 0))
    img_ds = ManifestDataset(args.manifest, args.root, size=128)
    print(f"geometries: {geoms}; validation materials give {len(va_all)} images in total "
          f"({len(va_all) // len(geoms)} per geometry)")

    names = list(CUE_SETS) + ["U-Net RGB (shape hold-out)"]
    preds = {n: np.full(len(paths), np.nan) for n in names}
    for g in geoms:
        train = tr_all[geo[tr_all] != g]
        test = va_all[geo[va_all] == g]
        for name, cols in CUE_SETS.items():
            X = np.stack([F[c] for c in cols], 1)
            tr, te = train, test
            if set(cols) & set(BG_REFL):
                tr, te = train[~no_bg[train]], test[~no_bg[test]]
            if len(te):
                preds[name][te] = fit_predict("boost", X[tr], yg[tr], X[te])
        ck_path = args.unet_dir.format(g=g) + "/last.pt"
        if os.path.exists(ck_path):
            ck = torch.load(ck_path)
            model = build_model(ck["args"])
            model.load_state_dict(ck["model"])
            model.cuda().eval()
            with torch.no_grad():
                out = [model(torch.stack([img_ds[i][0] for i in test[s:s + 32]]).cuda()).mean((1, 2, 3)).cpu()
                       for s in range(0, len(test), 32)]
            preds["U-Net RGB (shape hold-out)"][test] = torch.cat(out).numpy()
    if args.save_preds:
        np.savez(args.save_preds, paths=np.array(paths), geo=geo, va=va_all, glossiness=yg,
                 names=np.array(names), preds=np.stack([preds[n] for n in names]))

    print(f"\n{'held-out geometry':22s}{'model':32s}{'r':>6s}{'bal':>8s}{'rho>=2':>7s}{'AUC':>7s}   mean rating human / model")
    for g in geoms:
        test = va_all[geo[va_all] == g]
        human = yg[test].mean() * 6 + 1
        for name in names:
            pv = preds[name][test]
            if np.isnan(pv).any():
                continue
            print(f"{g:22s}{name:32s}{fmt(metrics(pv, yg[test]))}   {human:.2f} / {pv.mean() * 6 + 1:.2f}")
        print()

    print("mean over held-out geometries (geometries a model could not be tested on are left out)")
    print(f"{'model':32s}{'n':>3s}{'r':>7s}{'bal':>8s}{'rho>=2':>7s}{'AUC':>7s}  |shape effect error|")
    for name in names:
        ms, errs = [], []
        for g in geoms:
            test = va_all[geo[va_all] == g]
            pv = preds[name][test]
            if np.isnan(pv).any():
                continue
            ms.append(metrics(pv, yg[test]))
            errs.append(abs(pv.mean() - yg[test].mean()) * 6)
        if ms:
            m = np.mean(ms, 0)
            print(f"{name:32s}{len(ms):3d} {fmt(m)}  {np.mean(errs):6.2f}")
    print()

    print("shape effect: mean rating per geometry over the validation materials (rating scale 1-7)")
    print(f"{'':32s}" + "".join(f"{g[:14]:>16s}" for g in geoms))
    print(f"{'human':32s}" + "".join(f"{yg[va_all[geo[va_all] == g]].mean() * 6 + 1:16.2f}" for g in geoms))
    for name in names:
        if np.isnan(preds[name][va_all]).all():
            continue
        print(f"{name:32s}" + "".join(f"{preds[name][va_all[geo[va_all] == g]].mean() * 6 + 1:16.2f}" for g in geoms))

    # cue models trained on the sphere only, tested on the other geometries
    if "sphere" in geoms:
        print("\ncue models trained on the sphere only (training materials), tested on the validation materials")
        print(f"{'test geometry':22s}{'model':32s}{'r':>6s}{'bal':>8s}{'rho>=2':>7s}{'AUC':>7s}")
        train = tr_all[geo[tr_all] == "sphere"]
        for g in geoms:
            test = va_all[geo[va_all] == g]
            for name, cols in CUE_SETS.items():
                if set(cols) & set(BG_REFL) and no_bg[test].any():
                    continue
                X = np.stack([F[c] for c in cols], 1)
                pv = fit_predict("boost", X[train], yg[train], X[test])
                print(f"{g:22s}{name:32s}{fmt(metrics(pv, yg[test]))}")


if __name__ == "__main__":
    main()
