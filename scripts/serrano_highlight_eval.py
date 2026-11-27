"""Apply the stereo-trained highlight networks to Serrano and predict glossiness from their maps, leaving one shape out.

For each checkpoint (e.g. runs/highlight_residual = b2, runs/highlight_spec = b1) every Serrano image (sRGB, 128 px) is
mapped to a highlight map M (squared network output, in target units). Cues on the object pixels of each geometry
(shape_masks.py): mean and quantiles of M, share of pixels above fixed thresholds, and for the region H = {M > t}
the luminance contrast against the rest of the object and the mean gradient magnitude of M at its border (sharpness).
Boosted trees on these cues are then evaluated exactly like shape_holdout.py (train on the training materials of 8
geometries, test on the validation materials of the 9th) and compared with the earlier models saved by
`shape_holdout.py --save-preds`. Also reported: mean prediction for matte images (human rating 1) per geometry.
"""
import argparse
import os

import numpy as np
import torch
import torch.nn.functional as F

from scripts.cue_model import fit_predict
from scripts.reflection_eval import fmt, metrics
from scripts.shape_masks import geometry_masks, geometry_of
from src.data.datasets import ManifestDataset
from scripts.train_highlight import GREY
from src.models.unet import UNet
from src.train import split_indices

THR = (0.005, 0.01, 0.02, 0.05, 0.1)
LUM = torch.tensor([0.2126, 0.7152, 0.0722]).view(1, 3, 1, 1)


def map_cues(x, M, obj):
    """x: (B,3,H,W) sRGB in [0,1]; M: (B,H,W) highlight map; obj: (H,W) bool. Returns (B, n_cues) and names."""
    lin = x.clamp(0, 1) ** 2.2
    Y = (lin * LUM.to(x.device)).sum(1)
    Mo = M[:, obj]
    Yo = Y[:, obj]
    feats, names = [Mo.mean(1)], ["m_mean"]
    for q in (0.5, 0.9, 0.99):
        feats.append(torch.quantile(Mo, q, dim=1))
        names.append(f"m_q{int(q * 100)}")
    for t in THR:
        feats.append((Mo > t).float().mean(1))
        names.append(f"frac_{t}")
    gy, gx = torch.gradient(M, dim=(1, 2))
    g = (gx ** 2 + gy ** 2).sqrt()[:, obj]
    for t in (0.02, 0.05):
        H = Mo > t
        n = H.sum(1).clamp_min(1)
        y_in = (Yo * H).sum(1) / n
        y_out = (Yo * ~H).sum(1) / (~H).sum(1).clamp_min(1)
        feats.append(torch.where(H.any(1), (y_in - y_out) / (y_in + y_out).clamp_min(1e-6), torch.zeros_like(y_in)))
        feats.append((g * H).sum(1) / n)
        names += [f"contrast_{t}", f"grad_{t}"]
    return torch.stack(feats, 1), names


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", default="data/serrano/manifest_all9.csv")
    p.add_argument("--root", default="data/serrano/images")
    p.add_argument("--ckpts", default="runs/highlight_residual/last.pt,runs/highlight_spec/last.pt")
    p.add_argument("--prev", default="runs/highlight_cues/shape_preds_all9.npz", help="earlier models for comparison")
    p.add_argument("--out", default="runs/highlight_cues/serrano_highlight_maps.npz")
    args = p.parse_args()

    ds = ManifestDataset(args.manifest, args.root, size=128, cache=True)
    paths = np.array([r[0] for r in ds.rows])
    y = np.array([r[1] for r in ds.rows])
    geo = np.array([geometry_of(pth) for pth in paths])
    geoms = sorted(set(geo))
    tr_all, va_all = (np.array(i) for i in split_indices(ManifestDataset(args.manifest, "unused", size=8), 0.2, 0))
    ds256 = ManifestDataset(args.manifest, args.root, size=256)
    masks, sils = {}, {}
    for g in geoms:
        obj, _, _, sil, _ = geometry_masks(ds256, np.flatnonzero(geo == g))
        masks[g] = F.avg_pool2d(obj.float()[None, None], 2)[0, 0] > 0.99
        sils[g] = F.avg_pool2d(torch.from_numpy(sil).float()[None, None], 2)[0, 0] > 0.5  # silhouette, for --bg grey

    cues = {}
    for ck_path in args.ckpts.split(","):
        if not os.path.exists(ck_path):
            print(f"missing {ck_path}, skipped")
            continue
        ck = torch.load(ck_path)
        a = ck["args"]
        model = UNet(3, 1, a["base"], a["depth"], head="map").cuda().eval()
        model.load_state_dict(ck["model"])
        name = f"stereo-trained map: {a['target']}" + (" (grey bg)" if a.get("bg", "scene") == "grey" else "")
        X = None
        with torch.no_grad():
            for g in geoms:
                idx = np.flatnonzero(geo == g)
                obj = masks[g].cuda()
                sil = sils[g].cuda()
                for s in range(0, len(idx), 64):
                    x = torch.stack([ds[i][0] for i in idx[s:s + 64]]).cuda()
                    if a.get("bg", "scene") == "grey":  # same background removal as in training
                        x = torch.where(sil, x, torch.full_like(x, GREY))
                    M = torch.sigmoid(model(x))[:, 0] ** 2
                    c, cn = map_cues(x, M, obj)
                    if X is None:
                        X = np.full((len(ds), c.shape[1]), np.nan, np.float32)
                    X[idx[s:s + 64]] = c.cpu().numpy()
        cues[name] = (X, cn)
        print(f"{name}: cues {cn}", flush=True)
    np.savez(args.out, paths=paths, **{k.split(": ")[1]: v[0] for k, v in cues.items()})

    preds = {}
    prev = np.load(args.prev, allow_pickle=True) if os.path.exists(args.prev) else None
    if prev is not None:
        assert (prev["paths"] == paths).all()
        for n_, pv in zip(prev["names"], prev["preds"]):
            if n_ in ("cues: trio + B", "U-Net RGB (shape hold-out)"):
                preds[str(n_)] = pv
    for name, (X, _) in cues.items():
        pv = np.full(len(paths), np.nan)
        for g in geoms:
            train, test = tr_all[geo[tr_all] != g], va_all[geo[va_all] == g]
            pv[test] = fit_predict("boost", X[train], y[train], X[test])
        preds[name] = pv

    names = list(preds)
    print(f"\n{'held-out geometry':22s}{'model':34s}{'r':>6s}{'bal':>8s}{'rho>=2':>7s}{'AUC':>7s}   mean human / model   matte images")
    for g in geoms:
        test = va_all[geo[va_all] == g]
        matte = test[y[test] * 6 + 1 <= 1]
        for name in names:
            pv = preds[name][test]
            print(f"{g[:21]:22s}{name:34s}{fmt(metrics(pv, y[test]))}   {y[test].mean() * 6 + 1:.2f} / "
                  f"{pv.mean() * 6 + 1:.2f}        {preds[name][matte].mean() * 6 + 1:.2f}")
        print()
    print(f"{'mean over geometries':22s}{'model':34s}{'r':>6s}{'bal':>8s}{'rho>=2':>7s}{'AUC':>7s}  |shape err|  matte mean")
    for name in names:
        ms, errs, mat = [], [], []
        for g in geoms:
            test = va_all[geo[va_all] == g]
            pv = preds[name][test]
            ms.append(metrics(pv, y[test]))
            errs.append(abs(pv.mean() - y[test].mean()) * 6)
            mat.append(preds[name][test[y[test] * 6 + 1 <= 1]].mean() * 6 + 1)
        print(f"{'':22s}{name:34s}{fmt(np.mean(ms, 0))}  {np.mean(errs):10.2f}  {np.mean(mat):10.2f}")


if __name__ == "__main__":
    main()
