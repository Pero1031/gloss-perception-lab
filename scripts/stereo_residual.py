"""Binocular highlight pilot: does the left-right residual keep highlights and drop matte shading?

Reads the views written by `stereo_render.py`. For each scene the right view (R: 6.5 cm baseline, R2: 13 cm) is warped
onto the left view with the rendered world positions (exact correspondence; occluded pixels are dropped) and the residual
contrast |L - R_warped| / (L + R_warped) of the luminance is computed. Diffuse (Lambertian) shading is the same from both
eyes and cancels; view-dependent reflections do not.

The left image is also run through the gloss map P (as in highlight_cues.py: sRGB, 256 px, Serrano threshold),
which marks highlight candidates H. Reported per scene: how much of H is real specular reflection (rendered glossy pass)
and the residual inside H. The core check: on matte materials of fine-detailed shapes, P fires on bright shading; the
residual there should stay at the noise floor, while on glossy materials it should be large.
"""
import argparse
import glob
import json
import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from src.models.features import GlossFeatures

matplotlib.rcParams.update({"font.family": "sans-serif", "font.size": 8})
LUM = np.array([0.2126, 0.7152, 0.0722], np.float32)


def blur(x, sigma):
    r = int(3 * sigma)
    k = torch.exp(-torch.arange(-r, r + 1, dtype=torch.float32) ** 2 / (2 * sigma ** 2))
    k = (k / k.sum()).to(x.device)
    x = F.conv2d(F.pad(x, (r, r, 0, 0), mode="replicate"), k.view(1, 1, 1, -1))
    return F.conv2d(F.pad(x, (0, 0, r, r), mode="replicate"), k.view(1, 1, -1, 1))


def warp_to(L, R, tol=0.003):
    """Sample view R at the pixels of view L via L's world positions. Returns sampled luminance and validity."""
    H, W = L["mask"].shape
    X = np.concatenate([L["position"], np.ones((H, W, 1), np.float32)], -1).reshape(-1, 4)
    clip = X @ (R["proj"] @ R["view"]).T
    ndc = clip[:, :2] / clip[:, 3:4]
    grid = torch.from_numpy(np.stack([ndc[:, 0], -ndc[:, 1]], -1).reshape(1, H, W, 2))
    to_t = lambda a: torch.from_numpy(np.ascontiguousarray(a)).permute(2, 0, 1)[None].float()
    samp = lambda a, mode: F.grid_sample(to_t(a), grid, mode=mode, align_corners=False)[0].permute(1, 2, 0).numpy()
    yR = samp((R["rgb"] @ LUM)[..., None], "bilinear")[..., 0]
    pR = samp(R["position"], "nearest")
    mR = samp(R["mask"][..., None].astype(np.float32), "nearest")[..., 0] > 0.5
    ok = L["mask"] & mR & (np.linalg.norm(pR - L["position"], axis=-1) < tol)
    ok &= (np.abs(ndc) < 1).all(1).reshape(H, W)
    return yR, ok


def to_srgb256(rgb, mask):
    """Auto-exposed sRGB at 256 px (99.5th percentile of object luminance -> 0.95)."""
    y = rgb @ LUM
    s = 0.95 / max(np.percentile(y[mask], 99.5), 1e-6)
    x = torch.from_numpy(np.clip(rgb * s, 0, 1) ** (1 / 2.2)).permute(2, 0, 1)[None].float()
    return F.avg_pool2d(x, rgb.shape[0] // 256)


def pool(a):
    t = torch.from_numpy(np.ascontiguousarray(a)).float()[None, None]
    return F.avg_pool2d(t, a.shape[0] // 256)[0, 0].numpy()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="runs/stereo_pilot")
    p.add_argument("--meta", default="runs/highlight_cues/cues_all9_q95.json")
    p.add_argument("--sigma", type=float, default=1.5, help="blur (px at render size) before comparing the views")
    p.add_argument("--T", type=float, default=0.02, help="residual threshold for the residual detector")
    args = p.parse_args()

    meta = json.load(open(args.meta))
    feat = GlossFeatures(meta["sigma_max"], sigma_step=1)
    thr = meta["threshold"]
    scenes = sorted(d for d in glob.glob(os.path.join(args.root, "*")) if os.path.exists(os.path.join(d, "L.npz")))
    rows, maps = [], {}
    for d in scenes:
        name = os.path.basename(d)
        shape, mat, world = name.rsplit("_", 2)
        L = dict(np.load(os.path.join(d, "L.npz")))
        x = to_srgb256(L["rgb"], L["mask"])
        with torch.no_grad():
            P = feat.gloss_map(x)[0].numpy()
        obj = pool(L["mask"].astype(np.float32)) > 0.99
        Hm = (P > thr) & obj
        yL = L["rgb"] @ LUM
        spec = pool(L["glossy"] @ LUM) / np.maximum(pool(yL), 1e-6)  # specular fraction of the left image
        row = dict(scene=name, shape=shape, mat=mat, world=world, H_frac=Hm.sum() / obj.sum(),
                   H_spec=float(np.median(spec[Hm])) if Hm.any() else np.nan)
        for cam in ("R", "R2"):
            if not os.path.exists(os.path.join(d, cam + ".npz")):
                continue
            R = dict(np.load(os.path.join(d, cam + ".npz")))
            yR, ok = warp_to(L, R)
            a = blur(torch.from_numpy(np.where(ok, yL, 0))[None, None], args.sigma)
            b = blur(torch.from_numpy(np.where(ok, yR, 0))[None, None], args.sigma)
            w = blur(torch.from_numpy(ok.astype(np.float32))[None, None], args.sigma)
            a, b = (a / w.clamp_min(1e-6))[0, 0].numpy(), (b / w.clamp_min(1e-6))[0, 0].numpy()
            res = np.abs(a - b) / np.maximum(a + b, 1e-6)
            res256, ok256 = pool(np.where(ok, res, 0)), pool(ok.astype(np.float32)) > 0.99
            sel = Hm & ok256
            row[f"res_H_{cam}"] = float(np.median(res256[sel])) if sel.any() else np.nan
            row[f"res_obj_{cam}"] = float(np.median(res256[obj & ok256]))
            row[f"valid_{cam}"] = float((ok256 & obj).sum() / obj.sum())
            # does the residual find real specular pixels? AUC over object pixels: spec fraction > 0.5 vs < 0.1
            pos, neg = res256[obj & ok256 & (spec > 0.5)], res256[obj & ok256 & (spec < 0.1)]
            if len(pos) > 20 and len(neg) > 20:
                ranks = np.argsort(np.argsort(np.concatenate([pos, neg])))
                row[f"auc_{cam}"] = float((ranks[:len(pos)].sum() - len(pos) * (len(pos) - 1) / 2) / (len(pos) * len(neg)))
            # residual as a detector (threshold T) vs the P detector H: share of flagged object pixels, precision
            # (flagged pixels whose light is mostly specular, spec > 0.3) and recall (of pixels with spec > 0.5)
            det = (res256 > args.T) & obj & ok256
            row[f"det_frac_{cam}"] = float(det.sum() / obj.sum())
            row[f"det_prec_{cam}"] = float((spec[det] > 0.3).mean()) if det.any() else np.nan
            row[f"det_rec_{cam}"] = float(det[obj & ok256 & (spec > 0.5)].mean()) if (obj & ok256 & (spec > 0.5)).any() else np.nan
            if cam == "R":
                row["H_prec"] = float((spec[Hm] > 0.3).mean()) if Hm.any() else np.nan
                maps[name] = dict(img=x[0].permute(1, 2, 0).numpy(), H=Hm, res=np.where(ok256, res256, np.nan),
                                  spec=np.where(obj, spec, np.nan))
        rows.append(row)

    keys = ["H_frac", "H_prec", "det_frac_R", "det_prec_R", "det_rec_R", "det_rec_R2", "res_H_R", "auc_R", "valid_R"]
    print(f"{'scene':34s}" + "".join(f"{k:>10s}" for k in keys))
    for r in rows:
        print(f"{r['scene']:34s}" + "".join(f"{r.get(k, np.nan):10.3f}" for k in keys))
    json.dump(rows, open(os.path.join(args.root, "residual_summary.json"), "w"), indent=1)

    # figure: per shape, matte and glossy under the first world: image with H, residual, rendered specular fraction
    world = rows[0]["world"]
    shapes = sorted({r["shape"] for r in rows}, key=["sphere", "blob", "blob_fine", "erato"].index)
    cols = [("matte", "img"), ("matte", "res"), ("glossy", "img"), ("glossy", "res"), ("glossy", "spec")]
    titles = {"img": "image + P detections (red)", "res": "left-right residual", "spec": "true specular fraction"}
    fig, axs = plt.subplots(len(shapes), len(cols), figsize=(2.3 * len(cols), 2.4 * len(shapes)), squeeze=False,
                            facecolor="#fcfcfb", constrained_layout=True)
    for i, s in enumerate(shapes):
        for j, (m, k) in enumerate(cols):
            ax, mp = axs[i, j], maps.get(f"{s}_{m}_{world}")
            ax.set_xticks([]), ax.set_yticks([])
            if mp is None:
                continue
            if k == "img":
                ax.imshow(mp["img"])
                ax.contour(mp["H"].astype(float), levels=[0.5], colors=["#e34948"], linewidths=0.6)
            else:
                ax.set_facecolor("#d9d8d4")
                ax.imshow(mp[k], cmap="magma", vmin=0, vmax=0.5 if k == "res" else 1)
            if i == 0:
                ax.set_title(f"{m}: {titles[k]}", fontsize=8)
            if j == 0:
                ax.set_ylabel(s, fontsize=9)
    fig.suptitle(f"Light {world}, baseline 6.5 cm. Residual = |L - R| / (L + R) (shown 0-0.5)", x=0.01, ha="left", fontsize=9)
    fig.savefig(os.path.join(args.root, f"residual_maps_{world}.png"), dpi=150, bbox_inches="tight")
    print("saved", os.path.join(args.root, f"residual_maps_{world}.png"))


if __name__ == "__main__":
    main()
