"""Highlight regions and image-computable gloss cues for the Serrano sphere images.

The highlight region H is the part of the sphere where the gloss map P (Nagata et al.) exceeds a
threshold. Cues follow Marlow, Kim & Anderson (coverage, contrast, sharpness) plus body lightness and colour.
The operational definitions are our own (see the CUES list); luminance is linear luminance of the sRGB image.
"""
import argparse
import csv
import json
import random
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

from scripts.shape_masks import geometry_masks, geometry_of
from src.data.datasets import ManifestDataset
from src.models.features import GlossFeatures

CUES = [
    "coverage",         # |H| / |sphere|
    "contrast",         # Michelson contrast between the brightest 10% of H and the median of a ring around H
    "contrast_log",     # log10 of the same luminance ratio
    "contrast_edge",    # Michelson contrast across the boundary of H (inner band vs outer band)
    "sharp_grad",       # mean luminance gradient on the boundary of H divided by the highlight height (1/px)
    "sharp_scale",      # mean sigma of the response profile at the strongest point of P (small = sharp)
    "body_lum",         # mean luminance of the sphere outside H
    "d_sat",            # saturation of H minus saturation of the body
    "d_chrom",          # distance between the mean chromaticities of H and of the body
    "P_mean", "P_max",  # baselines: statistics of the gloss map over the sphere
    "lum_p99",          # baseline: 99th percentile of luminance over the sphere
    "has_hl",           # 1 if H is not empty
    "peak_lum", "ring_lum",  # luminance of the brightest 10% of H and median luminance of the ring around H
]
P_LEVELS = [0.5, 0.75, 0.9, 0.95, 0.98, 0.99, 0.995, 0.999]
Y_LEVELS = [0.9, 0.99, 0.999, 1.0]
# threshold-free cues: quantiles of P over the sphere, log luminance ratios of sphere quantiles to its median
CUES += [f"Pq{round(q * 1000)}" for q in P_LEVELS] + [f"Yr{round(q * 1000)}" for q in Y_LEVELS]
# illumination descriptors from the visible background (outside the sphere)
BG = ["bg_mean", "bg_p50", "bg_p99", "bg_max", "bg_range", "bg_contrast", "bright_area"]
CUES += BG


def srgb_to_linear(x):
    return torch.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)


def dilate(m, r):
    return F.max_pool2d(m.float()[:, None], 2 * r + 1, 1, r)[:, 0] > 0


def erode(m, r):
    return ~dilate(~m, r)


def mmean(v, m):
    n = m.sum((1, 2)).float()
    return (v * m).sum((1, 2)) / n.clamp_min(1)


def masked_stat(v, m, fn):
    nan = torch.full_like(v, float("nan"))
    return fn(torch.where(m, v, nan).flatten(1))


def cues_for_batch(x, feat, obj, thr, bg=None):
    """Cues and masks for a batch x of shape (B, 3, H, W) in [0, 1]. bg: mask of the background (outside the sphere)."""
    lin = srgb_to_linear(x)
    Y = 0.2126 * lin[:, 0] + 0.7152 * lin[:, 1] + 0.0722 * lin[:, 2]
    p = feat(x)[:, 1:]
    P = feat.gloss_map_from(p)
    obj = obj[None].expand_as(Y)
    nan = torch.full_like(Y[:, 0, 0], float("nan"))

    Hm = (P > thr) & obj
    has = Hm.flatten(1).any(1)
    cov = Hm.sum((1, 2)).float() / obj.sum((1, 2)).float()

    ring = dilate(Hm, 12) & ~dilate(Hm, 4) & obj
    ring = torch.where(ring.flatten(1).any(1)[:, None, None], ring, obj & ~Hm)
    Ys = masked_stat(Y, ring, lambda v: v.nanmedian(dim=1).values)
    q = masked_stat(Y, Hm, lambda v: torch.nanquantile(v, 0.9, dim=1))
    Yp = mmean(Y, Hm & (Y >= q[:, None, None]))
    contrast = (Yp - Ys) / (Yp + Ys).clamp_min(1e-6)
    contrast_log = torch.log10((Yp + 1e-3) / (Ys + 1e-3))

    inner, outer = Hm & ~erode(Hm, 2), dilate(Hm, 2) & ~Hm & obj
    contrast_edge = (mmean(Y, inner) - mmean(Y, outer)) / (mmean(Y, inner) + mmean(Y, outer)).clamp_min(1e-6)

    k = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=Y.dtype, device=Y.device) / 8
    gx = F.conv2d(Y[:, None], k[None, None], padding=1)[:, 0]
    gy = F.conv2d(Y[:, None], k.T[None, None], padding=1)[:, 0]
    grad = (gx**2 + gy**2).sqrt()
    height = Yp - Ys
    sharp_grad = torch.where(height > 1e-3, mmean(grad, inner | outer) / height.clamp_min(1e-3), nan)

    idx = (P * obj).flatten(1).argmax(1)
    prof = p.flatten(2)[torch.arange(len(x)), :, idx]  # (B, n_scales) response profile at the strongest point
    sigmas = torch.tensor(feat.sigmas, dtype=prof.dtype, device=prof.device)
    sharp_scale = (prof * sigmas).sum(1) / prof.sum(1).clamp_min(1e-12)

    body = obj & ~dilate(Hm, 4)
    sat = (lin.amax(1) - lin.amin(1)) / lin.amax(1).clamp_min(1e-6)
    chrom = lin / lin.sum(1, keepdim=True).clamp_min(1e-6)
    cH = torch.stack([mmean(chrom[:, c], Hm) for c in range(3)], 1)
    cB = torch.stack([mmean(chrom[:, c], body) for c in range(3)], 1)

    out = {
        "coverage": cov,
        "contrast": contrast, "contrast_log": contrast_log, "contrast_edge": contrast_edge,
        "sharp_grad": sharp_grad, "sharp_scale": sharp_scale,
        "body_lum": mmean(Y, body),
        "d_sat": mmean(sat, Hm) - mmean(sat, body),
        "d_chrom": (cH - cB).norm(dim=1),
        "P_mean": mmean(P, obj),
        "P_max": masked_stat(P, obj, lambda v: v.nan_to_num(-1).amax(1)),
        "lum_p99": masked_stat(Y, obj, lambda v: torch.nanquantile(v, 0.99, dim=1)),
        "has_hl": has.float(),
        "peak_lum": Yp, "ring_lum": Ys,
    }
    for k_ in ("contrast", "contrast_log", "contrast_edge", "sharp_grad", "sharp_scale", "d_sat", "d_chrom",
               "peak_lum", "ring_lum"):
        out[k_] = torch.where(has, out[k_], nan)  # undefined without a highlight

    Pq = torch.quantile(P[:, obj[0]], torch.tensor(P_LEVELS, device=P.device), dim=1)
    for q_, v in zip(P_LEVELS, Pq):
        out[f"Pq{round(q_ * 1000)}"] = v
    Yq = torch.quantile(Y[:, obj[0]], torch.tensor(Y_LEVELS + [0.5], device=Y.device), dim=1)
    for q_, v in zip(Y_LEVELS, Yq[:-1]):
        out[f"Yr{round(q_ * 1000)}"] = torch.log10((v + 1e-3) / (Yq[-1] + 1e-3))

    if bg is None or not bg.any():  # no visible background (object fills the frame)
        out.update({k_: nan.clone() for k_ in BG})
    else:
        Yb = Y[:, bg]  # (B, n_background_pixels)
        b50, b99 = Yb.median(1).values, torch.quantile(Yb, 0.99, dim=1)
        out.update({
            "bg_mean": Yb.mean(1), "bg_p50": b50, "bg_p99": b99, "bg_max": Yb.amax(1),
            "bg_range": torch.log10((b99 + 1e-3) / (b50 + 1e-3)),
            "bg_contrast": Yb.std(1) / Yb.mean(1).clamp_min(1e-6),
            "bright_area": (Yb > 0.5).float().mean(1),
        })
    return out, Hm, ring


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", default="data/serrano/manifest_sphere.csv")
    p.add_argument("--root", default="data/serrano/images")
    p.add_argument("--quantile", type=float, default=0.97,
                   help="P threshold = this quantile of P over sphere pixels of a random sample of images")
    p.add_argument("--sigma-max", type=int, default=32)
    p.add_argument("--erode", type=float, default=10)
    p.add_argument("--thr-from", help="JSON of an earlier run whose P threshold is reused (e.g. the sphere run)")
    p.add_argument("--out", default="runs/highlight_cues/cues_q97.csv")
    args = p.parse_args()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    ds = ManifestDataset(args.manifest, args.root, size=256, cache=True)
    geoms = {}
    for i, (path, _) in enumerate(ds.rows):
        geoms.setdefault(geometry_of(path), []).append(i)
    masks = {}
    for g, idx in geoms.items():
        obj_g, bg_g, _, _, agree = geometry_masks(ds, idx, erode=args.erode)
        masks[g] = (obj_g.cuda(), bg_g.cuda())
        print(f"{g}: {len(idx)} images, object pixels {int(obj_g.sum())}, "
              f"silhouette agreement across illuminations {agree:.3f}")
    feat = GlossFeatures(args.sigma_max, sigma_step=1).cuda()

    rng = random.Random(0)
    if args.thr_from:
        thr = json.load(open(args.thr_from))["threshold"]
        print(f"threshold on P = {thr:.3e} (taken from {args.thr_from})")
    else:
        pooled = []
        with torch.no_grad():
            for g, idx in geoms.items():
                for _ in range(10):
                    x = torch.stack([ds[i][0] for i in rng.sample(idx, 10)]).cuda()
                    pooled.append(feat.gloss_map(x)[:, masks[g][0]].flatten().cpu())
        thr = torch.quantile(torch.cat(pooled), args.quantile).item()
        print(f"threshold on P = {thr:.5f} (quantile {args.quantile} of P over object pixels, "
              f"100 sample images per geometry)")

    rows = {k: torch.zeros(len(ds)) for k in CUES}
    with torch.no_grad():
        for g, idx in geoms.items():
            obj, bg = masks[g]
            for s in range(0, len(idx), 8):
                sel = idx[s:s + 8]
                x = torch.stack([ds[i][0] for i in sel]).cuda()
                c, _, _ = cues_for_batch(x, feat, obj, thr, bg)
                for k in CUES:
                    rows[k][sel] = c[k].cpu()

    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["path"] + CUES)
        for i, (path, _) in enumerate(ds.rows):
            w.writerow([path] + [f"{float(rows[k][i]):.6g}" for k in CUES])
    json.dump({"threshold": thr, "quantile": args.quantile, "sigma_max": args.sigma_max, "geometries": list(geoms)},
              open(out.with_suffix(".json"), "w"))
    has = rows["has_hl"].bool()
    print(f"saved {out}; images with a highlight region: {int(has.sum())} / {len(has)}")
    for k in CUES[:9]:
        v = rows[k][has]
        v = v[~v.isnan()]
        print(f"  {k:14s} median={v.median():9.4f}  p10={v.quantile(0.1):9.4f}  p90={v.quantile(0.9):9.4f}")

    # overlay of the regions on example images of each geometry (two per glossiness level 1, 3, 5, 7)
    for g, idx in geoms.items():
        obj, bg = masks[g]
        picks = []
        for level in (0, 2, 4, 6):
            pool = [i for i in idx if round(ds.rows[i][1] * 6) == level]
            picks += rng.sample(pool, min(2, len(pool)))
        xs = torch.stack([ds[i][0] for i in picks]).cuda()
        with torch.no_grad():
            c, Hm, ring = cues_for_batch(xs, feat, obj, thr, bg)
        fig, axes = plt.subplots(1, len(picks), figsize=(2.3 * len(picks), 2.9))
        for col, i in enumerate(picks):
            ax = axes[col]
            ax.imshow(xs[col].cpu().permute(1, 2, 0))
            ax.contour(obj.cpu().numpy(), levels=[0.5], colors="yellow", linewidths=0.4)
            if Hm[col].any():
                ax.contour(Hm[col].cpu().numpy(), levels=[0.5], colors="red", linewidths=0.9)
            ax.contour(ring[col].cpu().numpy(), levels=[0.5], colors="cyan", linewidths=0.5)
            ax.set_title(f"{g}: gloss {ds.rows[i][1] * 6 + 1:.0f}\ncov {c['coverage'][col]:.3f}  "
                         f"con {c['contrast'][col]:.2f}", fontsize=7)
            ax.axis("off")
        fig.savefig(out.with_name(f"{out.stem}_overlay_{g}.png"), dpi=130, bbox_inches="tight")
        plt.close(fig)
    print("saved overlays", out.with_name(out.stem + "_overlay_*.png"))


if __name__ == "__main__":
    main()
