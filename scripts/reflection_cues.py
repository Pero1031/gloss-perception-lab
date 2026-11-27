"""Reflection cues: how much the sphere shows the structure of its surroundings (not only isolated highlights).

A smooth surface mirrors the environment; humans read such reflections as low roughness and rate gloss high even when
there is no distinct highlight. The sphere interior is compared with the visible background and described by:
  spec_*   structure (rectified LoG response) of the sphere interior per band of scales, relative to its lightness
  energy_ratio, shape_dist, lum_dist   how similar the interior is to the background (structure energy, spectrum shape,
                                       luminance distribution)
  struct_frac  fraction of the interior outside the highlight region with fine-scale structure above the background median
  ring_*   radial profile (centre to limb): luminance gain towards the limb (Fresnel) and structure per ring
"""
import argparse
import csv
import json
from pathlib import Path

import torch

from scripts.highlight_cues import dilate, mmean, srgb_to_linear
from scripts.shape_masks import geometry_masks, geometry_of
from src.data.datasets import ManifestDataset
from src.models.features import GlossFeatures

BANDS = [(1, 2), (3, 4), (5, 8), (9, 16), (17, 32)]
RINGS = [(0.0, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 0.9)]
SPEC = [f"spec_{a}_{b}" for a, b in BANDS]
SIM = ["energy_ratio", "shape_dist", "lum_dist"]
STRUCT = ["struct_frac"]
RADIAL = [f"ring_lum_{k}" for k in (2, 3, 4)] + [f"ring_str_{k}" for k in (1, 2, 3, 4)]
REFL = SPEC + SIM + STRUCT + RADIAL
BG_REFL = ["energy_ratio", "shape_dist", "lum_dist", "struct_frac"]  # need a visible background
LUM_LEVELS = torch.linspace(0.05, 0.95, 16)


def reflection_cues(x, feat, rho, obj, bg, thr):
    lin = srgb_to_linear(x)
    Y = 0.2126 * lin[:, 0] + 0.7152 * lin[:, 1] + 0.0722 * lin[:, 2]
    p = feat(x)[:, 1:]
    sig = torch.tensor(feat.sigmas, device=x.device)
    P = feat.gloss_map_from(p)
    ob = obj[None].expand_as(Y)
    Hm = (P > thr) & ob

    def spectrum(mask):  # mean response per scale over the pixels of a (H, W) mask -> (B, n_scales)
        return p[:, :, mask].mean(2)

    s_obj = spectrum(obj)
    mean_y = mmean(Y, ob)
    out = {}
    for (a, b), name in zip(BANDS, SPEC):
        sel = (sig >= a) & (sig <= b)
        out[name] = s_obj[:, sel].mean(1) / (mean_y + 1e-3)
    fine = p[:, 1:6].mean(1)  # scales 2..6
    if bg.any():
        s_bg = spectrum(bg)
        out["energy_ratio"] = torch.log10((s_obj.sum(1) + 1e-9) / (s_bg.sum(1) + 1e-9))
        lo, lb = torch.log(s_obj + 1e-9), torch.log(s_bg + 1e-9)
        out["shape_dist"] = ((lo - lo.mean(1, keepdim=True)) - (lb - lb.mean(1, keepdim=True))).pow(2).mean(1).sqrt()
        q = LUM_LEVELS.to(x.device)
        qo = torch.log10(torch.quantile(Y[:, obj], q, dim=1) + 1e-3)
        qb = torch.log10(torch.quantile(Y[:, bg], q, dim=1) + 1e-3)
        out["lum_dist"] = (qo - qb).abs().mean(0)
        bg_med = fine[:, bg].median(1).values
        area = ob & ~dilate(Hm, 4)
        out["struct_frac"] = mmean((fine > bg_med[:, None, None]).float(), area)
    else:  # no visible background (object fills the frame): the object-vs-background cues are undefined
        for k in BG_REFL:
            out[k] = torch.full_like(mean_y, float("nan"))

    ring_y, ring_f = [], []
    for lo_, hi_ in RINGS:
        m = ((rho >= lo_) & (rho < hi_) & obj)[None].expand_as(Y)
        ring_y.append(mmean(Y, m))
        ring_f.append(mmean(fine, m))
    for k in (2, 3, 4):
        out[f"ring_lum_{k}"] = torch.log10((ring_y[k - 1] + 1e-3) / (ring_y[0] + 1e-3))
    for k in (1, 2, 3, 4):
        out[f"ring_str_{k}"] = ring_f[k - 1] / (mean_y + 1e-3)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", default="data/serrano/manifest_sphere.csv")
    p.add_argument("--root", default="data/serrano/images")
    p.add_argument("--meta", default="runs/highlight_cues/cues_q95.json", help="JSON with the P threshold of the cues run")
    p.add_argument("--out", default="runs/highlight_cues/reflection_q95.csv")
    args = p.parse_args()
    thr = json.load(open(args.meta))["threshold"]

    ds = ManifestDataset(args.manifest, args.root, size=256, cache=True)
    geoms = {}
    for i, (path, _) in enumerate(ds.rows):
        geoms.setdefault(geometry_of(path), []).append(i)
    feat = GlossFeatures(32, sigma_step=1).cuda()

    rows = {k: torch.zeros(len(ds)) for k in REFL}
    with torch.no_grad():
        for g, idx in geoms.items():
            obj_g, bg_g, rho_g, _, _ = geometry_masks(ds, idx)
            obj, bg, rho = obj_g.cuda(), bg_g.cuda(), rho_g.cuda()
            for s in range(0, len(idx), 8):
                sel = idx[s:s + 8]
                x = torch.stack([ds[i][0] for i in sel]).cuda()
                c = reflection_cues(x, feat, rho, obj, bg, thr)
                for k in REFL:
                    rows[k][sel] = c[k].cpu()
            print(f"{g}: done", flush=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["path"] + REFL)
        for i, (path, _) in enumerate(ds.rows):
            w.writerow([path] + [f"{float(rows[k][i]):.6g}" for k in REFL])
    print("saved", args.out)
    for k in REFL:
        v = rows[k]
        print(f"  {k:14s} median={v.median():9.4f}  p10={v.quantile(0.1):9.4f}  p90={v.quantile(0.9):9.4f}  nan={int(v.isnan().sum())}")


if __name__ == "__main__":
    main()
