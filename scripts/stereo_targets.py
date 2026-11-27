"""Training targets for the monocular highlight network from the stereo scenes of stereo_dataset.py.

For every scene_XXXXX.npz: the right view is warped onto the left one with the rendered positions (exact
correspondence, occlusions removed), and two per-pixel targets are computed at render size, then pooled to --size:
  residual  |L - R_warped| / (L + R_warped) of blurred luminance: what two eyes reveal (b2, no physics label)
  spec      share of the left luminance that comes from the glossy pass (b1, renderer ground truth)
Also stored: the left image (linear RGB, float16; tone mapping is done at training time), the object mask and where the
residual is defined. Everything goes into one .npz (arrays stacked over scenes) that train_highlight.py reads.
"""
import argparse
import glob
import json
import os

import numpy as np
import torch
import torch.nn.functional as F

LUM = np.array([0.2126, 0.7152, 0.0722], np.float32)


def blur(x, sigma):
    r = max(1, int(3 * sigma))
    k = torch.exp(-torch.arange(-r, r + 1, dtype=torch.float32) ** 2 / (2 * sigma ** 2))
    k = (k / k.sum())
    x = F.conv2d(F.pad(x, (r, r, 0, 0), mode="replicate"), k.view(1, 1, 1, -1))
    return F.conv2d(F.pad(x, (0, 0, r, r), mode="replicate"), k.view(1, 1, -1, 1))


def targets(d, sigma, tol):
    f = lambda k: d[k].astype(np.float32)
    H, W = d["L_mask"].shape
    yL, yR = f("L_rgb") @ LUM, f("R_rgb") @ LUM
    X = np.concatenate([f("L_position"), np.ones((H, W, 1), np.float32)], -1).reshape(-1, 4)
    clip = X @ (d["R_proj"] @ d["R_view"]).T
    ndc = clip[:, :2] / clip[:, 3:4]
    grid = torch.from_numpy(np.stack([ndc[:, 0], -ndc[:, 1]], -1).reshape(1, H, W, 2)).float()
    t = lambda a: torch.from_numpy(np.ascontiguousarray(a)).float().permute(2, 0, 1)[None]
    yRw = F.grid_sample(t(yR[..., None]), grid, mode="bilinear", align_corners=False)[0, 0].numpy()
    pR = F.grid_sample(t(f("R_position")), grid, mode="nearest", align_corners=False)[0].permute(1, 2, 0).numpy()
    mR = F.grid_sample(t(d["R_mask"][..., None].astype(np.float32)), grid, mode="nearest",
                       align_corners=False)[0, 0].numpy() > 0.5
    ok = d["L_mask"] & mR & (np.linalg.norm(pR - f("L_position"), axis=-1) < tol)
    ok &= (np.abs(ndc) < 1).all(1).reshape(H, W)
    okt = torch.from_numpy(ok.astype(np.float32))[None, None]
    w = blur(okt, sigma).clamp_min(1e-6)
    a = blur(torch.from_numpy(np.where(ok, yL, 0))[None, None], sigma) / w
    b = blur(torch.from_numpy(np.where(ok, yRw, 0))[None, None], sigma) / w
    res = ((a - b).abs() / (a + b).clamp_min(1e-6))[0, 0].numpy()
    spec = f("L_glossy") / np.maximum(yL, 1e-6)
    return np.where(ok, res, 0), ok, np.clip(spec, 0, 1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="data/stereo_ds")
    p.add_argument("--out", default="data/stereo_ds_targets.npz")
    p.add_argument("--size", type=int, default=128)
    p.add_argument("--sigma", type=float, default=1.0, help="blur in render pixels before comparing the views")
    p.add_argument("--tol", type=float, default=0.003, help="max distance (m) between matched surface points")
    args = p.parse_args()

    files = sorted(glob.glob(os.path.join(args.root, "scene_*.npz")))
    out = {k: [] for k in ("rgb", "residual", "spec", "valid", "mask")}
    names, params = [], []
    for i, fp in enumerate(files):
        d = dict(np.load(fp))
        res, ok, spec = targets(d, args.sigma, args.tol)
        k = d["L_mask"].shape[0] // args.size
        pool = lambda a: F.avg_pool2d(torch.from_numpy(np.ascontiguousarray(a)).float()[None, None], k)[0, 0].numpy()
        rgb = F.avg_pool2d(torch.from_numpy(d["L_rgb"].astype(np.float32)).permute(2, 0, 1)[None], k)[0].numpy()
        valid = pool(ok.astype(np.float32))
        out["rgb"].append(rgb.astype(np.float16))
        out["residual"].append((pool(res) / np.maximum(valid, 1e-6)).astype(np.float16))
        out["spec"].append(pool(spec * d["L_mask"]).astype(np.float16) / np.maximum(pool(d["L_mask"].astype(np.float32)), 1e-6).astype(np.float16))
        out["valid"].append(valid > 0.99)
        out["mask"].append(pool(d["L_mask"].astype(np.float32)) > 0.5)
        names.append(os.path.basename(fp))
        params.append(str(d["params"]))
        if (i + 1) % 500 == 0:
            print(f"{i + 1}/{len(files)}", flush=True)
    np.savez(args.out, names=np.array(names), params=np.array(params), **{k: np.stack(v) for k, v in out.items()})
    r = np.stack(out["residual"]).astype(np.float32)
    v = np.stack(out["valid"])
    print(f"saved {args.out}: {len(names)} scenes; residual on valid pixels: median {np.median(r[v]):.4f}, "
          f"p99 {np.percentile(r[v], 99):.3f}; valid share of object {v.sum() / np.stack(out['mask']).sum():.3f}")


if __name__ == "__main__":
    main()
