"""Binocular highlight pilot, step 1: the same left-right residual, but with correspondence estimated by stereo matching.

stereo_residual.py warps the right view with the renderer's world positions (exact correspondence). An observer only has
the two images, so here the pair is rectified from the camera matrices and the disparity is estimated with OpenCV's
semi-global block matching (SGBM) on 8-bit tone-mapped luminance. Reported per scene, on object pixels of the rectified
left view:
  - residual |L - R_warped| / (L + R_warped) with the estimated and with the exact disparity: share of pixels above T on
    matte materials (false alarms) and recall / AUC for specular pixels (rendered glossy pass) on semi / glossy
  - disparity error against the exact disparity, for specular and non-specular pixels
  - left-right inconsistency (disparity from the left and from the right image disagree by > 1 px), which is expected to
    concentrate on reflections ("proto-rivalry", Muryy et al. 2016)

Needs OpenCV (run with a Python that has cv2, e.g. a system Python with opencv-python):
  python scripts/stereo_matching.py --root runs/stereo_pilot
"""
import argparse
import glob
import json
import os

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.rcParams.update({"font.family": "sans-serif", "font.size": 8})
LUM = np.array([0.2126, 0.7152, 0.0722], np.float32)
FLIP = np.diag([1.0, -1.0, -1.0])  # Blender camera (y up, looking down -z) -> OpenCV camera (y down, looking down +z)


def intrinsics(v, size):
    f = v["proj"][0, 0] * size / 2
    c = (size - 1) / 2  # OpenCV puts pixel centres at integers
    return np.array([[f, 0, c], [0, f, c], [0, 0, 1]])


def extrinsics(v):
    Rt = FLIP @ v["view"][:3, :4].astype(np.float64)
    return Rt[:, :3], Rt[:, 3]


def rectify(L, R, size):
    K = intrinsics(L, size)
    RL, tL = extrinsics(L)
    RR, tR = extrinsics(R)
    Rrel = RR @ RL.T
    T = tR - Rrel @ tL
    R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(K, np.zeros(5), K, np.zeros(5), (size, size), Rrel, T.reshape(3, 1), alpha=0)
    maps = [cv2.initUndistortRectifyMap(K, np.zeros(5), Ri, Pi, (size, size), cv2.CV_32FC1) for Ri, Pi in ((R1, P1), (R2, P2))]
    warp = lambda img, m, interp=cv2.INTER_LINEAR: cv2.remap(img, m[0], m[1], interp, borderValue=0)
    return warp, maps, (RL, tL, R1, P1), (RR, tR, R2, P2)


def project(X, R, t, Rrect, Prect):
    """World points -> pixel coordinates in a rectified view."""
    Xc = (X @ R.T + t) @ Rrect.T
    x = Xc @ Prect[:, :3].T + Prect[:, 3]
    return x[..., :2] / x[..., 2:3]


def residual(a, b, ok, sigma):
    k = (0, 0)
    w = cv2.GaussianBlur(ok.astype(np.float32), k, sigma)
    a = cv2.GaussianBlur(np.where(ok, a, 0).astype(np.float32), k, sigma) / np.maximum(w, 1e-6)
    b = cv2.GaussianBlur(np.where(ok, b, 0).astype(np.float32), k, sigma) / np.maximum(w, 1e-6)
    return np.abs(a - b) / np.maximum(a + b, 1e-6)


def auc(pos, neg):
    if len(pos) < 20 or len(neg) < 20:
        return np.nan
    r = np.argsort(np.argsort(np.concatenate([pos, neg])))
    return float((r[:len(pos)].sum() - len(pos) * (len(pos) - 1) / 2) / (len(pos) * len(neg)))


def sgbm(a8, b8, dmin, ndisp, block):
    m = cv2.StereoSGBM_create(minDisparity=dmin, numDisparities=ndisp, blockSize=block, P1=8 * block ** 2,
                              P2=32 * block ** 2, uniquenessRatio=5, speckleWindowSize=0, disp12MaxDiff=-1,
                              mode=cv2.STEREO_SGBM_MODE_HH)
    return m.compute(a8, b8).astype(np.float32) / 16


def analyse(d, cam, args):
    L, R = (dict(np.load(os.path.join(d, c + ".npz"))) for c in ("L", cam))
    size = L["mask"].shape[0]
    warp, maps, camL, camR = rectify(L, R, size)
    yL, yR = warp(L["rgb"] @ LUM, maps[0]), warp(R["rgb"] @ LUM, maps[1])
    mL = warp(L["mask"].astype(np.uint8), maps[0], cv2.INTER_NEAREST) > 0
    mR = warp(R["mask"].astype(np.uint8), maps[1], cv2.INTER_NEAREST) > 0
    posL = np.dstack([warp(L["position"][..., i], maps[0], cv2.INTER_NEAREST) for i in range(3)])
    posR = np.dstack([warp(R["position"][..., i], maps[1], cv2.INTER_NEAREST) for i in range(3)])
    spec = warp(L["glossy"] @ LUM, maps[0]) / np.maximum(yL, 1e-6)

    # exact disparity: where the left pixel's surface point lands in the rectified right view (occlusion-checked)
    xs = np.arange(size, dtype=np.float32)[None].repeat(size, 0)
    uv = project(posL, *camL[:3], camR[3])  # P2 maps points given in the rectified left camera frame
    d_gt = xs - uv[..., 0]
    ui, vi = np.round(uv[..., 0]).astype(int), np.round(uv[..., 1]).astype(int)
    inside = (ui >= 0) & (ui < size) & (vi >= 0) & (vi < size)
    vis = np.zeros_like(mL)
    vis[inside] = np.linalg.norm(posR[vi[inside], ui[inside]] - posL[inside], axis=-1) < 0.003
    vis &= mL & mR[np.clip(vi, 0, size - 1), np.clip(ui, 0, size - 1)]

    # 8-bit tone map with one exposure for both eyes (from the left object pixels)
    s = 0.95 / max(np.percentile(yL[mL], 99.5), 1e-6)
    to8 = lambda y: (np.clip(y * s, 0, 1) ** (1 / 2.2) * 255).astype(np.uint8)
    a8, b8 = to8(yL), to8(yR)
    # search range: the vergence offset of the toe-in cameras (~90 px) is removed by shifting the right image, so that
    # SGBM's invalid left band is only as wide as the residual range (an observer knows its vergence; the range itself
    # is taken from the exact disparities, +-16 px)
    lo, hi = np.percentile(d_gt[vis], [0.5, 99.5])
    shift = int(np.floor(lo)) - 16
    ndisp = int(np.ceil((hi - shift + 16) / 16) * 16)
    b8s = np.zeros_like(b8)
    b8s[:, shift:] = b8[:, :size - shift] if shift >= 0 else b8[:, -shift:]
    dL = sgbm(a8, b8s, 0, ndisp, args.block)
    # mirrored pair: the mirrored right image becomes the left one, with the same disparity sign and range;
    # mirroring back gives the disparity at (shifted) right-image pixels
    dR = np.fliplr(sgbm(np.fliplr(b8s).copy(), np.fliplr(a8).copy(), 0, ndisp, args.block))
    valid = dL >= 0
    xr = np.clip(np.round(xs - dL).astype(int), 0, size - 1)
    lr_bad = valid & (np.abs(dL - dR[np.arange(size)[:, None], xr]) > 1)
    dL = dL + shift

    sample = lambda disp: cv2.remap(yR, (xs - disp).astype(np.float32), np.arange(size, dtype=np.float32)[:, None].repeat(size, 1),
                                    cv2.INTER_LINEAR, borderValue=0)
    obj = mL & vis  # pixels where the exact correspondence exists
    res_gt = residual(yL, sample(d_gt), obj, args.sigma)
    ok_est = mL & valid
    res_est = residual(yL, sample(np.where(valid, dL, 0)), ok_est, args.sigma)

    sp, ns = obj & (spec > 0.5), obj & (spec < 0.1)
    out = {}
    for tag, res, ok in (("gt", res_gt, obj), ("est", res_est, obj & valid)):
        out[f"flag_{tag}"] = float(((res > args.T) & ok).sum() / ok.sum())
        out[f"rec_{tag}"] = float((res[sp & ok] > args.T).mean()) if (sp & ok).sum() > 20 else np.nan
        out[f"auc_{tag}"] = auc(res[sp & ok], res[ns & ok])
    err = np.abs(dL - d_gt)
    out["disp_err_nonspec"] = float(np.median(err[ns & valid])) if (ns & valid).any() else np.nan
    out["disp_err_spec"] = float(np.median(err[sp & valid])) if (sp & valid).sum() > 20 else np.nan
    out["valid"] = float((valid & obj).sum() / obj.sum())
    out["lr_bad_nonspec"] = float(lr_bad[ns].mean()) if ns.any() else np.nan
    out["lr_bad_spec"] = float(lr_bad[sp].mean()) if sp.sum() > 20 else np.nan
    out["lr_auc"] = auc(lr_bad[sp].astype(float) + 0, lr_bad[ns].astype(float) + 0)
    vis_maps = dict(img=a8, res_est=np.where(ok_est, res_est, np.nan), res_gt=np.where(obj, res_gt, np.nan),
                    lr=np.where(mL, lr_bad.astype(float), np.nan), spec=np.where(mL, spec, np.nan))
    return out, vis_maps


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="runs/stereo_pilot")
    p.add_argument("--sigma", type=float, default=1.5)
    p.add_argument("--T", type=float, default=0.02)
    p.add_argument("--block", type=int, default=5)
    args = p.parse_args()

    scenes = sorted(d for d in glob.glob(os.path.join(args.root, "*")) if os.path.exists(os.path.join(d, "R.npz")))
    rows, maps = [], {}
    for d in scenes:
        name = os.path.basename(d)
        for cam in ("R", "R2"):
            if not os.path.exists(os.path.join(d, cam + ".npz")):
                continue
            r, m = analyse(d, cam, args)
            rows.append(dict(scene=name, cam=cam, **r))
            if cam == "R":
                maps[name] = m
    keys = ["flag_gt", "flag_est", "rec_gt", "rec_est", "auc_gt", "auc_est", "disp_err_nonspec", "disp_err_spec",
            "valid", "lr_bad_nonspec", "lr_bad_spec"]
    print(f"{'scene':30s}{'cam':>4s}" + "".join(f"{k[:11]:>12s}" for k in keys))
    for r in rows:
        print(f"{r['scene']:30s}{r['cam']:>4s}" + "".join(f"{r[k]:12.3f}" for k in keys))
    json.dump(rows, open(os.path.join(args.root, "matching_summary.json"), "w"), indent=1)

    world = "courtyard"
    shapes = [s for s in ["sphere", "blob", "blob_fine", "erato"] if any(k.startswith(s + "_") and k.count("_") == s.count("_") + 2 for k in maps)]
    cols = [("matte", "res_est"), ("matte", "lr"), ("glossy", "img"), ("glossy", "res_est"), ("glossy", "lr"),
            ("glossy", "spec")]
    titles = {"img": "left image", "res_est": "residual (estimated correspondence)", "lr": "left-right inconsistency (white)", "spec": "true specular fraction"}
    fig, axs = plt.subplots(len(shapes), len(cols), figsize=(2.2 * len(cols), 2.3 * len(shapes)), squeeze=False,
                            facecolor="#fcfcfb", constrained_layout=True)
    for i, s in enumerate(shapes):
        for j, (mat, k) in enumerate(cols):
            ax, mp = axs[i, j], maps.get(f"{s}_{mat}_{world}")
            ax.set_xticks([]), ax.set_yticks([])
            if mp is None:
                continue
            ax.set_facecolor("#d9d8d4")
            if k == "img":
                ax.imshow(mp["img"], cmap="gray", vmin=0, vmax=255)
            else:
                ax.imshow(mp[k], cmap="gray" if k == "lr" else "magma", vmin=0,
                          vmax={"res_est": 0.5, "lr": 1, "spec": 1}[k])
            if i == 0:
                ax.set_title(f"{mat}: {titles[k]}", fontsize=7.5)
            if j == 0:
                ax.set_ylabel(s, fontsize=9)
    fig.suptitle(f"Correspondence from stereo matching (SGBM). Light {world}, baseline 6.5 cm, rectified left-eye coordinates",
                 x=0.01, ha="left", fontsize=9)
    fig.savefig(os.path.join(args.root, f"matching_maps_{world}.png"), dpi=150, bbox_inches="tight")
    print("saved", os.path.join(args.root, f"matching_maps_{world}.png"))


if __name__ == "__main__":
    main()
