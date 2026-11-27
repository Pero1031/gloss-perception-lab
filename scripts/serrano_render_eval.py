"""Compare the stereo-trained networks on our Blender renders of the Serrano geometries with the true targets and with
Serrano's own images of the same geometries (renders from serrano_shape_render.py).

Per render: true residual / specular share (stereo_targets.targets, exact correspondence) and the network maps on the
tone-mapped left image (exposure 1.0, 8-bit, 128 px, as in training validation), averaged over object pixels. For the
Serrano images of the same geometry (rating 1 = matte, >= 5.5 = glossy) the same network means are given for reference.
Writes runs/serrano_render/summary.txt and maps.png.
"""
import glob
import json
import os

import matplotlib
import numpy as np
import torch
import torch.nn.functional as F

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scripts.shape_masks import geometry_masks, geometry_of
from scripts.stereo_targets import targets
from scripts.train_highlight import GREY, LUM, tone_map
from src.data.datasets import ManifestDataset
from src.models.unet import UNet

OUT = "runs/serrano_render"


def blur(x, sigma):
    r = int(3 * sigma)
    k = torch.exp(-torch.arange(-r, r + 1, dtype=torch.float32) ** 2 / (2 * sigma ** 2))
    k = (k / k.sum()).to(x)
    c = x.shape[1]
    x = F.conv2d(F.pad(x, (r, r, 0, 0), mode="replicate"), k.view(1, 1, 1, -1).repeat(c, 1, 1, 1), groups=c)
    return F.conv2d(F.pad(x, (0, 0, r, r), mode="replicate"), k.view(1, 1, -1, 1).repeat(c, 1, 1, 1), groups=c)


GREY_BG = False  # set from the checkpoints: True when they were trained with --bg grey


def nets():
    global GREY_BG
    out = {}
    for t in ("residual", "spec"):
        ck = torch.load(f"runs/highlight_{t}/last.pt")
        a = ck["args"]
        GREY_BG = a.get("bg", "scene") == "grey"
        m = UNet(3, 1, a["base"], a["depth"], head="map").cuda().eval()
        m.load_state_dict(ck["model"])
        out[t] = m
    return out


def predict(models, x):
    with torch.no_grad():
        return {t: (torch.sigmoid(m(x.cuda()))[:, 0] ** 2).cpu() for t, m in models.items()}


def main():
    os.makedirs(OUT, exist_ok=True)
    models = nets()
    rows = []
    vis = {}
    for fp in sorted(glob.glob("data/serrano_render/*.npz")):
        d = dict(np.load(fp))
        p = json.loads(str(d["params"]))
        res, ok, spec = targets(d, 1.0, 0.003)
        pool = lambda a: F.avg_pool2d(torch.from_numpy(np.ascontiguousarray(a, np.float32))[None, None], 2)[0, 0]
        okp = pool(ok.astype(np.float32))
        res_p = (pool(res) / okp.clamp_min(1e-6)).numpy()
        mask = pool(d["L_mask"].astype(np.float32)).numpy() > 0.5
        valid = okp.numpy() > 0.99
        spec_p = pool(spec * d["L_mask"]).numpy()
        rgb = F.avg_pool2d(torch.from_numpy(d["L_rgb"].astype(np.float32)).permute(2, 0, 1)[None], 2)
        x = tone_map(rgb, torch.from_numpy(mask)[None], fixed=1.0)
        if GREY_BG:
            x = torch.where(torch.from_numpy(mask)[None, None], x, torch.full_like(x, GREY))
        pr = predict(models, x)
        rows.append(dict(shape=p["shape"], material=p["material"], light=p["light"],
                         true_res=float(res_p[valid].mean()), pred_res=float(pr["residual"][0][torch.from_numpy(mask)].mean()),
                         true_spec=float(spec_p[mask].mean()), pred_spec=float(pr["spec"][0][torch.from_numpy(mask)].mean())))
        if p["light"].startswith("1_1"):
            vis[(p["shape"], p["material"])] = (x[0].permute(1, 2, 0).numpy(), pr["residual"][0].numpy(),
                                                np.where(valid, res_p, 0))

    # Serrano images of the same geometries
    ds = ManifestDataset("data/serrano/manifest_all9.csv", "data/serrano/images", size=128)
    paths = [r[0] for r in ds.rows]
    y = np.array([r[1] for r in ds.rows]) * 6 + 1
    geo = np.array([geometry_of(pth) for pth in paths])
    ds256 = ManifestDataset("data/serrano/manifest_all9.csv", "data/serrano/images", size=256)
    rng = np.random.default_rng(0)
    ser, ser_vis = {}, {}
    variants = ("original", "re-exposed", "bg grey", "bg blurred")
    for g in sorted({r["shape"] for r in rows}):
        obj_e, _, _, sil, _ = geometry_masks(ds256, np.flatnonzero(geo == g))
        obj = F.avg_pool2d(obj_e.float()[None, None], 2)[0, 0] > 0.99  # eroded: where the map is averaged
        soft = F.avg_pool2d(torch.from_numpy(sil).float()[None, None], 2)[0, 0]  # silhouette: for compositing
        hard = soft > 0.5
        for kind, sel in (("matte", y <= 1), ("glossy", y >= 5.5)):
            idx = rng.choice(np.flatnonzero((geo == g) & sel), 200, replace=False)
            ms = {(v, t): [] for v in variants for t in ("residual", "spec")}
            for s in range(0, 200, 50):
                x0 = torch.stack([ds[i][0] for i in idx[s:s + 50]])
                if GREY_BG:  # as in training: background replaced by grey
                    x0 = torch.where(hard, x0, torch.full_like(x0, GREY))
                for v in variants:  # replace the background (object mask from shape_masks.py)
                    if v == "original":
                        x = x0
                    elif v == "re-exposed":  # object 99.5th luminance percentile -> 1.0, as at validation
                        lin = x0.clamp(0, 1) ** 2.2
                        q = torch.stack([torch.quantile(Yi[obj], 0.995) for Yi in (lin * LUM).sum(1)])
                        x = torch.round((lin / q.clamp_min(1e-4).view(-1, 1, 1, 1)).clamp(0, 1) ** (1 / 2.2) * 255) / 255
                        if GREY_BG:
                            x = torch.where(hard, x, torch.full_like(x, GREY))
                    else:
                        bg = torch.full_like(x0, 0.5) if v == "bg grey" else blur(x0, 4.0)
                        x = x0 * soft + bg * (1 - soft)
                    pr = predict(models, x)
                    for t in ("residual", "spec"):
                        ms[(v, t)].append(pr[t][:, obj].mean(1))
                    if kind == "matte" and s == 0:
                        ser_vis[(g, v)] = (x[0].permute(1, 2, 0).clamp(0, 1).numpy(), pr["residual"][0].numpy())
            ser[(g, kind)] = {k: float(torch.cat(v).mean()) for k, v in ms.items()}

    lines = ["Network maps on our renders of the Serrano geometries vs true targets (object-pixel means over 3 lights)",
             f"{'shape':18s}{'material':13s}{'true res':>9s}{'pred res':>9s}{'true spec':>10s}{'pred spec':>10s}"]
    for g in sorted({r["shape"] for r in rows}):
        for m in ("matte_grey", "matte_dark", "semi_blue", "glossy_blue"):
            rs = [r for r in rows if r["shape"] == g and r["material"] == m]
            if rs:
                mean = lambda k: np.mean([r[k] for r in rs])
                lines.append(f"{g:18s}{m:13s}{mean('true_res'):9.4f}{mean('pred_res'):9.4f}"
                             f"{mean('true_spec'):10.3f}{mean('pred_spec'):10.3f}")
        for kind in ("matte", "glossy"):
            if (g, kind) in ser:
                v = ser[(g, kind)]
                for var in variants:
                    lines.append(f"{g:18s}{'Serrano ' + kind:13s}{'':9s}{v[(var, 'residual')]:9.4f}{'':10s}"
                                 f"{v[(var, 'spec')]:10.3f}   ({var})")
        lines.append("")
    open(os.path.join(OUT, "summary.txt"), "w", encoding="utf-8").write("\n".join(lines))
    print("\n".join(lines))

    shapes = sorted({k[0] for k in vis})
    cols = [("matte_grey", "our render, matte grey"), ("matte_dark", "our render, matte dark"), ("glossy_blue", "our render, glossy")]
    fig, ax = plt.subplots(len(shapes), 13, figsize=(22, 1.9 * len(shapes) + 0.5))
    for r, g in enumerate(shapes):
        c = 0
        for m, title in cols:
            img, pm, tr = vis[(g, m)]
            for a, im, tt, kw in ((ax[r, c], img, title, {}),
                                  (ax[r, c + 1], pm, "b2 prediction", dict(cmap="magma", vmin=0, vmax=0.3)),
                                  (ax[r, c + 2], tr, "b2 target", dict(cmap="magma", vmin=0, vmax=0.3))):
                a.imshow(im, **kw)
                a.set_title(tt, fontsize=7)
            c += 3
        for k, (v, title) in enumerate((("original", "Serrano matte"), ("bg grey", "grey background"))):
            img, pm = ser_vis[(g, v)]
            ax[r, 9 + 2 * k].imshow(img)
            ax[r, 9 + 2 * k].set_title(title, fontsize=7)
            ax[r, 10 + 2 * k].imshow(pm, cmap="magma", vmin=0, vmax=0.3)
            ax[r, 10 + 2 * k].set_title("b2 prediction", fontsize=7)
        ax[r, 0].set_ylabel(g[:14], fontsize=8)
    for a in ax.flat:
        a.set_xticks([])
        a.set_yticks([])
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "maps.png"), dpi=80)


if __name__ == "__main__":
    main()
