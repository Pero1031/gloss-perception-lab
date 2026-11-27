"""Gloss suppression with the intrinsic decomposition of Careaga & Aksoy (see intrinsic_model.py).

The network splits an image into albedo x diffuse shading and a residual; the positive residual holds specular
reflections. Suppressed image = albedo x diffuse shading. Amount of gloss = positive residual / luminance over the
object.
  renders:  python -m scripts.intrinsic_eval                 -> runs/gloss_edit/intrinsic.txt, intrinsic.png
            (sets A and B of specular_separation.py, same error: RMSE x100 against the true diffuse image)
  photos:   python -m scripts.intrinsic_eval photo1.jpg ...  -> runs/photo/intrinsic_<name>.jpg (input | suppressed |
            positive residual)
"""
import glob
import sys
from pathlib import Path

import matplotlib
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageOps

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scripts.gloss_edit import LUM, linear_to_srgb, srgb_to_linear
from scripts.highlight_synthesis import geometry
from scripts.intrinsic_model import decompose, load
from scripts.specular_separation import auc, rmse
from scripts.specular_separation import load as load_render

LUMV = LUM.view(3)


def run(models, lin):
    """lin: (3,H,W) tensor, linear. Returns (diffuse image, positive residual), both (3,H,W) at the input size."""
    h, w = lin.shape[1:]
    res = decompose(models, lin.clamp(0, 1).permute(1, 2, 0).numpy().astype(np.float32))
    back = lambda a: F.interpolate(torch.from_numpy(np.ascontiguousarray(a)).float().permute(2, 0, 1)[None], size=(h, w),
                                   mode="bilinear", align_corners=False, antialias=True)[0]
    return back(res["dif_img"]).clamp(0, 1), back(res["pos_res"]).clamp_min(0)


def renders(models):
    out_dir = Path("runs/gloss_edit")
    rng = np.random.default_rng(0)
    held = sorted(glob.glob("data/stereo_ds/scene_*.npz"))[-1000:]
    sets = {"A: Serrano geometries": sorted(glob.glob("data/serrano_render/*.npz")),
            "B: random scenes": [held[i] for i in rng.choice(len(held), 150, replace=False)]}
    examples = []
    with open(out_dir / "intrinsic.txt", "w") as f:
        for name, paths in sets.items():
            err, share = {}, []
            for q in paths:
                c = load_render(q)
                sel = geometry(c)[2]
                dif, pos = run(models, c["lin"])
                y = (c["lin"].clamp(0, 1) * LUM[0]).sum(0)
                true_diffuse = linear_to_srgb(c["lin"] - c["glossy"][None])
                g = ("matte " if c["matte"] else "glossy ") + c["shape"]
                e = {"none": rmse(linear_to_srgb(c["lin"]), true_diffuse, sel),
                     "diffuse image": rmse(linear_to_srgb(dif), true_diffuse, sel),
                     "image - residual": rmse(linear_to_srgb(c["lin"] - pos), true_diffuse, sel)}
                for k, v in e.items():
                    err.setdefault(g, {}).setdefault(k, []).append(v)
                share.append((c["matte"], float((pos * LUM[0]).sum(0)[sel].sum() / y[sel].sum())))
                if name.startswith("A") and "cambridge" in c["name"] and ("glossy_blue" in c["name"] or "matte_dark" in c["name"]):
                    examples.append((c, dif, pos, sel))
            cols = ["none", "diffuse image", "image - residual"]
            print(f"\n== {name}: suppression error against the true diffuse image (RMSE x100) ==", file=f)
            print(f"{'group':26s}" + "".join(f"{k:>18s}" for k in cols), file=f)
            for g in sorted(err):
                print(f"{g:26s}" + "".join(f"{np.mean(err[g][k]):18.2f}" for k in cols), file=f)
            m = np.array([v for is_m, v in share if is_m])
            gl = np.array([v for is_m, v in share if not is_m])
            print(f"positive residual share of the object's light: matte median {np.median(m):.3f}, non-matte "
                  f"{np.median(gl):.3f}, AUC {auc(gl, m):.3f}", file=f)
    print(open(out_dir / "intrinsic.txt").read())
    titles = ["input", "positive residual", "true specular", "diffuse image (suppressed)", "true diffuse"]
    fig, ax = plt.subplots(len(examples), len(titles), figsize=(2.6 * len(titles), 2.7 * len(examples)))
    for r, (c, dif, pos, sel) in enumerate(examples):
        img = lambda lin: linear_to_srgb(lin).permute(1, 2, 0).numpy()
        vmax = float(torch.quantile(c["glossy"][sel], 0.99)) + 0.05
        ims = [img(c["lin"]), (pos * LUM[0]).sum(0) * sel, c["glossy"] * sel, img(dif), img(c["lin"] - c["glossy"][None])]
        for k, im in enumerate(ims):
            ax[r, k].imshow(np.asarray(im), **(dict(cmap="magma", vmin=0, vmax=vmax) if im.ndim == 2 else {}))
            ax[r, k].set_xticks([])
            ax[r, k].set_yticks([])
            if r == 0:
                ax[r, k].set_title(titles[k], fontsize=8)
        ax[r, 0].set_ylabel(f"{c['shape'][:9]}\n{'matte' if c['matte'] else 'glossy'}", fontsize=8)
    plt.tight_layout()
    plt.savefig(out_dir / "intrinsic.png", dpi=75)


def photos(models, paths, size=1024):
    for path in paths:
        img = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
        k = size / max(img.size)
        img = img.resize((round(img.width * k), round(img.height * k)), Image.LANCZOS)
        x = torch.from_numpy(np.asarray(img).copy()).float().permute(2, 0, 1) / 255
        lin = srgb_to_linear(x)
        dif, pos = run(models, lin)
        y = (lin * LUM[0]).sum(0)
        print(f"{Path(path).name}: positive residual is {float((pos * LUM[0]).sum(0).sum() / y.sum()):.3f} of the image's light")
        to8 = lambda t: (linear_to_srgb(t).permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)
        sheet = np.concatenate([to8(lin), to8(dif), to8(pos * 2)], 1)
        out = Path("runs/photo") / f"intrinsic_{Path(path).stem}.jpg"
        Image.fromarray(sheet).save(out, quality=90)
        Image.fromarray(to8(dif)).save(out.with_name(f"intrinsic_{Path(path).stem}_suppressed.png"))


if __name__ == "__main__":
    net = load()
    photos(net, sys.argv[1:]) if len(sys.argv) > 1 else renders(net)
