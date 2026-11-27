"""Where should a highlight go on a matte object? Shading-only placement against placement from surface normals.

For each Serrano geometry and light, the matte render is the input, and the specular (glossy) pass of the semi-glossy
and glossy renders of the same shape and light is the truth. Placement maps (all on object pixels):
  shading       the brightest part of the smoothed diffuse shading, (Y - t) / (max - t) above the 90th percentile, squared
  P             gloss map of Nagata et al. on the matte image
  normals       Blinn-Phong lobe (n.h)^k from the true normals (from the position pass), the view direction, and one
                light direction fitted to the shading by least squares (Lambert)
  normals+rim   the same plus a Fresnel term (1 - n.v)^5, which a dielectric shows towards the silhouette
Scores: Spearman correlation with the true specular luminance, and how much of the top 5% of each map lies in the
true top 5% (chance 0.05). Writes runs/gloss_edit/synthesis.txt and synthesis.png.
"""
import glob
from pathlib import Path

import matplotlib
import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scripts.gloss_edit import LUM, blur, linear_to_srgb
from src.models.features import GlossFeatures

SHAPES = ["sphere", "bunny_vt", "happy_vrip_vt", "xyzrgb_dragon_vt"]
EXPOSURE = 0.8
METHODS = ["shading", "P", "normals k=30", "normals k=300", "normals+rim k=30"]


def load(shape, material, light):
    d = np.load(glob.glob(f"data/serrano_render/{shape}__{material}__{light}*.npz")[0])
    rgb = torch.from_numpy(d["L_rgb"].astype(np.float32)).permute(2, 0, 1)
    mask = torch.from_numpy(d["L_mask"])
    y = (rgb * LUM[0]).sum(0)
    scale = EXPOSURE / torch.quantile(y[mask], 0.995).clamp_min(1e-6)
    cam = torch.from_numpy(np.linalg.inv(d["L_view"])[:3, 3].astype(np.float32))
    return dict(lin=rgb * scale, mask=mask, glossy=torch.from_numpy(d["L_glossy"].astype(np.float32)) * scale,
                pos=torch.from_numpy(d["L_position"].astype(np.float32)), cam=cam)


def normals(pos, mask, cam):
    """Unit normals from the world-position pass (central differences), facing the camera."""
    dx = pos[:, 2:] - pos[:, :-2]
    dy = pos[2:] - pos[:-2]
    n = torch.zeros_like(pos)
    n[1:-1, 1:-1] = torch.cross(dx[1:-1], dy[:, 1:-1], dim=-1)
    n = F.normalize(n, dim=-1)
    v = F.normalize(cam - pos, dim=-1)
    n = torch.where(((n * v).sum(-1, keepdim=True) < 0), -n, n)
    inner = -F.max_pool2d(-mask.float()[None, None], 3, 1, 1)[0, 0] > 0.5  # differences across the silhouette are wrong
    return n, v, inner


def placements(c, feat):
    y = (c["lin"].clamp(0, 1) * LUM[0]).sum(0)
    m = c["mask"]
    ys = blur(y[None, None], 2.0)[0, 0]
    t, top = torch.quantile(ys[m], 0.90), ys[m].max()
    out = {"shading": ((ys - t) / (top - t).clamp_min(1e-6)).clamp(0, 1) ** 2}
    x = torch.round(linear_to_srgb(c["lin"]) * 255) / 255
    out["P"] = feat.gloss_map(x[None].cuda())[0].cpu()
    n, v, inner = normals(c["pos"], m, c["cam"])
    sel = m & inner
    A = torch.cat([n[sel], torch.ones(int(sel.sum()), 1)], 1)
    sol = torch.linalg.lstsq(A, y[sel][:, None]).solution[:, 0]  # Y ~ n.L + b
    light = F.normalize(sol[:3], dim=0)
    h = F.normalize(light + v, dim=-1)
    nh = (n * h).sum(-1).clamp(0, 1)
    rim = (1 - (n * v).sum(-1).clamp(0, 1)) ** 5
    for k in (30, 300):
        out[f"normals k={k}"] = nh**k
    out["normals+rim k=30"] = nh**30 + rim
    return {k: v_ * sel for k, v_ in out.items()}, sel


def score(pred, true, sel):
    p, t = pred[sel].numpy(), true[sel].numpy()
    rho = spearmanr(p, t)[0] if p.std() > 0 else np.nan
    n = max(1, int(0.05 * len(p)))
    top_p, top_t = np.argsort(-p, kind="stable")[:n], set(np.argsort(-t, kind="stable")[:n].tolist())
    return rho, np.mean([i in top_t for i in top_p])


def main():
    out = Path("runs/gloss_edit")
    out.mkdir(parents=True, exist_ok=True)
    feat = GlossFeatures(32, sigma_step=1).cuda()
    lights = sorted({Path(f).stem.split("__")[2] for f in glob.glob("data/serrano_render/*.npz")})
    res, examples = {}, []
    for shape in SHAPES:
        for light in lights:
            c = load(shape, "matte_grey", light)
            maps, sel = placements(c, feat)
            for target in ("semi_blue", "glossy_blue"):
                t = load(shape, target, light)
                for k in METHODS:
                    res.setdefault((shape, target, k), []).append(score(maps[k], t["glossy"], sel))
            if light == lights[0]:
                examples.append((shape, c, maps, load(shape, "semi_blue", light)))
    with open(out / "synthesis.txt", "w") as f:
        for target, lab in (("semi_blue", "semi-glossy (roughness 0.25)"), ("glossy_blue", "glossy (roughness 0.05)")):
            for j, what in enumerate(("Spearman correlation with the true specular luminance",
                                      "share of the map's top 5% inside the true top 5% (chance 0.05)")):
                print(f"\n== {lab}: {what} ==", file=f)
                print(f"{'shape':20s}" + "".join(f"{k:>18s}" for k in METHODS), file=f)
                for shape in SHAPES:
                    print(f"{shape:20s}" + "".join(f"{np.nanmean([r[j] for r in res[(shape, target, k)]]):18.2f}"
                                                   for k in METHODS), file=f)
    print(open(out / "synthesis.txt").read())

    titles = ["matte input", "true specular (semi-glossy)", "placement: shading", "placement: normals+rim",
              "matte + shading highlight", "matte + normals highlight", "real semi-glossy render"]
    fig, ax = plt.subplots(len(examples), len(titles), figsize=(2.6 * len(titles), 2.7 * len(examples)))
    for r, (shape, c, maps, t) in enumerate(examples):
        gain = float(torch.quantile(t["glossy"][t["mask"]], 0.99))  # peak level of the true specular layer
        def add(m):
            return linear_to_srgb(c["lin"] + gain * (m / m.max().clamp_min(1e-6))[None]).permute(1, 2, 0).numpy()
        ims = [linear_to_srgb(c["lin"]).permute(1, 2, 0).numpy(), t["glossy"] * t["mask"], maps["shading"],
               maps["normals+rim k=30"], add(maps["shading"]), add(maps["normals+rim k=30"]),
               linear_to_srgb(t["lin"]).permute(1, 2, 0).numpy()]
        for k, im in enumerate(ims):
            if im.ndim == 2:
                ax[r, k].imshow(np.asarray(im), cmap="magma", vmin=0, vmax=float(np.quantile(np.asarray(im), 0.995)) + 1e-6)
            else:
                ax[r, k].imshow(im)
            ax[r, k].set_xticks([])
            ax[r, k].set_yticks([])
            if r == 0:
                ax[r, k].set_title(titles[k], fontsize=8)
        ax[r, 0].set_ylabel(shape[:9], fontsize=9)
    plt.tight_layout()
    plt.savefig(out / "synthesis.png", dpi=90)


if __name__ == "__main__":
    main()
