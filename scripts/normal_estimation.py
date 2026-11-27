"""Surface normals from a single image of a matte object, scored for gloss synthesis.

Test images: dark matte renders of the Serrano geometries at 512 px (serrano_shape_render.py --size 512
--out data/serrano_render_512), whose true normals come from the position pass. Estimators:
  inflation        normals from the silhouette only: the outward direction on the contour, diffused into the interior
  inflation+shade  the same, then each normal is moved along the fitted light until it reproduces the shading
  MoGe-2 (B / L)   Wang et al., feed-forward ViT (code in external/MoGe, weights from the Hugging Face hub)
  MoGe-2 L+shade   MoGe-2 L followed by the same shading step
  Marigold         Marigold-Normals v1-1, Stable Diffusion based, 1 and 4 steps (diffusers)
Scores on object pixels: angular error of the normals and of their smoothed version (overall shape), agreement of the
fine relief, error of the light direction fitted to the shading, and agreement of the specular layer computed from
estimated normals with the layer from true normals (highlight_synthesis.physical_layer, each with its own light fit).
Writes runs/normals/summary.txt, normals.png and synthesis.png.
"""
import glob
import sys
import time
from pathlib import Path

import matplotlib
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scripts.gloss_edit import LUM, blur, linear_to_srgb, masked_blur
from scripts.highlight_synthesis import SHAPES, geometry, light_direction, physical_layer

ROOT = "data/serrano_render_512"
FLIPS = [(1, 1, 1), (1, -1, 1), (1, -1, -1), (1, 1, -1), (-1, 1, 1), (-1, -1, 1), (-1, -1, -1), (-1, 1, -1)]


def load(path):
    d = np.load(path)
    rgb = torch.from_numpy(d["L_rgb"].astype(np.float32)).permute(2, 0, 1)
    mask = torch.from_numpy(d["L_mask"])
    y = (rgb * LUM[0]).sum(0)
    q = torch.quantile(y[mask], 0.995).clamp_min(1e-6)
    c = dict(lin=rgb * 0.5 / q, mask=mask, pos=torch.from_numpy(d["L_position"].astype(np.float32)),
             view=torch.from_numpy(d["L_view"].astype(np.float32)))
    c["photo"] = torch.round(linear_to_srgb(rgb * 0.8 / q) * 255) / 255  # what an estimator is given
    return c


def angle(a, b):
    return torch.rad2deg(torch.acos((a * b).sum(-1).clamp(-1, 1)))


def smooth_normals(n, sel, sigma):
    m = sel[None, None].float()
    return F.normalize(masked_blur(n.permute(2, 0, 1)[None], m, sigma)[0].permute(1, 2, 0), dim=-1)


def inflate(mask, iters=300):
    """Normals from the silhouette: outward on the contour, harmonic in the interior (coarse to fine)."""
    m = mask.float()[None, None]
    g = blur(m, 2.0)
    gx = F.pad(g[..., :, 2:] - g[..., :, :-2], (1, 1))
    gy = F.pad(g[..., 2:, :] - g[..., :-2, :], (0, 0, 1, 1))
    rim = F.normalize(torch.cat([-gx, gy], 1), dim=1)  # outward; image rows run down, camera y runs up
    edge = (m > 0) & (-F.max_pool2d(-m, 3, 1, 1) < 1)
    size = mask.shape[-1]
    field = None
    for s in [s for s in (32, 64, 128, 256, 512, 1024) if s <= size]:
        ms = F.interpolate(m, size=s, mode="area") > 0.5
        es = (F.interpolate((edge.float() * 1), size=s, mode="area") > 0) & ms
        rs = F.normalize(F.interpolate(rim * edge, size=s, mode="area"), dim=1)
        field = torch.zeros(1, 2, s, s) if field is None else F.interpolate(field, size=s, mode="bilinear", align_corners=False)
        w = ms.float()
        for _ in range(iters):
            avg = F.avg_pool2d(field * w, 3, 1, 1) / F.avg_pool2d(w, 3, 1, 1).clamp_min(1e-6)
            field = torch.where(es, rs, avg) * w
    xy = field[0].permute(1, 2, 0)
    r = xy.norm(dim=-1, keepdim=True).clamp(max=0.999)
    xy = F.normalize(xy, dim=-1) * r
    return torch.cat([xy, (1 - (xy**2).sum(-1, keepdim=True)).sqrt()], -1)


def shade(n, y, sel, rounds=2):
    """Move each normal along the fitted light until n.L + b equals the shading (smallest change that does so)."""
    for _ in range(rounds):
        A = torch.cat([n[sel], torch.ones(int(sel.sum()), 1)], 1)
        sol = torch.linalg.lstsq(A, y[sel][:, None]).solution[:, 0]
        L, b = sol[:3], sol[3]
        e = y - ((n * L).sum(-1) + b)
        n = F.normalize(n + (e / (L * L).sum())[..., None] * L, dim=-1)
        n = torch.where((n[..., 2:] < 0.02), F.normalize(torch.cat([n[..., :2], torch.full_like(n[..., 2:], 0.02)], -1), dim=-1), n)
    return n


class Estimators:
    def __init__(self):
        sys.path.insert(0, "external/MoGe")
        from diffusers import MarigoldNormalsPipeline
        from moge.model.v2 import MoGeModel
        self.moge = {k: MoGeModel.from_pretrained(f"Ruicheng/moge-2-vit{k.lower()}-normal").eval() for k in ("B", "L")}
        self.marigold = MarigoldNormalsPipeline.from_pretrained("prs-eth/marigold-normals-v1-1", torch_dtype=torch.float32)
        self.marigold.set_progress_bar_config(disable=True)
        self.times = {}

    def timed(self, name, fn):
        torch.cuda.synchronize()
        t = time.time()
        out = fn()
        torch.cuda.synchronize()
        self.times.setdefault(name, []).append(time.time() - t)
        return out

    @torch.no_grad()
    def run(self, c):
        y, mask = (c["lin"].clamp(0, 1) * LUM[0]).sum(0), c["mask"]
        out = {}
        out["inflation"] = self.timed("inflation", lambda: inflate(mask))
        out["inflation+shade"] = self.timed("inflation+shade", lambda: shade(inflate(mask), y, mask))
        for k, model in self.moge.items():
            model.cuda()
            out[f"MoGe-2 {k}"] = self.timed(f"MoGe-2 {k}", lambda: model.infer(c["photo"].cuda(), use_fp16=False)["normal"].cpu())
            model.cpu()
        self.marigold.to("cuda")
        img = Image.fromarray((c["photo"].permute(1, 2, 0).numpy() * 255).astype(np.uint8))
        for steps in (1, 4):
            run = lambda: self.marigold(img, num_inference_steps=steps, ensemble_size=1, processing_resolution=512,
                                        generator=torch.Generator("cuda").manual_seed(0)).prediction
            out[f"Marigold {steps}"] = torch.from_numpy(np.asarray(self.timed(f"Marigold {steps}", run))[0]).float()
        self.marigold.to("cpu")
        return {k: F.normalize(torch.nan_to_num(v), dim=-1) for k, v in out.items()}


def scores(n, nt, v, y, sel):
    lo, lo_t = smooth_normals(n, sel, 8.0), smooth_normals(nt, sel, 8.0)
    hi, hi_t = (n - lo)[sel][:, :2].flatten(), (nt - lo_t)[sel][:, :2].flatten()
    lt, le = light_direction(nt, y, sel)[0], light_direction(n, y, sel)[0]
    wt, we = (physical_layer(a, v, y, sel, 0.25, compact=0.0)[0][sel] for a in (nt, n))
    tt, te = (physical_layer(a, v, y, sel, 0.25)[0][sel] for a in (nt, n))
    k = max(1, int(0.05 * len(tt)))
    top = len(set(torch.topk(tt, k).indices.tolist()) & set(torch.topk(te, k).indices.tolist())) / k
    return [float(angle(n, nt)[sel].mean()), float(angle(lo, lo_t)[sel].mean()), float(torch.corrcoef(torch.stack([hi, hi_t]))[0, 1]),
            float(angle(le, lt)), float(torch.corrcoef(torch.stack([we, wt]))[0, 1]),
            float(torch.corrcoef(torch.stack([te, tt]))[0, 1]), top]


def main():
    out = Path("runs/normals")
    out.mkdir(parents=True, exist_ok=True)
    est = Estimators()
    files = sorted(glob.glob(f"{ROOT}/*__matte_dark__*.npz"))
    data = []
    for f in files:
        c = load(f)
        nt, v, sel = geometry(c)
        data.append((Path(f).stem.split("__"), c, nt, v, sel, est.run(c)))
    # axis conventions of the network outputs: the sign pattern that fits the sphere best, fixed for all images
    names = list(data[0][5])
    flips = {}
    for k in names:
        if k.startswith("inflation"):
            flips[k] = torch.ones(3)
            continue
        err = [np.mean([float(angle(m[k] * torch.tensor(fl, dtype=torch.float32), nt)[sel].mean())
                        for (shape, *_), c, nt, v, sel, m in data if shape == "sphere"]) for fl in FLIPS]
        flips[k] = torch.tensor(FLIPS[int(np.argmin(err))], dtype=torch.float32)
    res = {}
    for (shape, _, light), c, nt, v, sel, m in data:
        y = (c["lin"].clamp(0, 1) * LUM[0]).sum(0)
        for k in names:
            m[k] = m[k] * flips[k]
        m["MoGe-2 L+shade"] = shade(m["MoGe-2 L"].clone(), y, sel)
        for k, n in m.items():
            res.setdefault((shape, k), []).append(scores(n, nt, v, y, sel))
    names = list(data[0][5])
    cols = ["angle", "shape angle", "relief r", "light err", "wide r", "layer r", "top 5%"]
    with open(out / "summary.txt", "w") as f:
        print("axis signs applied to (x, y, z): " + ", ".join(f"{k} {tuple(int(t) for t in flips[k])}" for k in flips
                                                              if not k.startswith("inflation")), file=f)
        print("seconds per image: " + ", ".join(f"{k} {np.mean(t[1:] or t):.2f}" for k, t in est.times.items()), file=f)
        print("\nangle: mean angular error (deg). shape angle: the same after smoothing both (sigma 8 px). relief r: "
              "correlation of the fine detail.\nlight err: error (deg) of the light fitted to the shading. wide r / layer r: "
              "correlation of the wide part / whole specular layer\nwith the layer from true normals. top 5%: overlap of the "
              "brightest 5% of the two layers (chance 0.05).", file=f)
        for shape in SHAPES + ["mean"]:
            print(f"\n== {shape} ==\n{'estimator':18s}" + "".join(f"{c_:>13s}" for c_ in cols), file=f)
            for k in names:
                rows = res[(shape, k)] if shape != "mean" else [np.mean(res[(s, k)], 0) for s in SHAPES]
                print(f"{k:18s}" + "".join(f"{x:13.2f}" for x in np.mean(rows, 0)), file=f)
    print(open(out / "summary.txt").read())

    show = [k for k in names if k not in ("MoGe-2 B", "Marigold 1")]
    picks = [d for d in data if "cambridge" in d[0][2]]
    rgb = lambda n, sel: ((n * 0.5 + 0.5) * sel[..., None]).numpy()
    fig, ax = plt.subplots(len(picks), len(show) + 2, figsize=(2.5 * (len(show) + 2), 2.6 * len(picks)))
    fig2, ax2 = plt.subplots(len(picks), len(show) + 2, figsize=(2.5 * (len(show) + 2), 2.6 * len(picks)))
    for r, ((shape, _, _), c, nt, v, sel, m) in enumerate(picks):
        y = (c["lin"].clamp(0, 1) * LUM[0]).sum(0)
        def synth(n):
            layer, diff, _ = physical_layer(n, v, y, sel, 0.25)
            d = torch.where(sel, diff, torch.ones_like(diff))
            return linear_to_srgb(c["lin"] * d[None] + (layer * sel)[None]).permute(1, 2, 0).numpy()
        ims = [c["photo"].permute(1, 2, 0).numpy(), rgb(nt, sel)] + [rgb(m[k], sel) for k in show]
        ims2 = [linear_to_srgb(c["lin"]).permute(1, 2, 0).numpy(), synth(nt)] + [synth(m[k]) for k in show]
        for a, images, first in ((ax, ims, "input"), (ax2, ims2, "matte input")):
            for k_, im in enumerate(images):
                a[r, k_].imshow(im)
                a[r, k_].set_xticks([])
                a[r, k_].set_yticks([])
                if r == 0:
                    a[r, k_].set_title(([first, "true normals"] + show)[k_], fontsize=9)
            a[r, 0].set_ylabel(shape[:9], fontsize=9)
    fig.tight_layout()
    fig.savefig(out / "normals.png", dpi=80)
    fig2.tight_layout()
    fig2.savefig(out / "synthesis.png", dpi=80)


if __name__ == "__main__":
    main()
