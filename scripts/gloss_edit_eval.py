"""Test the gloss-manipulation prototype on renders whose true specular component is known.

Sets: A = Serrano geometries with our materials (data/serrano_render, from serrano_shape_render.py),
      B = held-out random scenes (data/stereo_ds, last 10%; the b2 network was not trained on them).
1. Detection: pixel AUC of each detector score for specular pixels (specular share > 0.5 vs < 0.1) on non-matte
   images, and, at the threshold that finds half of the specular pixels, the share of object pixels flagged on matte
   images (every flag there is false) and the precision on non-matte images (flags with specular share > 0.3).
2. Editing: layer[...] splits the accepted region into base + layer; share[...] takes a per-pixel specular share
   (b1 network or the truth) and scales that part of the luminance. Gain 0 (suppress) and gain 2 (enhance) are
   compared with the same edit of the true specular component (RMSE of sRGB values on object pixels, x100). The additive rule
   I + gain * P is given its best gain per direction.
Writes runs/gloss_edit/summary.txt and examples.png.
"""
import argparse
import glob
import json
from pathlib import Path

import matplotlib
import numpy as np
import torch
import torch.nn.functional as F

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scripts.gloss_edit import LUM, Detector, edit, edit_additive, edit_share, linear_to_srgb, specular_layer
from src.models.unet import UNet

GREY = 128 / 255
EXPOSURE = 0.8  # the object's 99.5th luminance percentile after scaling
DETECTORS = ["P", "P_balance", "P_skew", "P_white", "b2", "b1"]
BRIGHT = 0.0  # --bright: specular pixels count as positives only if their specular luminance exceeds this (linear)


def load(path):
    d = np.load(path)
    rgb = torch.from_numpy(d["L_rgb"].astype(np.float32)).permute(2, 0, 1)
    mask = torch.from_numpy(d["L_mask"])
    glossy = torch.from_numpy(d["L_glossy"].astype(np.float32))
    y = (rgb * LUM[0]).sum(0)
    scale = EXPOSURE / torch.quantile(y[mask], 0.995).clamp_min(1e-6)
    p = json.loads(str(d["params"]))
    if isinstance(p["shape"], str):
        shape, matte = p["shape"], p["material"].startswith("matte")
    else:
        fine = any(layer[0] == "fine" for layer in p["shape"].get("layers", []))
        shape, matte = ("relief" if fine else "smooth"), float(glossy[mask].max()) <= 0
    return dict(lin=rgb * scale, mask=mask, glossy=glossy * scale, spec=(glossy / y.clamp_min(1e-6)).clamp(0, 1),
                shape=shape, matte=matte, name=Path(path).stem)


def load_net(name):
    ck = torch.load(f"runs/highlight_{name}/last.pt")
    a = ck["args"]
    net = UNet(3, 1, a["base"], a["depth"], head="map").cuda().eval()
    net.load_state_dict(ck["model"])
    return net


@torch.no_grad()
def net_map(net, x, mask):
    x128 = F.interpolate(x, size=128, mode="area")
    m128 = F.avg_pool2d(mask[:, None].float(), 2) > 0.5
    x128 = torch.where(m128, x128, torch.full_like(x128, GREY))
    return F.interpolate(torch.sigmoid(net(x128)) ** 2, size=x.shape[-1], mode="bilinear", align_corners=False)[:, 0]


def auc(pos, neg):
    v = np.concatenate([pos, neg])
    r = np.empty(len(v))
    order = np.argsort(v, kind="stable")
    r[order] = np.arange(1, len(v) + 1)
    return (r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def rmse(a, b, m):
    return float(((a - b) ** 2).mean(1)[0][m].mean().sqrt()) * 100


def run_set(items, det, nets, rng):
    B = 16
    samples = {k: {"pos": [], "neg": []} for k in DETECTORS}
    cache = []
    for s in range(0, len(items), B):
        chunk = items[s:s + B]
        lin = torch.stack([c["lin"] for c in chunk]).cuda()
        mask = torch.stack([c["mask"] for c in chunk]).cuda()
        x = torch.round(linear_to_srgb(lin) * 255) / 255
        maps = det.maps(x, mask)
        maps["b2"], maps["b1"] = (net_map(n, x, mask) for n in nets)
        for i, c in enumerate(chunk):
            spec, m = c["spec"].cuda(), mask[i]
            c["x"] = x[i].cpu()
            c["maps"] = {k: maps[k][i].cpu() for k in DETECTORS}
            if not c["matte"]:
                pos = (spec > 0.5) & m & (c["glossy"].cuda() > BRIGHT)
                for lab, sel in (("pos", pos), ("neg", (spec < 0.1) & m)):
                    idx = torch.nonzero(sel.flatten())[:, 0]
                    idx = idx[torch.from_numpy(rng.permutation(len(idx))[:3000]).to(idx.device)]
                    for k in DETECTORS:
                        samples[k][lab].append(maps[k][i].flatten()[idx].cpu().numpy())
            cache.append(c)
    res, thr = {}, {}
    for k in DETECTORS:
        pos, neg = np.concatenate(samples[k]["pos"]), np.concatenate(samples[k]["neg"])
        thr[k] = float(np.median(pos))  # recall 0.5
        false, prec = {}, []
        for c in cache:
            flag = (c["maps"][k] > thr[k]) & c["mask"]
            if c["matte"]:
                false.setdefault(c["shape"], []).append(float(flag.sum()) / float(c["mask"].sum()))
            elif flag.any():
                prec.append(float((c["spec"][flag] > 0.3).float().mean()))
        res[k] = dict(auc=auc(pos, neg), prec=np.mean(prec), false={g: 100 * np.mean(v) for g, v in false.items()})
    return cache, res, thr


@torch.no_grad()
def edit_eval(cache, thr):
    """RMSE (x100, sRGB) between each edit and the same edit of the true specular component."""
    rows = {}
    groups = sorted({("matte " if c["matte"] else "glossy ") + c["shape"] for c in cache})
    add_gains = (0.1, 0.2, 0.3, 0.5, 0.75, 1.0)
    acc = {g: {} for g in groups}
    for c in cache:
        g = ("matte " if c["matte"] else "glossy ") + c["shape"]
        x, m = c["x"][None].cuda(), c["mask"].cuda()
        lin, glossy = c["lin"][None].cuda(), c["glossy"][None, None].cuda()
        oracle = {0: linear_to_srgb(lin - glossy), 2: linear_to_srgb(lin + glossy)}
        e = {}
        for gain in (0, 2):
            e[("none", gain)] = rmse(x, oracle[gain], m)
            for k in ("P", "P_balance", "b2"):
                base, layer = specular_layer(x, ((c["maps"][k] > thr[k]) & c["mask"])[None].cuda())
                e[(f"layer[{k}]", gain)] = rmse(edit(base, layer, gain), oracle[gain], m)
            base, layer = specular_layer(x, ((c["spec"] > 0.3) & c["mask"])[None].cuda())
            e[("layer[true region]", gain)] = rmse(edit(base, layer, gain), oracle[gain], m)
            for k, share in (("share[b1]", c["maps"]["b1"]), ("share[true]", c["spec"])):
                e[(k, gain)] = rmse(edit_share(x, (share * c["mask"])[None].cuda(), gain), oracle[gain], m)
            for a in add_gains:
                e[(f"additive P {a}", gain)] = rmse(edit_additive(x, c["maps"]["P"][None].cuda(), a if gain else -a),
                                                    oracle[gain], m)
        for k_, v in e.items():
            acc[g].setdefault(k_, []).append(v)
    for g in groups:
        rows[g] = {k_: float(np.mean(v)) for k_, v in acc[g].items()}
    return rows, add_gains


def write_tables(name, res, rows, add_gains, f):
    shapes = sorted({g for r in res.values() for g in r["false"]})
    print(f"\n== {name}: detection (threshold at recall 0.5 of specular pixels) ==", file=f)
    print(f"{'detector':12s}{'AUC':>7s}{'precision':>11s}   flagged % of object on matte images: "
          + "".join(f"{g[:12]:>14s}" for g in shapes), file=f)
    for k, r in res.items():
        print(f"{k:12s}{r['auc']:7.3f}{r['prec']:11.2f}{'':41s}" + "".join(f"{r['false'].get(g, np.nan):14.2f}" for g in shapes),
              file=f)
    print(f"\n== {name}: edit error against the true-specular edit (RMSE x100, sRGB, object pixels) ==", file=f)
    methods = ["none", "layer[P]", "layer[b2]", "layer[true region]", "share[b1]", "share[true]"]
    for gain, lab in ((0, "suppress (gain 0)"), (2, "enhance (gain 2)")):
        mean_add = {a: np.mean([rows[g][(f"additive P {a}", gain)] for g in rows]) for a in add_gains}
        best = min(mean_add, key=mean_add.get)
        cols = methods + [f"additive P {best}"]
        print(f"{lab:26s}" + "".join(f"{c.replace('additive P', 'add P'):>19s}" for c in cols), file=f)
        for g in rows:
            print(f"  {g:24s}" + "".join(f"{rows[g][(c, gain)]:19.2f}" for c in cols), file=f)


def examples(cache, path):
    want = [("xyzrgb_dragon_vt", True), ("xyzrgb_dragon_vt", False), ("happy_vrip_vt", True), ("bunny_vt", False),
            ("sphere", False)]
    picks = []
    for shape, matte in want:
        c = [c for c in cache if c["shape"] == shape and c["matte"] == matte and "cambridge" in c["name"]
             and ("matte_grey" in c["name"] or "glossy" in c["name"])]
        picks += c[:1]
    titles = ["image", "predicted specular share (b1)", "true specular share", "suppress (b1)",
              "suppress (true specular)", "enhance x2 (b1)", "enhance x2 (true specular)"]
    fig, ax = plt.subplots(len(picks), len(titles), figsize=(2.6 * len(titles), 2.7 * len(picks)))
    for r, c in enumerate(picks):
        x = c["x"][None].cuda()
        share = c["maps"]["b1"] * c["mask"]
        lin, glossy = c["lin"][None].cuda(), c["glossy"][None, None].cuda()
        imgs = [c["x"], share, c["spec"] * c["mask"], edit_share(x, share[None].cuda(), 0)[0].cpu(),
                linear_to_srgb(lin - glossy)[0].cpu(), edit_share(x, share[None].cuda(), 2)[0].cpu(),
                linear_to_srgb(lin + glossy)[0].cpu()]
        for k, im in enumerate(imgs):
            if im.dim() == 2:
                ax[r, k].imshow(im.numpy(), cmap="magma", vmin=0, vmax=1)
            else:
                ax[r, k].imshow(im.permute(1, 2, 0).clamp(0, 1).numpy())
            ax[r, k].set_xticks([])
            ax[r, k].set_yticks([])
            if r == 0:
                ax[r, k].set_title(titles[k], fontsize=8)
        ax[r, 0].set_ylabel(f"{c['shape'][:9]}\n{'matte' if c['matte'] else 'glossy'}", fontsize=9)
    plt.tight_layout()
    plt.savefig(path, dpi=90)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="runs/gloss_edit")
    p.add_argument("--n-random", type=int, default=400, help="held-out random scenes used for set B")
    p.add_argument("--bright", type=float, default=0.0,
                   help="positives: only specular pixels whose specular luminance exceeds this (0-1, linear)")
    args = p.parse_args()
    global BRIGHT
    BRIGHT = args.bright
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    det, nets = Detector(), (load_net("residual"), load_net("spec"))
    held = sorted(glob.glob("data/stereo_ds/scene_*.npz"))[-1000:]
    sets = {"A: Serrano geometries": sorted(glob.glob("data/serrano_render/*.npz")),
            "B: random scenes": [held[i] for i in rng.choice(len(held), args.n_random, replace=False)]}
    with open(out / "summary.txt", "w") as f:
        for name, paths in sets.items():
            cache, res, thr = run_set([load(q) for q in paths], det, nets, rng)
            rows, add_gains = edit_eval(cache, thr)
            write_tables(f"{name} ({len(cache)} images, {sum(c['matte'] for c in cache)} matte)", res, rows, add_gains, f)
            if name.startswith("A"):
                examples(cache, out / "examples.png")
    print(open(out / "summary.txt").read())


if __name__ == "__main__":
    main()
