"""Figures of the leave-one-shape-out results and of why fine-detailed shapes are over-predicted.

Reads the per-image predictions saved by `shape_holdout.py --save-preds`, the cue file and the manifest, and writes
  shape_results.png   per-shape accuracy (r) and mean rating per shape: human vs models
  shape_why.png       where the error is (matte images), what the image cues see there, and how humans rate shapes
  shape_examples.png  one matte and one glossy material on a smooth and on fine-detailed shapes, highlight region H outlined
Shapes are ordered by the median highlight contrast of matte images (a proxy for how much fine surface detail produces
highlight-like structure without any gloss).
"""
import argparse
import csv
import json
import re

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch

matplotlib.rcParams.update({"font.family": "sans-serif", "font.size": 9, "axes.spines.top": False,
                            "axes.spines.right": False, "axes.edgecolor": "#000000", "axes.labelcolor": "#0b0b0b",
                            "xtick.color": "#000000", "ytick.color": "#000000", "axes.titleweight": "bold",
                            "axes.titlesize": 10, "axes.titlelocation": "left"})
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#b0b0b0"
S1, S2, S3 = "#2a78d6", "#eb6834", "#1baf7a"  # one colour per model
SHORT = {"blob": "blob", "bunny_vt": "bunny", "cylinder": "cylinder", "happy_vrip_vt": "happy", "sphere": "sphere",
         "statuette_vt": "statuette", "surface2-1000x1000part": "surface", "teapot_vt_scale": "teapot",
         "xyzrgb_dragon_vt": "dragon"}
GEO_RE = re.compile(r"\]\[([^\]]+)\]\[")


def geo_of(p):
    return GEO_RE.search(p).group(1)


def key_of(p):  # (illumination, material)
    return re.match(r"\[([^\]]+)\]", p).group(1), re.search(r"\]\[[^\]]+\]\[([^\]]+)\]", p).group(1)


def style(ax, xgrid=True):
    ax.grid(axis="x" if xgrid else "y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(length=0)


def dots(ax, order, series, xlabel, xlim=None, ref=None):
    """Horizontal dot plot: one row per shape, one marker per series [(label, values dict, colour, marker)]."""
    ypos = np.arange(len(order))[::-1]
    for y, g in zip(ypos, order):
        vals = [s[1][g] for s in series if np.isfinite(s[1].get(g, np.nan))]
        if vals:
            ax.plot([min(vals), max(vals)], [y, y], color=GRID, lw=2, zorder=1)
    for lab, vals, col, mk in series:
        xs = [vals.get(g, np.nan) for g in order]
        ax.scatter(xs, ypos, s=46, color=col, marker=mk, label=lab, zorder=3, edgecolors="#fcfcfb", linewidths=1.2)
    if ref is not None:
        ax.axvline(ref, color=MUTED, lw=0.8, ls=(0, (3, 3)))
    ax.set_yticks(ypos, [SHORT[g] for g in order])
    ax.set_xlabel(xlabel)
    if xlim:
        ax.set_xlim(*xlim)
    style(ax)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--preds", default="runs/highlight_cues/shape_preds_all9.npz")
    p.add_argument("--cues", default="runs/highlight_cues/cues_all9_q95.csv")
    p.add_argument("--meta", default="runs/highlight_cues/cues_all9_q95.json")
    p.add_argument("--manifest", default="data/serrano/manifest_all9.csv")
    p.add_argument("--root", default="data/serrano/images")
    p.add_argument("--out", default="runs/figures")
    args = p.parse_args()

    d = np.load(args.preds, allow_pickle=True)
    names, P, va, geo, paths = list(d["names"]), d["preds"] * 6 + 1, d["va"], d["geo"], d["paths"]
    human = d["glossiness"] * 6 + 1
    iT, iR, iU = (names.index(n) for n in ("cues: trio + B", "cues: trio + B + reflection", "U-Net RGB (shape hold-out)"))
    geoms = sorted(set(geo))

    rows = list(csv.DictReader(open(args.cues)))
    man = {r["path"]: float(r["gloss_raw"]) for r in csv.DictReader(open(args.manifest))}
    cg = np.array([geo_of(r["path"]) for r in rows])
    cr = np.array([man[r["path"]] for r in rows])
    contrast = np.array([float(r["contrast"]) if r["contrast"] not in ("", "nan") else np.nan for r in rows])
    matte_c = {g: np.nanmedian(contrast[(cg == g) & (cr <= 1)]) for g in geoms}
    gloss_c = {g: np.nanmedian(contrast[(cg == g) & (cr >= 5)]) for g in geoms}
    order = sorted(geoms, key=lambda g: matte_c[g])  # smooth -> fine detail (top -> bottom)

    test = {g: va[geo[va] == g] for g in geoms}
    r_of = lambda i: {g: np.corrcoef(P[i][test[g]], human[test[g]])[0, 1] if np.isfinite(P[i][test[g]]).all() else np.nan
                      for g in geoms}
    mean_of = lambda i, sel=None: {g: np.nanmean(P[i][test[g] if sel is None else test[g][sel(g)]]) for g in geoms}
    hmean = {g: human[test[g]].mean() for g in geoms}
    models = [("highlight cues", iT, S1, "o"), ("+ reflection cues", iR, S3, "s"), ("U-Net", iU, S2, "D")]

    # ---- figure 1: results
    fig, axs = plt.subplots(1, 2, figsize=(10, 4.2), facecolor="#fcfcfb", constrained_layout=True)
    dots(axs[0], order, [(lab, r_of(i), c, m) for lab, i, c, m in models], "Pearson r (held-out shape)", (0, 1))
    axs[0].set_title("a  Accuracy on each held-out shape")
    dots(axs[1], order, [("human", hmean, INK, "|")] + [(lab, mean_of(i), c, m) for lab, i, c, m in models],
         "mean glossiness rating (1-7; 104 validation materials x 9 lights)", (1, 6.5))
    axs[1].set_title("b  Mean rating per shape: human vs models")
    axs[1].legend(loc="upper right", frameon=False, fontsize=8)
    fig.text(0.01, -0.04, "Shapes are ordered by highlight-candidate contrast on matte materials (smooth at the top, fine relief at the bottom). "
             "surface has no background, so '+ reflection' is not evaluated there. U-Net: one seed.", color=MUTED, fontsize=8)
    fig.savefig(f"{args.out}/shape_results.png", dpi=160, bbox_inches="tight")

    # ---- figure 2: why
    H = {}
    for pth, v in man.items():
        H.setdefault(geo_of(pth), {})[key_of(pth)] = v
    sph_glossy = [k for k, v in H["sphere"].items() if v >= 5]
    on_shape = {g: np.mean([H[g][k] for k in sph_glossy if k in H[g]]) for g in geoms}
    slope = {g: np.polyfit(*np.array([(H["sphere"][k], H[g][k]) for k in H["sphere"] if k in H[g]]).T, 1)[0]
             for g in geoms}

    fig, axs = plt.subplots(1, 3, figsize=(13, 4.2), facecolor="#fcfcfb", constrained_layout=True)
    matte = lambda g: human[test[g]] <= 1
    dots(axs[0], order, [(lab, mean_of(i, matte), c, m) for lab, i, c, m in models if i != iR],
         "mean prediction on matte images (human rating = 1)", (1, 6), ref=1)
    axs[0].set_title("a  Main error: matte materials predicted glossy")
    axs[0].legend(loc="upper right", frameon=False, fontsize=8)
    dots(axs[1], order, [("matte materials (rating 1)", matte_c, S1, "o"), ("glossy materials (rating >= 5)", gloss_c, S2, "D")],
         "highlight-candidate contrast (median)", (0, 1), ref=gloss_c["sphere"])
    axs[1].set_title("b  Image: on fine shapes, matte materials show\n     highlight candidates as strong as glossy ones")
    axs[1].legend(loc="lower left", frameon=False, fontsize=8)
    dots(axs[2], order, [("material-light pairs rated >= 5 on the sphere", on_shape, S2, "D")],
         "human mean rating of the same pairs on each shape", (1, 7), ref=np.mean([H["sphere"][k] for k in sph_glossy]))
    for y, g in zip(np.arange(len(order))[::-1], order):
        axs[2].text(1.1, y, f"slope {slope[g]:.2f}", ha="left", va="center", color=MUTED, fontsize=8)
    axs[2].set_title("c  Humans: on fine shapes, glossy materials\n     are rated less glossy")
    fig.text(0.01, -0.08, "a: predictions on held-out shapes, dashed = human rating. b: cue values of all 42,120 images, dashed = glossy sphere value."
             "\nc: the 909 material-light pairs rated >= 5 on the sphere, dashed = their sphere mean. Slope = regression of the shape's ratings on the sphere ratings.",
             color=MUTED, fontsize=8)
    fig.savefig(f"{args.out}/shape_why.png", dpi=160, bbox_inches="tight")

    # ---- figure 3: examples
    from scripts.highlight_cues import cues_for_batch
    from scripts.shape_masks import geometry_masks
    from src.data.datasets import ManifestDataset
    from src.models.features import GlossFeatures

    meta = json.load(open(args.meta))
    show = ["sphere", "teapot_vt_scale", "happy_vrip_vt", "xyzrgb_dragon_vt"]
    pidx = {pth: i for i, pth in enumerate(paths)}
    illum = "1_1_cambridge_2k"
    vkeys = {key_of(paths[i]) for i in va if key_of(paths[i])[0] == illum}
    full = lambda k, g: f"[{k[0]}][{g}][{k[1]}].jpg"
    ok = [k for k in sorted(vkeys) if all(full(k, g) in pidx for g in show)]
    # matte: rated 1 everywhere, largest U-Net prediction on the dragon; glossy: rated >= 6 on the sphere, lowest on dragon
    mk = max((k for k in ok if all(man[full(k, g)] <= 1 for g in show)), key=lambda k: P[iU][pidx[full(k, show[-1])]])
    gk = min((k for k in ok if man[full(k, "sphere")] >= 6), key=lambda k: man[full(k, show[-1])])

    ds = ManifestDataset(args.manifest, args.root, size=256)
    feat = GlossFeatures(meta["sigma_max"], sigma_step=1).cuda()
    allg = np.array([geo_of(pth) for pth in paths])
    fig, axs = plt.subplots(2, len(show), figsize=(2.6 * len(show), 6.2), facecolor="#fcfcfb", constrained_layout=True)
    for c, g in enumerate(show):
        obj, bg, *_ = geometry_masks(ds, np.flatnonzero(allg == g))
        for r, k in enumerate((mk, gk)):
            i = pidx[full(k, g)]
            x = ds[i][0][None].cuda()
            _, Hm, _ = cues_for_batch(x, feat, obj.cuda(), meta["threshold"], bg.cuda())
            ax = axs[r, c]
            ax.imshow(x[0].permute(1, 2, 0).cpu().numpy())
            ax.contour(Hm[0].cpu().numpy(), levels=[0.5], colors=["#e34948"], linewidths=0.7)
            ax.set_xticks([]), ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(False)
            ax.set_title(f"{SHORT[g]}\nhuman {man[full(k, g)]:.0f} / cues {P[iT][i]:.1f} / U-Net {P[iU][i]:.1f}",
                         fontsize=8.5, fontweight="normal", loc="center")
    axs[0, 0].set_ylabel(f"matte: {mk[1]}", fontsize=8.5)
    axs[1, 0].set_ylabel(f"glossy: {gk[1]}", fontsize=8.5)
    fig.suptitle("Same material and light, different shape (red line = detected highlight-candidate region H; numbers are glossiness ratings 1-7)",
                 fontsize=9.5, x=0.01, ha="left")
    fig.savefig(f"{args.out}/shape_examples.png", dpi=160, bbox_inches="tight")
    print(f"matte example {mk}, glossy example {gk}; saved figures to {args.out}")


if __name__ == "__main__":
    main()
