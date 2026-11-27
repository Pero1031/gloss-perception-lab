"""The gloss map P of Nagata et al. (product of responses across scales) on the Serrano sphere images.

No training: P is computed from fixed centre-surround filters. Reports how well P (and simpler statistics)
correlate with the human gloss ratings, and draws maps for example images.
"""
import argparse
import csv
import random
import re
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from src.data.datasets import ManifestDataset
from src.models.features import GlossFeatures
from src.train import rating_bins


def sphere_mask(ds, n_per_illum=60, thresh=0.03):
    """Pixels that change with the material under a fixed illumination belong to the object; fit a circle to them."""
    by_illum = defaultdict(list)
    for i, (path, _) in enumerate(ds.rows):
        by_illum[re.match(r"\[(.+?)\]", path).group(1)].append(i)
    votes = []
    for idx in by_illum.values():
        sel = idx[::max(1, len(idx) // n_per_illum)][:n_per_illum]
        votes.append(torch.stack([ds[i][0] for i in sel]).std(0).mean(0) > thresh)
    obj = torch.stack(votes).float().mean(0) > 0.5
    h, w = obj.shape
    ys, xs = torch.nonzero(obj, as_tuple=True)
    cy, cx = ys.float().mean(), xs.float().mean()
    radius = (obj.sum().float() / torch.pi).sqrt()
    yy, xx = torch.meshgrid(torch.arange(h).float(), torch.arange(w).float(), indexing="ij")
    return ((yy - cy) ** 2 + (xx - cx) ** 2).sqrt(), float(radius), (float(cy), float(cx))


def average_ranks(v):
    u, inv, counts = torch.unique(v, return_inverse=True, return_counts=True)
    ends = counts.cumsum(0).float()
    return (ends - (counts - 1) / 2)[inv]


def pearson(a, b):
    a, b = a - a.mean(), b - b.mean()
    return float((a * b).sum() / (a.norm() * b.norm()).clamp_min(1e-12))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", default="data/serrano/manifest_sphere.csv")
    p.add_argument("--root", default="data/serrano/images")
    p.add_argument("--size", type=int, default=256)
    p.add_argument("--sigma-max", type=int, default=32)
    p.add_argument("--erode", type=float, default=10, help="pixels to stay away from the sphere silhouette")
    p.add_argument("--out", default="runs/gloss_map_P")
    args = p.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    ds = ManifestDataset(args.manifest, args.root, size=args.size, cache=True)
    dist, radius, (cy, cx) = sphere_mask(ds)
    mask = (dist <= radius - args.erode).cuda()
    print(f"sphere: centre=({cy:.0f},{cx:.0f}) radius={radius:.0f}px; measuring inside radius {radius - args.erode:.0f}px "
          f"({int(mask.sum())} pixels)")

    feat = GlossFeatures(args.sigma_max, sigma_step=1).cuda()
    print(f"scales: 1..{args.sigma_max} ({len(feat.sigmas)} scales)")
    names = ["lum_mean", "lum_p99", "P_mean", "P_max", "P_p99", "S_mean", "S_max", "S_p99"]
    stats = {k: [] for k in names}

    def reduce(m, prefix, with_max=True):
        v = m[:, mask]  # (B, n_pixels)
        stats[f"{prefix}_mean"].append(v.mean(1).cpu())
        if with_max:
            stats[f"{prefix}_max"].append(v.amax(1).cpu())
        stats[f"{prefix}_p99"].append(torch.quantile(v, 0.99, dim=1).cpu())

    for s in range(0, len(ds), 8):
        x = torch.stack([ds[i][0] for i in range(s, min(s + 8, len(ds)))]).cuda()
        lum = 0.299 * x[:, 0] + 0.587 * x[:, 1] + 0.114 * x[:, 2]
        reduce(lum, "lum", with_max=False)
        reduce(feat.gloss_map(x, product=True), "P")
        reduce(feat.gloss_map(x, product=False), "S")
    stats = {k: torch.cat(v) for k, v in stats.items() if v}
    y = torch.tensor([r[1] for r in ds.rows])

    print("\ncorrelation with the human gloss rating (all 4,680 sphere images, no fitting):")
    print(f"{'statistic':10s} {'Spearman':>9s} {'Pearson':>9s}")
    ry = average_ranks(y)
    for k, v in stats.items():
        print(f"{k:10s} {pearson(average_ranks(v), ry):9.3f} {pearson(v, y):9.3f}")

    bins = rating_bins(y)
    print("\nmean of each statistic per rating level (P and S scaled by their overall mean):")
    print("rating  n     " + "  ".join(f"{k:>8s}" for k in stats))
    for b in range(7):
        sel = bins == b
        print(f"{b + 1:6d} {int(sel.sum()):5d} " + "  ".join(
            f"{float(v[sel].mean() / v.mean()):8.2f}" for v in stats.values()))

    with open(out / "stats.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["path", "gloss"] + list(stats))
        for i, (path, g) in enumerate(ds.rows):
            w.writerow([path, g] + [float(v[i]) for v in stats.values()])

    # example maps: two images for each of the rating levels 1, 3, 5, 7
    rng = random.Random(0)
    picks = []
    for level in (0, 2, 4, 6):
        picks += rng.sample([i for i in range(len(ds)) if bins[i] == level], 2)
    xs = torch.stack([ds[i][0] for i in picks]).cuda()
    maps = {"P (product)": feat.gloss_map(xs, True).cpu(), "S (plain sum)": feat.gloss_map(xs, False).cpu()}
    fig, axes = plt.subplots(3, len(picks), figsize=(2.0 * len(picks), 6.4))
    inside = mask.cpu()
    for col, i in enumerate(picks):
        axes[0, col].imshow(xs[col].cpu().permute(1, 2, 0))
        axes[0, col].set_title(f"rating {ds.rows[i][1] * 6 + 1:.0f}", fontsize=9)
    for row, (name, m) in enumerate(maps.items(), start=1):
        vmax = torch.quantile(m[:, inside], 0.995).item()
        for col in range(len(picks)):
            axes[row, col].imshow(m[col], cmap="inferno", vmin=0, vmax=vmax)
        axes[row, 0].text(-0.05, 0.5, name, transform=axes[row, 0].transAxes, rotation=90, va="center",
                          ha="right", fontsize=9)
    for ax in axes.ravel():
        ax.axis("off")
    fig.savefig(out / "maps.png", dpi=130, bbox_inches="tight")
    plt.close(fig)

    # mask check
    fig, ax = plt.subplots(figsize=(3, 3))
    ax.imshow(xs[0].cpu().permute(1, 2, 0))
    ax.contour(inside.numpy(), levels=[0.5], colors="cyan", linewidths=1)
    ax.axis("off")
    fig.savefig(out / "mask_check.png", dpi=110, bbox_inches="tight")
    print("saved", out)


if __name__ == "__main__":
    main()
