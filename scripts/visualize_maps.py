"""Show the per-pixel output map of a trained model (what is averaged into the gloss rating)."""
import argparse
import random
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

from src.data.datasets import ManifestDataset
from src.models.features import GlossFeatures
from src.models.unet import UNet
from src.train import rating_bins, split_indices


def build_model(a):
    if a.get("input", "rgb") == "multiscale":
        features = GlossFeatures(a.get("sigma_max", 39))
        return nn.Sequential(features, UNet(features.out_channels, 1, a["base"], a["depth"], head="map"))
    return UNet(base=a["base"], depth=a["depth"], head="map")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--out", default=None, help="PNG path (default: next to the checkpoint)")
    p.add_argument("--per-bin", type=int, default=2, help="images per rating level (1, 3, 5, 7)")
    p.add_argument("--per-image-scale", action="store_true", help="scale the color range of each map separately")
    args = p.parse_args()

    ck = torch.load(args.ckpt)
    a = ck["args"]
    model = build_model(a)
    model.load_state_dict(ck["model"])
    model.cuda().eval()

    ds = ManifestDataset(a["manifest"], a["root"], size=a["size"])
    _, val_idx = split_indices(ds, a["val_frac"], a["seed"])
    bins = rating_bins(torch.tensor([ds.rows[i][1] for i in val_idx]))
    rng = random.Random(0)
    picks = []
    for level in (0, 2, 4, 6):
        pool = [i for i, b in zip(val_idx, bins) if b == level]
        picks += rng.sample(pool, min(args.per_bin, len(pool)))

    xs = torch.stack([ds[i][0] for i in picks]).cuda()
    with torch.no_grad():
        maps = model(xs).cpu()[:, 0]
    vmax = maps.abs().max().item()

    fig, axes = plt.subplots(2, len(picks), figsize=(2.2 * len(picks), 4.8))
    for col, (i, m) in enumerate(zip(picks, maps)):
        gt = ds.rows[i][1]
        axes[0, col].imshow(xs[col].cpu().permute(1, 2, 0)[..., :3] if xs.shape[1] == 3 else xs[col, 0].cpu(), cmap="gray")
        axes[0, col].set_title(f"rating {gt * 6 + 1:.0f}", fontsize=9)
        lim = m.abs().max().item() if args.per_image_scale else vmax
        im = axes[1, col].imshow(m, cmap="RdBu_r", vmin=-lim, vmax=lim)
        axes[1, col].set_title(f"mean = {m.mean():.2f}", fontsize=9)
        for row in (0, 1):
            axes[row, col].axis("off")
    fig.colorbar(im, ax=axes[1, :], shrink=0.8, label="output map (red = pushes the rating up)")
    out = Path(args.out) if args.out else Path(args.ckpt).with_name("maps.png")
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print("saved", out)


if __name__ == "__main__":
    main()
