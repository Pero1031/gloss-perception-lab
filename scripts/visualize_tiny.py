"""Show the learned kernels of a TinyGlossNet and where each kernel responds in validation images."""
import argparse
import random
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from src.data.datasets import ManifestDataset
from src.models.tiny import TinyGlossNet
from src.train import rating_bins, split_indices


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--out", default=None, help="PNG path (default: next to the checkpoint)")
    p.add_argument("--per-bin", type=int, default=2, help="images per rating level (1, 3, 5, 7)")
    p.add_argument("--top", type=float, default=0.01, help="fraction of strongest response pixels to mark")
    args = p.parse_args()

    ck = torch.load(args.ckpt)
    a = ck["args"]
    model = TinyGlossNet(a["kernels"], a["kernel_size"])
    model.load_state_dict(ck["model"])
    model.cuda().eval()
    K, ks = a["kernels"], a["kernel_size"]
    fc_w = model.fc.weight[0].detach().cpu()
    print("fc weights (rating change per unit of kernel max):", [round(float(w), 4) for w in fc_w])

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
        resp = model.response(xs).cpu()  # (N, K, h, w), "valid" convolution
        pred = model(xs).cpu()
    pad = ks // 2
    n = len(picks)

    # kernels: colour image (normalised) and its channel mean (positive = brighter than the surround)
    w = model.conv.weight.detach().cpu()
    fig, axes = plt.subplots(2, K, figsize=(2.6 * K, 5.2), squeeze=False)
    for k in range(K):
        wk = w[k]
        axes[0, k].imshow(((wk - wk.min()) / (wk.max() - wk.min())).permute(1, 2, 0))
        axes[0, k].set_title(f"kernel {k} (colour)\nfc weight {fc_w[k]:+.3f}", fontsize=9)
        m = wk.mean(0)
        lim = m.abs().max()
        axes[1, k].imshow(m, cmap="RdBu_r", vmin=-lim, vmax=lim)
        axes[1, k].set_title("channel mean (red = +)", fontsize=9)
        for r in (0, 1):
            axes[r, k].axis("off")
    kernel_png = Path(args.out).with_name("kernels.png") if args.out else Path(args.ckpt).with_name("kernels.png")
    fig.savefig(kernel_png, dpi=130, bbox_inches="tight")
    plt.close(fig)

    # responses: input / response map per kernel (cross = strongest response) / top pixels of the key kernel
    key = int(fc_w.abs().argmax())
    rows = 1 + K + 1
    fig, axes = plt.subplots(rows, n, figsize=(2.0 * n, 2.1 * rows), squeeze=False)
    for col, i in enumerate(picks):
        img = xs[col].cpu().permute(1, 2, 0)
        axes[0, col].imshow(img)
        axes[0, col].set_title(f"rating {ds.rows[i][1] * 6 + 1:.0f}  pred {pred[col] * 6 + 1:.1f}", fontsize=8)
        for k in range(K):
            r = resp[col, k]
            lim = resp[:, k].abs().max()
            ax = axes[1 + k, col]
            ax.imshow(r, cmap="RdBu_r", vmin=-lim, vmax=lim)
            y, x = divmod(int(r.argmax()), r.shape[1])
            ax.plot([x], [y], "k+", ms=9, mew=1.5)
        r = resp[col, key]
        thr = torch.quantile(r.flatten(), 1 - args.top)
        overlay = img.clone()
        mask = torch.zeros(img.shape[:2], dtype=torch.bool)
        mask[pad:pad + r.shape[0], pad:pad + r.shape[1]] = r >= thr
        overlay[mask] = torch.tensor([1.0, 0.0, 1.0])
        axes[-1, col].imshow(overlay)
    for ax in axes.ravel():
        ax.axis("off")
    for k in range(K):
        axes[1 + k, 0].text(-0.05, 0.5, f"kernel {k}", transform=axes[1 + k, 0].transAxes, rotation=90,
                            va="center", ha="right", fontsize=9)
    axes[-1, 0].text(-0.05, 0.5, f"top {args.top:.0%} of kernel {key}", transform=axes[-1, 0].transAxes,
                     rotation=90, va="center", ha="right", fontsize=9)
    out = Path(args.out) if args.out else Path(args.ckpt).with_name("tiny_maps.png")
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print("saved", kernel_png, out)


if __name__ == "__main__":
    main()
