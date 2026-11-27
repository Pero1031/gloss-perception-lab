"""Side-by-side look at the stereo training scenes and the Serrano test images (as the networks see them).

Writes runs/figures/train_scenes.png (random held-out training scenes: tone-mapped left image, b2 residual target,
b1 specular target, with the scene parameters) and runs/figures/serrano_test_images.png (per geometry: matte and
glossy Serrano images at 128 px with the b2 network map next to each).
"""
import json

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scripts.shape_masks import geometry_of
from scripts.train_highlight import tone_map
from src.data.datasets import ManifestDataset
from src.models.unet import UNet



def label(p):
    sh, m = p["shape"], p["material"]
    if sh["kind"] == "erato":
        s = "Erato"
    else:
        lay = [l[0] for l in sh.get("layers", []) if l[2] > 0.002]
        s = "sphere" if not lay else "blob(" + "+".join(lay) + ")"
    if m.get("kind") == "diffuse":
        mt = "diffuse"
    else:
        mt = ("metal" if m.get("kind") == "metal" else "dielectric") + f" roughness {m.get('roughness', 0):.2f}"
        if m.get("coat_weight", 0):
            mt += " coat"
    return f"{s} / {mt}"


def net(target):
    ck = torch.load(f"runs/highlight_{target}/last.pt")
    a = ck["args"]
    m = UNet(3, 1, a["base"], a["depth"], head="map").cuda().eval()
    m.load_state_dict(ck["model"])
    return m


def train_figure(rng):
    d = np.load("data/stereo_ds_targets.npz")
    n = len(d["names"])
    pick = rng.choice(np.arange(n - n // 10, n), 16, replace=False)
    fig, ax = plt.subplots(8, 6, figsize=(12, 16.5))
    for k, i in enumerate(pick):
        r, c = k % 8, (k // 8) * 3
        rgb = torch.from_numpy(d["rgb"][i:i + 1].astype(np.float32))
        x = tone_map(rgb, torch.from_numpy(d["mask"][i:i + 1]), fixed=1.0)[0].permute(1, 2, 0).numpy()
        p = json.loads(d["params"][i])
        ax[r, c].imshow(x)
        ax[r, c].set_title(label(p) + f"\nlight {p['world']['name']}", fontsize=7)
        ax[r, c + 1].imshow(np.sqrt(d["residual"][i].astype(np.float32).clip(0, 1)), cmap="magma", vmin=0, vmax=0.6)
        ax[r, c + 2].imshow(d["spec"][i].astype(np.float32), cmap="magma", vmin=0, vmax=1)
        if r == 0:
            ax[r, c + 1].set_title("b2 target: left-right residual (sqrt)", fontsize=8)
            ax[r, c + 2].set_title("b1 target: specular fraction", fontsize=8)
    for a in ax.flat:
        a.set_xticks([])
        a.set_yticks([])
    fig.suptitle("Training data: 16 random held-out scenes, as the net sees them (exposure 1.0)", fontsize=11)
    plt.tight_layout()
    plt.savefig("runs/figures/train_scenes.png", dpi=85)


def serrano_figure(rng):
    ds = ManifestDataset("data/serrano/manifest_all9.csv", "data/serrano/images", size=128)
    paths = [r[0] for r in ds.rows]
    y = np.array([r[1] for r in ds.rows]) * 6 + 1
    geo = np.array([geometry_of(p) for p in paths])
    m = net("residual")
    G = sorted(set(geo))
    fig, ax = plt.subplots(len(G), 8, figsize=(13, 1.75 * len(G) + 0.6))
    for r, g in enumerate(G):
        sel = [rng.choice(np.flatnonzero((geo == g) & (y <= 1)), 2, replace=False),
               rng.choice(np.flatnonzero((geo == g) & (y >= 5.5)), 2, replace=False)]
        for c, i in enumerate(np.concatenate(sel)):
            x = ds[i][0]
            with torch.no_grad():
                M = (torch.sigmoid(m(x[None].cuda()))[0, 0] ** 2).cpu()
            ax[r, 2 * c].imshow(x.permute(1, 2, 0).clamp(0, 1))
            ax[r, 2 * c].set_title(f"{'matte' if c < 2 else 'glossy'}  human {y[i]:.0f}", fontsize=7)
            ax[r, 2 * c + 1].imshow(M, cmap="magma", vmin=0, vmax=0.3)
            ax[r, 2 * c + 1].set_title("b2 map", fontsize=7)
        ax[r, 0].set_ylabel(g[:14], fontsize=8)
    for a in ax.flat:
        a.set_xticks([])
        a.set_yticks([])
    fig.suptitle("Test images (Serrano, 128 px) and b2 maps (colour limit 0.3; on training data diffuse ~0.01, glossy ~0.05)",
                 fontsize=10)
    plt.tight_layout()
    plt.savefig("runs/figures/serrano_test_images.png", dpi=85)


if __name__ == "__main__":
    rng = np.random.default_rng(1)
    train_figure(rng)
    serrano_figure(rng)
    print("saved runs/figures/train_scenes.png, runs/figures/serrano_test_images.png")
