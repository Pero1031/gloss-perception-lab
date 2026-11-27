"""Where do the highlight detectors fire on matte, fine-detailed shapes? Overlays and close-ups.

Rows: our Blender renders of Serrano geometries (matte = no specular reflection at all, so every detection is false;
glossy for comparison, with the true specular share) and Serrano images rated 1 (matte) on the same geometries.
Columns: image; the top 3% of object pixels by the gloss map P (red), by the b2 map (cyan), both (yellow);
close-up of the densest part of H; b2 network map (grey background, as trained); true specular share (renders only).
Writes runs/figures/false_highlights.png.
"""
import glob

import matplotlib
import numpy as np
import torch
import torch.nn.functional as F
from matplotlib.patches import Rectangle

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scripts.shape_masks import geometry_masks, geometry_of
from scripts.train_highlight import GREY, LUM, tone_map
from src.data.datasets import ManifestDataset
from src.models.features import GlossFeatures
from src.models.unet import UNet

SHAPES = ["xyzrgb_dragon_vt", "statuette_vt", "happy_vrip_vt", "bunny_vt"]
ZOOM = 56  # close-up window (px at 256)
TOP = 0.03  # share of object pixels marked per detector


def load_b2():
    ck = torch.load("runs/highlight_residual/last.pt")
    a = ck["args"]
    m = UNet(3, 1, a["base"], a["depth"], head="map").cuda().eval()
    m.load_state_dict(ck["model"])
    return m, a.get("bg", "scene") == "grey"


def analyse(x256, sil256, feat, b2, grey):
    """x256: (3,256,256) sRGB; sil256: (256,256) bool. Returns H (bool), b2 map (256 px), close-up box."""
    with torch.no_grad():
        P = feat.gloss_map(x256[None].cuda())[0].cpu()
        x128 = F.interpolate(x256[None], size=128, mode="area")
        if grey:
            s128 = F.avg_pool2d(sil256.float()[None, None], 2)[0, 0] > 0.5
            x128 = torch.where(s128, x128, torch.full_like(x128, GREY))
        M = torch.sigmoid(b2(x128.cuda()))[0, 0].cpu() ** 2
    M = F.interpolate(M[None, None], size=256, mode="bilinear", align_corners=False)[0, 0]
    # per image: the TOP share of object pixels by P and by the b2 map = what each detector finds most highlight-like
    H = (P > torch.quantile(P[sil256], 1 - TOP)) & sil256
    Hb = (M > torch.quantile(M[sil256], 1 - TOP)) & sil256
    dens = F.avg_pool2d(H.float()[None, None], ZOOM, stride=4)[0, 0]
    iy, ix = np.unravel_index(int(dens.argmax()), dens.shape)
    return H, Hb, M, (ix * 4, iy * 4)


def overlay(x, H, Hb):
    o = x.permute(1, 2, 0).clamp(0, 1).numpy().copy()
    o[H.numpy()] = 0.3 * o[H.numpy()] + 0.7 * np.array([1, 0, 0])
    both = (H & Hb).numpy()
    only_b = (Hb & ~H).numpy()
    o[only_b] = 0.3 * o[only_b] + 0.7 * np.array([0, 0.9, 1])
    o[both] = 0.3 * o[both] + 0.7 * np.array([1, 1, 0])
    return o


def main():
    feat = GlossFeatures(32, sigma_step=1).cuda()
    b2, grey = load_b2()
    cases = []
    for g in ("xyzrgb_dragon_vt", "happy_vrip_vt", "bunny_vt"):
        for m in ("matte_grey", "glossy_blue"):
            d = np.load(glob.glob(f"data/serrano_render/{g}__{m}__1_1_cambridge_2k.npz")[0])
            rgb = torch.from_numpy(d["L_rgb"].astype(np.float32)).permute(2, 0, 1)[None]
            mask = torch.from_numpy(d["L_mask"])
            x = tone_map(rgb, mask[None], fixed=1.0)[0]
            spec = d["L_glossy"].astype(np.float32) / np.maximum((d["L_rgb"].astype(np.float32) @ LUM.view(3).numpy()), 1e-6)
            cases.append((f"our render {g[:9]}\n{'matte (no specular)' if m.startswith('matte') else 'glossy'}", x, mask,
                          np.clip(spec, 0, 1) * d["L_mask"]))
    ds = ManifestDataset("data/serrano/manifest_all9.csv", "data/serrano/images", size=256)
    y = np.array([r[1] for r in ds.rows]) * 6 + 1
    geo = np.array([geometry_of(r[0]) for r in ds.rows])
    rng = np.random.default_rng(5)
    for g in ("xyzrgb_dragon_vt", "statuette_vt", "happy_vrip_vt"):
        idx = np.flatnonzero(geo == g)
        sil = torch.from_numpy(geometry_masks(ds, idx)[3])
        i = rng.choice(idx[y[idx] <= 1])
        cases.append((f"Serrano {g[:9]}\nmatte (human 1)", ds[i][0], sil, None))

    fig, ax = plt.subplots(len(cases), 5, figsize=(13, 2.75 * len(cases)))
    for r, (title, x, sil, spec) in enumerate(cases):
        H, Hb, M, (bx, by) = analyse(x, sil, feat, b2, grey)
        ov = overlay(x, H, Hb)
        ax[r, 0].imshow(x.permute(1, 2, 0).clamp(0, 1))
        ax[r, 0].set_ylabel(title, fontsize=8)
        ax[r, 1].imshow(ov)
        ax[r, 1].add_patch(Rectangle((bx, by), ZOOM, ZOOM, fill=False, ec="yellow", lw=1.2))
        ax[r, 2].imshow(np.concatenate([x.permute(1, 2, 0).clamp(0, 1).numpy()[by:by + ZOOM, bx:bx + ZOOM],
                                        ov[by:by + ZOOM, bx:bx + ZOOM]], 1), interpolation="nearest")
        ax[r, 3].imshow(M * sil, cmap="magma", vmin=0, vmax=0.3)
        if spec is not None:
            ax[r, 4].imshow(spec, cmap="magma", vmin=0, vmax=1)
        else:
            ax[r, 4].axis("off")
        if r == 0:
            for c, t in enumerate(["image", "top 3%: red P, cyan b2, yellow both", "zoom (left: image, right: flags)", "b2 map",
                                   "true specular share (renders only)"]):
                ax[r, c].set_title(t, fontsize=9)
    for a in ax.flat:
        a.set_xticks([])
        a.set_yticks([])
    plt.tight_layout()
    plt.savefig("runs/figures/false_highlights.png", dpi=80)
    print("saved runs/figures/false_highlights.png")


if __name__ == "__main__":
    main()
