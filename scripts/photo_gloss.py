"""Add a specular layer to a matte object in a photograph.

Normals come from MoGe-2 (external/MoGe), the object region from a mask image or from a hue range, and the layer from
highlight_synthesis.physical_layer: the lighting implied by a Lambert fit to the object's shading, reflected by a
smooth surface. Nothing is known about the photo's exposure or the object's albedo, so --albedo sets the strength.

  python -m scripts.photo_gloss --image photo.jpg --crop x0 y0 x1 y1 --hue 0.58 0.80 --out runs/photo/name
"""
import argparse
import colorsys
import sys
from pathlib import Path

import matplotlib
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageOps

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scripts.gloss_edit import LUM, blur, linear_to_srgb, srgb_to_linear
from scripts.highlight_synthesis import light_direction, physical_layer


def hue_mask(x, lo, hi, min_sat=0.15, min_value=0.0):
    """x: (3,H,W) sRGB. Pixels whose hue lies in [lo, hi] (0-1), that are not grey (saturation above min_sat) and not
    dark (largest channel above min_value); small gaps closed."""
    mx, mn = x.amax(0), x.amin(0)
    d = (mx - mn).clamp_min(1e-6)
    r, g, b = x
    h = torch.where(mx == r, ((g - b) / d) % 6, torch.where(mx == g, (b - r) / d + 2, (r - g) / d + 4)) / 6
    m = ((h >= lo) & (h <= hi) & ((mx - mn) / mx.clamp_min(1e-6) > min_sat) & (mx > min_value)).float()[None, None]
    m = blur(m, 3.0) > 0.5
    return m[0, 0]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--image", required=True)
    p.add_argument("--crop", type=int, nargs=4, help="x0 y0 x1 y1 in the original image")
    p.add_argument("--width", type=int, default=896, help="working width after cropping")
    p.add_argument("--mask", help="mask image of the object (white = object), same framing as the crop")
    p.add_argument("--hue", type=float, nargs=2, help="hue range (0-1) of the object, used when no mask is given")
    p.add_argument("--min-sat", type=float, default=0.15, help="with --hue: least saturation of the object")
    p.add_argument("--min-value", type=float, default=0.0, help="with --hue: least brightness (largest sRGB channel)")
    p.add_argument("--seed", type=float, nargs=2, help="a point on the object (fractions of width and height): the "
                   "object is the region of similar chromaticity around it")
    p.add_argument("--box", type=float, nargs=4, default=[0, 0, 1, 1], help="x0 y0 x1 y1 as fractions; limits the seed region")
    p.add_argument("--tol", type=float, default=0.06, help="chromaticity distance that still counts as the object")
    p.add_argument("--albedo", type=float, default=0.1, help="luminance albedo of the object; sets the layer's strength")
    p.add_argument("--roughness", type=float, nargs="+", default=[0.3, 0.15])
    p.add_argument("--sheen", choices=["chromatic", "white"], default="chromatic",
                   help="chromatic: the wide part of the layer scales the pixel's own colour (chromaticity kept); "
                        "white: it is added as white light")
    p.add_argument("--keep-mean", action=argparse.BooleanOptionalAction, default=True,
                   help="darken the diffuse part so that the object's mean luminance stays the same: a smoother "
                        "surface redistributes the reflected light, it does not add any")
    p.add_argument("--model", default="Ruicheng/moge-2-vitl-normal")
    p.add_argument("--out", default="runs/photo/out")
    args = p.parse_args()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    img = ImageOps.exif_transpose(Image.open(args.image)).convert("RGB")
    if args.crop:
        img = img.crop(args.crop)
    h = round(args.width * img.height / img.width / 14) * 14
    w = args.width // 14 * 14
    x = torch.from_numpy(np.asarray(img.resize((w, h), Image.LANCZOS))).float().permute(2, 0, 1) / 255
    if args.mask:
        m = torch.from_numpy(np.asarray(Image.open(args.mask).convert("L").resize((w, h)))) > 127
    elif args.seed:
        from scripts.photo_suppress import object_mask
        m = object_mask(x, (int(args.seed[0] * w), int(args.seed[1] * h)),
                        [int(args.box[0] * w), int(args.box[1] * h), int(args.box[2] * w), int(args.box[3] * h)], args.tol)
    else:
        m = hue_mask(x, *args.hue, args.min_sat, args.min_value)

    sys.path.insert(0, "external/MoGe")
    from moge.model.v2 import MoGeModel
    model = MoGeModel.from_pretrained(args.model).cuda().eval()
    with torch.no_grad():
        pred = model.infer(x.cuda(), use_fp16=False)
    flip = torch.tensor([1.0, -1.0, -1.0])  # OpenCV camera (y down, z forward) -> x right, y up, z towards the viewer
    n = F.normalize(torch.nan_to_num(pred["normal"].cpu()) * flip, dim=-1)
    v = F.normalize(-torch.nan_to_num(pred["points"].cpu(), nan=0.0, posinf=0.0, neginf=0.0) * flip, dim=-1)
    v = torch.where(v.norm(dim=-1, keepdim=True) > 0.5, v, torch.tensor([0.0, 0.0, 1.0]).expand_as(v))
    sel = m & (n.norm(dim=-1) > 0.5)

    lin = srgb_to_linear(x)
    y = (lin * LUM[0]).sum(0)
    light, r2 = light_direction(n, y, sel)
    print(f"object pixels {int(sel.sum())}; shading fit R2 {r2:.2f}; light (right, up, towards viewer) "
          + " ".join(f"{t:+.2f}" for t in light.tolist()))

    results = []
    for r in args.roughness:
        layer, diff, info = physical_layer(n, v, y, sel, r, albedo=args.albedo)
        wide, narrow = info["wide"] * sel, info["narrow"] * sel
        if args.sheen == "white":  # the whole layer is white light added to the image
            body, white = lin * torch.where(sel, diff, torch.ones_like(diff))[None], wide + narrow
        else:  # the sheen changes luminance only, so the body keeps its chromaticity; highlights stay white
            body, white = lin * torch.where(sel, 1 + wide / y.clamp_min(1e-4), torch.ones_like(y))[None], narrow
        if args.keep_mean:
            k = ((y[sel].mean() - white[sel].clamp(max=1).mean()) / (body * LUM[0]).sum(0)[sel].mean()).clamp(0.2, 1)
            body = torch.where(sel[None], body * k, body)
        res = linear_to_srgb(body + white[None])
        Image.fromarray((res.permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)).save(f"{out}_r{r}.png")
        results.append((r, res, layer * sel))
    Image.fromarray((x.permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)).save(f"{out}_input.png")

    panels = [("input", x.permute(1, 2, 0).numpy()), ("object region", m.numpy()),
              ("estimated normals", ((n * 0.5 + 0.5) * sel[..., None]).numpy()),
              (f"specular layer (roughness {results[0][0]})", results[0][2].numpy())]
    panels += [(f"+ specular, roughness {r}", res.permute(1, 2, 0).numpy()) for r, res, _ in results]
    fig, ax = plt.subplots(2, 3, figsize=(15, 10 * h / w))
    for a, (title, im) in zip(ax.flat, panels):
        a.imshow(im, cmap="magma" if im.ndim == 2 and im.dtype != bool else ("gray" if im.ndim == 2 else None),
                 vmax=float(np.quantile(im, 0.995)) if im.ndim == 2 and im.dtype != bool else None)
        a.set_title(title, fontsize=10)
        a.set_xticks([])
        a.set_yticks([])
    plt.tight_layout()
    plt.savefig(f"{out}_overview.png", dpi=80)


if __name__ == "__main__":
    main()
