"""Judge how glossy an object in a photograph is, and suppress its gloss.

The object is picked with a seed point: pixels around it with a similar chromaticity, with enclosed holes filled
(highlights are whiter than the body and would otherwise drop out). Normals come from MoGe-2 run on the whole image.
The specular part is what exceeds a diffuse shading fitted to the normals (specular_separation.py): the two-sided
fit gives the amount of gloss, and if that says glossy, the one-sided estimate is removed and the gap takes the
colour of the surrounding body.

  python -m scripts.photo_suppress --image photo.jpg --seed 0.58 0.59 --box 0.45 0.40 0.85 0.75 --out runs/photo/name
(seed and box are fractions of the image width and height; the box limits the search for the object.)
"""
import argparse
import sys
from pathlib import Path

import matplotlib
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageOps
from scipy import ndimage

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scripts.gloss_edit import LUM, blur, linear_to_srgb, masked_blur, srgb_to_linear
from scripts.specular_separation import GLOSSY, fill_rim, shading_specular


def object_mask(x, seed, box, tol, close=6):
    """x: (3,H,W) sRGB. Region around `seed` (x, y in pixels) whose chromaticity is within `tol` of the seed's."""
    h, w = x.shape[1:]
    lin = srgb_to_linear(x)
    chroma = (lin / lin.sum(0).clamp_min(1e-4))[:2]
    sx, sy = seed
    ref = chroma[:, max(sy - 4, 0):sy + 5, max(sx - 4, 0):sx + 5].flatten(1).median(dim=1).values
    m = ((chroma - ref[:, None, None]).norm(dim=0) < tol)
    inbox = torch.zeros_like(m)
    inbox[box[1]:box[3], box[0]:box[2]] = True
    m = (m & inbox).numpy()
    m = ndimage.binary_closing(m, iterations=close)
    m = ndimage.binary_fill_holes(m)
    m = ndimage.binary_opening(m, iterations=2)
    lab, _ = ndimage.label(m)
    return torch.from_numpy(lab == lab[sy, sx]) if lab[sy, sx] else torch.from_numpy(m)


def remove(lin, y, spec, mask, sigma=6.0):
    """Image without the specular part: diffuse luminance (y - spec) in the colour of the body.

    Subtracting white light leaves a tint when the light is not white, and breaks on clipped pixels. Instead the
    pixel keeps its own chromaticity where little was removed and takes the chromaticity of nearby body pixels (those
    with a small specular fraction) where much was."""
    frac = (spec / y.clamp_min(1e-4)).clamp(0, 1)
    own = lin / y.clamp_min(1e-4)
    w = (mask & (frac < 0.1)).float()[None, None]
    body = masked_blur(own[None] * w, w, sigma)[0]
    support = blur(w, sigma)[0, 0]
    overall = own[:, w[0, 0] > 0].median(dim=1).values  # body colour of the whole object, where no body pixel is near
    body = torch.where(support > 0.05, body, overall[:, None, None].expand_as(body))
    g = (3 * frac).clamp(0, 1)
    return (y - spec).clamp_min(0) * ((1 - g) * own + g * body)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--image", required=True)
    p.add_argument("--seed", type=float, nargs=2, help="a point on the object, as fractions of width and height")
    p.add_argument("--hue", type=float, nargs=2, help="hue range (0-1) of the object, instead of a seed point")
    p.add_argument("--box", type=float, nargs=4, default=[0, 0, 1, 1], help="x0 y0 x1 y1 as fractions; limits the object")
    p.add_argument("--tol", type=float, default=0.06, help="chromaticity distance that still counts as the object")
    p.add_argument("--size", type=int, default=1022, help="long side of the working image")
    p.add_argument("--model", default="Ruicheng/moge-2-vitl-normal")
    p.add_argument("--force", action="store_true", help="remove the estimate even if the object is judged matte")
    p.add_argument("--out", default="runs/photo/out")
    args = p.parse_args()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    img = ImageOps.exif_transpose(Image.open(args.image)).convert("RGB")
    k = args.size / max(img.size)
    w, h = (round(img.width * k / 14) * 14, round(img.height * k / 14) * 14)
    x = torch.from_numpy(np.asarray(img.resize((w, h), Image.LANCZOS)).copy()).float().permute(2, 0, 1) / 255
    box = [int(args.box[0] * w), int(args.box[1] * h), int(args.box[2] * w), int(args.box[3] * h)]
    if args.hue:
        from scripts.photo_gloss import hue_mask
        mask = hue_mask(x, *args.hue)
    else:
        mask = object_mask(x, (int(args.seed[0] * w), int(args.seed[1] * h)), box, args.tol)

    sys.path.insert(0, "external/MoGe")
    from moge.model.v2 import MoGeModel
    model = MoGeModel.from_pretrained(args.model).cuda().eval()
    with torch.no_grad():
        pred = model.infer(x.cuda(), use_fp16=False)
    n = F.normalize(torch.nan_to_num(pred["normal"].cpu()) * torch.tensor([1.0, -1.0, -1.0]), dim=-1)
    sel = mask & (n.norm(dim=-1) > 0.5)

    lin = srgb_to_linear(x)
    y = (lin * LUM[0]).sum(0)
    two, _ = shading_specular(y, n, sel, True)
    one, _ = shading_specular(y, n, sel, False)
    gloss = float(two[sel].sum() / y[sel].sum())
    glossy = gloss > GLOSSY
    spec = fill_rim(one, y, mask, sel) if (glossy or args.force) else torch.zeros_like(y)
    res = linear_to_srgb(torch.where(mask[None], remove(lin, y, spec, mask), lin))
    print(f"{Path(args.image).name}: object pixels {int(sel.sum())}; specular share {gloss:.3f} -> judged "
          f"{'glossy' if glossy else 'matte'}; removed share {float(spec[sel].sum() / y[sel].sum()):.3f}")

    ys, xs = np.nonzero(mask.numpy())
    pad = 30
    y0, y1, x0, x1 = max(ys.min() - pad, 0), min(ys.max() + pad, h), max(xs.min() - pad, 0), min(xs.max() + pad, w)
    crop = lambda t: t[..., y0:y1, x0:x1]
    to_img = lambda t: (t.permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)
    Image.fromarray(to_img(crop(x))).save(f"{out}_input.png")
    Image.fromarray(to_img(crop(res))).save(f"{out}_suppressed.png")
    panels = [("input", crop(x).permute(1, 2, 0).numpy()), ("object region", crop(mask).numpy()),
              ("estimated normals", (crop(((n * 0.5 + 0.5) * sel[..., None]).permute(2, 0, 1))).permute(1, 2, 0).numpy()),
              ("specular, two-sided fit", crop(two).numpy()), ("specular, one-sided fit", crop(one).numpy()),
              (f"suppressed (share {gloss:.3f}, judged {'glossy' if glossy else 'matte'})", crop(res).permute(1, 2, 0).numpy())]
    fig, ax = plt.subplots(2, 3, figsize=(15, 10 * (y1 - y0) / (x1 - x0)))
    vmax = float(torch.quantile(one[sel], 0.995)) + 1e-3
    for a, (title, im) in zip(ax.flat, panels):
        if im.ndim == 2 and im.dtype != bool:
            a.imshow(im, cmap="magma", vmin=0, vmax=vmax)
        else:
            a.imshow(im, cmap="gray" if im.ndim == 2 else None)
        a.set_title(title, fontsize=10)
        a.set_xticks([])
        a.set_yticks([])
    plt.tight_layout()
    plt.savefig(f"{out}_overview.png", dpi=70)


if __name__ == "__main__":
    main()
