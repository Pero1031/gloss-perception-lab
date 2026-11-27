"""Separate the specular part of a glossy object without a trained network, to suppress gloss and to tell how glossy
an object is.

  two-sided, one-sided
             Diffuse shading is a smooth function of the surface normal (second-order spherical harmonics, 9 terms,
             Ramamoorthi & Hanrahan 2001). It is fitted to the luminance with outliers down-weighted; what exceeds the
             fit is the specular part. Needs normals and a uniform albedo. See shading_specular for the two variants.
  colour     Dichromatic model: a pixel is body colour * d + white * s. The body colour is the object's most saturated
             chromaticity; s follows per pixel. Needs no normals, and a coloured dielectric under white light.
  gated      Judge first, then edit: the specular share of the object's light under the two-sided fit is the amount of
             gloss. Above GLOSSY the object is glossy and the one-sided estimate is removed; otherwise nothing is.
Scored on renders with a known specular pass (sets A and B of gloss_edit_eval.py): error of the suppressed image
against the true diffuse image (RMSE x100, sRGB), and the share on matte against non-matte images.
Writes runs/gloss_edit/separation.txt and separation.png.
"""
import glob
import json
from pathlib import Path

import matplotlib
import numpy as np
import torch
import torch.nn.functional as F

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scripts.gloss_edit import LUM, linear_to_srgb, masked_blur
from scripts.highlight_synthesis import geometry

EXPOSURE = 0.8
METHODS = ["two-sided", "one-sided", "colour", "gated"]
GLOSSY = 0.02  # specular share (two-sided fit) above which an object is treated as glossy


def load(path):
    d = np.load(path)
    rgb = torch.from_numpy(d["L_rgb"].astype(np.float32)).permute(2, 0, 1)
    mask = torch.from_numpy(d["L_mask"])
    glossy = torch.from_numpy(d["L_glossy"].astype(np.float32))
    y = (rgb * LUM[0]).sum(0)
    scale = EXPOSURE / torch.quantile(y[mask], 0.995).clamp_min(1e-6)
    p = json.loads(str(d["params"]))
    if isinstance(p["shape"], str):
        shape, kind = p["shape"], p["material"]
    else:
        fine = any(layer[0] == "fine" for layer in p["shape"].get("layers", []))
        shape, kind = ("relief" if fine else "smooth"), p["material"]["kind"]
    c = dict(lin=rgb * scale, mask=mask, glossy=glossy * scale, shape=shape, kind=kind, name=Path(path).stem,
             matte=float(glossy[mask].max()) <= 0, pos=torch.from_numpy(d["L_position"].astype(np.float32)),
             view=torch.from_numpy(d["L_view"].astype(np.float32)))
    return c


def sh_basis(n):
    x, y, z = n[..., 0], n[..., 1], n[..., 2]
    return torch.stack([torch.ones_like(x), x, y, z, x * y, x * z, y * z, x * x - y * y, 3 * z * z - 1], -1)


def shading_specular(y, n, sel, two_sided, rounds=20, down=0.02):
    """Specular luminance as the excess over a diffuse shading fitted to the normals (9 harmonics, weighted least squares).

    After each round, outlying pixels get the small weight `down`.
    two_sided=True: pixels far from the fit on either side. Brighter ones are specular light, darker ones are cast
      shadows and creases, which the harmonics cannot describe. Conservative: a matte object gives almost nothing.
    two_sided=False: only pixels brighter than the fit, so the fit settles on the lower envelope. Removes more of a
      glossy object's reflection, and takes shading for gloss on a matte object with relief."""
    B, t = sh_basis(n)[sel], y[sel]
    w = torch.ones_like(t)
    for _ in range(rounds):
        sw = w.sqrt()[:, None]
        coef = torch.linalg.lstsq(B * sw, (t * sw[:, 0])[:, None]).solution[:, 0]
        e = t - B @ coef
        if two_sided:
            scale = (e - e.median()).abs().median().clamp_min(1e-4) * 1.48
            out = e.abs() > 2 * scale
        else:
            scale = e[e < 0].abs().median().clamp_min(1e-4)
            out = e > 2 * scale
        w = torch.where(out, torch.full_like(t, down), torch.ones_like(t))
    fit = (sh_basis(n) @ coef).clamp_min(0)
    return ((y - fit - 2 * scale).clamp_min(0)) * sel, fit


def colour_specular(lin, sel, white=(1.0, 1.0, 1.0)):
    """Specular luminance from the dichromatic model. Returns (s, how chromatic the body is: 0 = grey)."""
    w = torch.tensor(white).view(3, 1, 1)
    ysum = lin.sum(0).clamp_min(1e-6)
    chroma = lin / ysum
    sat = 1 - 3 * chroma.amin(0)
    lum = (lin * LUM[0]).sum(0)
    ok = sel & (lum > torch.quantile(lum[sel], 0.2))
    thr = torch.quantile(sat[ok], 0.9)
    body = F.normalize(lin[:, ok & (sat >= thr)].median(dim=1).values, dim=0)
    # lin = d * body + s * white, least squares per pixel
    A = torch.stack([body, w[:, 0, 0]], 1)
    sol = (torch.linalg.pinv(A) @ lin.flatten(1)).view(2, *lin.shape[1:])
    s = sol[1].clamp_min(0) * (w[:, 0, 0] * LUM[0, :, 0, 0]).sum()
    return s * sel, float(thr)


def rmse(a, b, m):
    return float(((a - b) ** 2).mean(0)[m].mean().sqrt()) * 100


def auc(pos, neg):
    v = np.concatenate([pos, neg])
    r = np.empty(len(v))
    r[np.argsort(v, kind="stable")] = np.arange(1, len(v) + 1)
    return (r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def fill_rim(spec, y, mask, sel, width=2):
    """Extend a specular estimate to the silhouette band, where normals are unreliable.

    The band is the object pixels within `width` px of the silhouette. Its diffuse luminance is taken from the
    suppressed interior next to it; what the band has above that is specular (Fresnel makes the rim the brightest
    part of the reflection, so leaving it out draws a white outline)."""
    inner = sel & (-F.max_pool2d(-sel.float()[None, None], 2 * width + 1, 1, width)[0, 0] > 0.5)
    band = mask & ~inner
    w = inner.float()[None, None]
    diffuse = masked_blur((y - spec)[None, None] * w, w, 2.0)[0, 0]
    return torch.where(band, (y - diffuse).clamp_min(0), spec * inner)


def separate(c, normals=None):
    """normals: (H, W, 3) estimated normals in camera space; None uses the true normals of the render."""
    n, v, sel = geometry(c)
    if normals is not None:
        n, sel = normals, c["mask"] & (normals.norm(dim=-1) > 0.5)
    lin = c["lin"].clamp(0, 1)
    y = (lin * LUM[0]).sum(0)
    out = {}
    out["two-sided"], _ = shading_specular(y, n, sel, True)
    out["one-sided"], _ = shading_specular(y, n, sel, False)
    out["colour"], _ = colour_specular(lin, sel)
    gloss = float(out["two-sided"][sel].sum() / y[sel].sum())  # the amount of gloss, from the conservative fit
    out["gated"] = fill_rim(out["one-sided"], y, c["mask"], sel) if gloss > GLOSSY else torch.zeros_like(y)
    out["gated, rim left"] = out["one-sided"] if gloss > GLOSSY else torch.zeros_like(y)
    return out, sel, y, gloss


def estimated_test(root="data/serrano_render_512"):
    """Same test with normals estimated from the image by MoGe-2 (dark matte and semi-glossy renders at 512 px)."""
    import sys
    sys.path.insert(0, "external/MoGe")
    from moge.model.v2 import MoGeModel
    model = MoGeModel.from_pretrained("Ruicheng/moge-2-vitl-normal").cuda().eval()
    flip = torch.tensor([1.0, -1.0, -1.0])
    rows, examples = {}, []
    for q in sorted(glob.glob(f"{root}/*.npz")):
        c = load(q)
        photo = torch.round(linear_to_srgb(c["lin"]) * 255) / 255
        with torch.no_grad():
            est = F.normalize(torch.nan_to_num(model.infer(photo.cuda(), use_fp16=False)["normal"].cpu()) * flip, dim=-1)
        true_diffuse = linear_to_srgb(c["lin"] - c["glossy"][None])
        for name, nrm in (("true normals", None), ("MoGe-2 normals", est)):
            s, sel, y, gloss = separate(c, nrm)
            inner = geometry(c)[2]
            key = (name, ("matte " if c["matte"] else "glossy ") + c["shape"])
            rows.setdefault(key, []).append([rmse(linear_to_srgb(c["lin"]), true_diffuse, inner),
                                             rmse(linear_to_srgb(c["lin"] - s["gated"][None]), true_diffuse, inner),
                                             gloss, float(gloss > GLOSSY)])
            if name.startswith("MoGe") and "cambridge" in c["name"]:
                examples.append((c, s, sel))
    out_dir = Path("runs/gloss_edit")
    with open(out_dir / "separation_estimated.txt", "w") as f:
        print(f"{'normals':16s}{'group':26s}{'no edit':>9s}{'gated':>9s}{'share':>9s}{'judged glossy':>15s}", file=f)
        for (name, g), v_ in sorted(rows.items()):
            m = np.mean(v_, 0)
            print(f"{name:16s}{g:26s}{m[0]:9.2f}{m[1]:9.2f}{m[2]:9.3f}{int(np.sum([r[3] for r in v_])):>10d} of {len(v_)}", file=f)
    print(open(out_dir / "separation_estimated.txt").read())
    titles = ["input", "specular, one-sided fit", "true specular", "suppressed (judge, then edit)", "true diffuse"]
    fig, ax = plt.subplots(len(examples), len(titles), figsize=(2.6 * len(titles), 2.7 * len(examples)))
    for r, (c, s, sel) in enumerate(examples):
        img = lambda lin: linear_to_srgb(lin).permute(1, 2, 0).numpy()
        vmax = float(torch.quantile(c["glossy"][sel], 0.99)) + 0.05
        ims = [img(c["lin"]), s["one-sided"], c["glossy"] * sel, img(c["lin"] - s["gated"][None]), img(c["lin"] - c["glossy"][None])]
        for k, im in enumerate(ims):
            ax[r, k].imshow(np.asarray(im), **(dict(cmap="magma", vmin=0, vmax=vmax) if im.ndim == 2 else {}))
            ax[r, k].set_xticks([])
            ax[r, k].set_yticks([])
            if r == 0:
                ax[r, k].set_title(titles[k], fontsize=8)
        ax[r, 0].set_ylabel(f"{c['shape'][:9]}\n{'matte' if c['matte'] else 'glossy'}", fontsize=8)
    plt.tight_layout()
    plt.savefig(out_dir / "separation_estimated.png", dpi=75)


def main():
    out_dir = Path("runs/gloss_edit")
    rng = np.random.default_rng(0)
    held = sorted(glob.glob("data/stereo_ds/scene_*.npz"))[-1000:]
    sets = {"A: Serrano geometries": sorted(glob.glob("data/serrano_render/*.npz")),
            "B: random scenes": [held[i] for i in rng.choice(len(held), 300, replace=False)]}
    examples = []
    with open(out_dir / "separation.txt", "w") as f:
        for name, paths in sets.items():
            err, share, judged = {}, {}, []
            for q in paths:
                c = load(q)
                s, sel, y, gloss = separate(c)
                true_diffuse = linear_to_srgb(c["lin"] - c["glossy"][None])
                group = ("matte " if c["matte"] else "glossy ") + c["shape"]
                e = {"none": rmse(linear_to_srgb(c["lin"]), true_diffuse, sel)}
                for k in METHODS:
                    e[k] = rmse(linear_to_srgb(c["lin"] - s[k][None]), true_diffuse, sel)
                    share.setdefault(k, []).append((c["matte"], float(s[k][sel].sum() / y[sel].sum())))
                judged.append((c["matte"], gloss > GLOSSY))
                share.setdefault("true", []).append((c["matte"], float(c["glossy"][sel].sum() / (c["lin"] * LUM[0]).sum(0)[sel].sum())))
                for k, v_ in e.items():
                    err.setdefault(group, {}).setdefault(k, []).append(v_)
                if name.startswith("A") and "cambridge" in c["name"] and ("glossy_blue" in c["name"] or "matte_dark" in c["name"]):
                    examples.append((c, s, sel))
            print(f"\n== {name}: suppression error against the true diffuse image (RMSE x100) ==", file=f)
            print(f"{'group':26s}" + "".join(f"{k:>10s}" for k in ["none"] + METHODS), file=f)
            for g in sorted(err):
                print(f"{g:26s}" + "".join(f"{np.mean(err[g][k]):10.2f}" for k in ["none"] + METHODS), file=f)
            print(f"\n== {name}: specular share of the object's light (median; matte / non-matte) and AUC for telling them apart ==",
                  file=f)
            for k in METHODS + ["true"]:
                m = np.array([v_ for is_m, v_ in share[k] if is_m])
                g = np.array([v_ for is_m, v_ in share[k] if not is_m])
                print(f"{k:10s} matte {np.median(m):.3f}   non-matte {np.median(g):.3f}   AUC {auc(g, m):.3f}", file=f)
            jm = [j for is_m, j in judged if is_m]
            jg = [j for is_m, j in judged if not is_m]
            print(f"judged glossy at share > {GLOSSY}: {sum(jm)} of {len(jm)} matte images, {sum(jg)} of {len(jg)} non-matte images",
                  file=f)
    print(open(out_dir / "separation.txt").read())

    titles = ["input", "specular, two-sided fit", "specular, one-sided fit", "true specular",
              "suppressed (judge, then edit)", "true diffuse"]
    fig, ax = plt.subplots(len(examples), len(titles), figsize=(2.5 * len(titles), 2.6 * len(examples)))
    for r, (c, s, sel) in enumerate(examples):
        img = lambda lin: linear_to_srgb(lin).permute(1, 2, 0).numpy()
        vmax = float(torch.quantile(c["glossy"][sel], 0.99)) + 0.05
        ims = [img(c["lin"]), s["two-sided"], s["one-sided"], c["glossy"] * sel, img(c["lin"] - s["gated"][None]),
               img(c["lin"] - c["glossy"][None])]
        for k, im in enumerate(ims):
            if im.ndim == 2:
                ax[r, k].imshow(np.asarray(im), cmap="magma", vmin=0, vmax=vmax)
            else:
                ax[r, k].imshow(im)
            ax[r, k].set_xticks([])
            ax[r, k].set_yticks([])
            if r == 0:
                ax[r, k].set_title(titles[k], fontsize=8)
        ax[r, 0].set_ylabel(f"{c['shape'][:9]}\n{'matte' if c['matte'] else 'glossy'}", fontsize=8)
    plt.tight_layout()
    plt.savefig(out_dir / "separation.png", dpi=85)


if __name__ == "__main__":
    import sys
    estimated_test() if "--estimated" in sys.argv else main()
