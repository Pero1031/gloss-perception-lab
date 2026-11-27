"""Adding gloss to a matte object: where should the specular layer go?

For each Serrano geometry and light, a dark matte render is the input, and the specular (glossy) pass of the
semi-glossy and glossy renders of the same shape and light is the truth (their diffuse colour is close to the input's).
Specular layers (all on object pixels):
  shading       the brightest part of the smoothed diffuse shading, (Y - t) / (max - t) above the 90th percentile, squared
  P             gloss map of Nagata et al. on the matte image
  normals       Blinn-Phong lobe (n.h)^k from the true normals (from the position pass), the view direction, and one
                light direction fitted to the shading by least squares (Lambert)
  graded        Fresnel * (narrow + wide lobe at the mirror direction of the estimated light) * occlusion:
                highlights plus a sheen over the lit side, without any assumed sky.
  N% sharp/soft no reflection of the actual surroundings: the specular reflection of one white light of finite size
                by the surface (GGX lobe, Fresnel), from the normals and the most probable light direction given the
                shading and the prior that light tends to come from above. The layer is specified by the gloss cues:
                coverage (met by the source size), contrast against the body, and sharpness of the border.
Scores: Spearman correlation with the true specular luminance, and how much of the top 5% of each layer lies in the
true top 5% (chance 0.05). Writes runs/gloss_edit/synthesis.txt and synthesis.png.
"""
import glob
from pathlib import Path

import matplotlib
import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scripts.gloss_edit import LUM, blur, linear_to_srgb
from src.models.features import GlossFeatures

SHAPES = ["sphere", "bunny_vt", "happy_vrip_vt", "xyzrgb_dragon_vt"]
INPUT = "matte_dark"
EXPOSURE = 0.5  # the matte object's 99.5th luminance percentile; leaves headroom for the added layer
CUE_SETS = {"3% sharp": (0.03, 0.08), "3% soft": (0.03, 0.3), "8% sharp": (0.08, 0.08)}  # coverage, roughness
GRADED = {"graded semi": 0.25, "graded glossy": 0.08}  # roughness
STRENGTH = 0.5  # mean of the graded layer relative to the mean luminance of the matte object (specular share 1/3,
                # as in the real semi-glossy and glossy renders: 0.29-0.40)
PHYSICAL = {"physical semi": 0.25, "physical glossy": 0.1}  # roughness
METHODS = ["shading", "P", "normals k=30"] + list(CUE_SETS) + list(GRADED) + list(PHYSICAL)
UP = torch.tensor([0.0, 1.0, 0.0])  # camera space: light comes from above in the image
F0 = 0.04  # normal-incidence reflectance of a dielectric


def load(shape, material, light, scale=None):
    d = np.load(glob.glob(f"data/serrano_render/{shape}__{material}__{light}*.npz")[0])
    rgb = torch.from_numpy(d["L_rgb"].astype(np.float32)).permute(2, 0, 1)
    mask = torch.from_numpy(d["L_mask"])
    y = (rgb * LUM[0]).sum(0)
    if scale is None:
        scale = EXPOSURE / torch.quantile(y[mask], 0.995).clamp_min(1e-6)
    view = torch.from_numpy(d["L_view"].astype(np.float32))
    return dict(lin=rgb * scale, mask=mask, glossy=torch.from_numpy(d["L_glossy"].astype(np.float32)) * scale,
                pos=torch.from_numpy(d["L_position"].astype(np.float32)), view=view, scale=scale)


def geometry(c):
    """Unit normals and view directions in camera space (x right, y up, z towards the viewer)."""
    pos, mask = c["pos"], c["mask"]
    rot, cam = c["view"][:3, :3], torch.linalg.inv(c["view"])[:3, 3]
    dx = pos[:, 2:] - pos[:, :-2]
    dy = pos[2:] - pos[:-2]
    n = torch.zeros_like(pos)
    n[1:-1, 1:-1] = torch.cross(dx[1:-1], dy[:, 1:-1], dim=-1)
    n = F.normalize(n, dim=-1)
    v = F.normalize(cam - pos, dim=-1)
    n = torch.where(((n * v).sum(-1, keepdim=True) < 0), -n, n)
    inner = -F.max_pool2d(-mask.float()[None, None], 3, 1, 1)[0, 0] > 0.5  # differences across the silhouette are wrong
    return n @ rot.T, v @ rot.T, mask & inner


def smoothstep(x, lo, hi):
    t = ((x - lo) / (hi - lo)).clamp(0, 1)
    return t * t * (3 - 2 * t)


def light_direction(n, y, sel, kappa_prior=2.0):
    """Most probable light direction given the shading and the prior that light tends to come from above.

    The shading is fitted with Y ~ n.L + b. The fit and the prior are treated as von Mises-Fisher distributions on the
    sphere, whose product is again one with mean direction kappa_fit * l_fit + kappa_prior * up. kappa_fit grows
    with how well a single direction explains the shading (R^2 / (1 - R^2)), so a clear shading gradient overrides
    the prior and an ambiguous one falls back on it."""
    A = torch.cat([n[sel], torch.ones(int(sel.sum()), 1)], 1)
    sol = torch.linalg.lstsq(A, y[sel][:, None]).solution[:, 0]
    r2 = 1 - ((A @ sol - y[sel]) ** 2).sum() / ((y[sel] - y[sel].mean()) ** 2).sum()
    kappa_fit = 10 * r2 / (1 - r2).clamp_min(1e-3)
    return F.normalize(kappa_fit * F.normalize(sol[:3], dim=0) + kappa_prior * UP, dim=0), float(r2)


def shape_layer(n, v, y, sel, roughness, source=0.0):
    """Specular reflection of the estimated light by the surface: a microfacet (GGX) lobe evaluated at every pixel.

    Follows the shape continuously through the angle between n and h (h: half vector of light and view).
    `roughness` sets how sharp the border of the highlight is. `source` (degrees) is the angular radius of the light:
    inside it the highlight has the full strength of a mirror image of the source, so a larger source gives a larger
    highlight that is still bright. Scaled by Fresnel reflectance, by n.l and by a shading-based occlusion.
    The plateau of the lobe is 1 (before Fresnel)."""
    light, _ = light_direction(n, y, sel)
    nl = (n * light).sum(-1).clamp(0, 1)
    h = F.normalize(light + v, dim=-1)
    ang = torch.acos((n * h).sum(-1).clamp(-1, 1))
    ang = (ang - np.deg2rad(source) / 2).clamp_min(0)  # the half vector turns half as far as the mirror direction
    a2 = roughness**4
    ggx = (a2 / (torch.cos(ang) ** 2 * (a2 - 1) + 1)) ** 2  # GGX normalised to 1 at the centre
    fresnel = (F0 + (1 - F0) * (1 - (v * h).sum(-1).clamp(0, 1)) ** 5) / F0
    ys = blur(y[None, None], 1.5)[0, 0]
    occl = (ys / torch.quantile(ys[sel], 0.5).clamp_min(1e-6)).clamp(0, 1) ** 0.5
    return ggx * fresnel.clamp(max=3) * (nl > 0) * occl, light


def cue_layer(n, v, y, sel, coverage, roughness, contrast=10.0):
    """Specular layer with a requested coverage, contrast and sharpness (the cues of Marlow, Kim & Anderson).

    coverage: share of the object (0-1) where the layer exceeds half its plateau; met by choosing the source size.
    contrast: luminance the plateau adds, in units of the object's median luminance. roughness: border sharpness."""
    lo, hi = 0.0, 120.0
    for _ in range(12):
        mid = (lo + hi) / 2
        cov = float(((shape_layer(n, v, y, sel, roughness, mid)[0] > 0.5) & sel).sum()) / float(sel.sum())
        lo, hi = (mid, hi) if cov < coverage else (lo, mid)
    return shape_layer(n, v, y, sel, roughness, hi)[0] * contrast * torch.median(y[sel]), hi


def graded_layer(n, v, y, sel, roughness):
    """Specular layer whose strength varies continuously with the orientation of the surface.

    Fresnel(n.v) * (narrow lobe + wide lobe, both around the mirror direction of the estimated light) *
    shading-based occlusion. Nothing of the surroundings is reflected and no sky is assumed. The wide lobe has no
    edge: it is brightest towards the light and falls to zero opposite to it, so the whole lit side gets a sheen and
    the shading is accentuated. Highlights alone leave the rest of the surface at its matte luminance and read as
    white paint; with the sheen they read as highlights on a smooth surface."""
    light, _ = light_direction(n, y, sel)
    nv = (n * v).sum(-1).clamp(0, 1)
    r = 2 * nv[..., None] * n - v  # mirror direction of the view
    k = 2 / roughness**4 - 2
    lobe = (r * light).sum(-1).clamp(0, 1) ** (k / 4) * (k / 4 + 1) / 8
    fresnel = F0 + (1 - F0) * (1 - nv) ** 5
    ys = blur(y[None, None], 1.5)[0, 0]
    occl = (ys / torch.quantile(ys[sel], 0.5).clamp_min(1e-6)).clamp(0, 1) ** 0.5
    wide = ((r * light).sum(-1) * 0.5 + 0.5) ** 3
    return fresnel * (lobe + wide) * occl


def physical_layer(n, v, y, sel, roughness, albedo=0.1, compact=0.5):
    """Specular layer computed from the lighting that the shading fit implies. Returns (layer, diffuse factor, info).

    The fit Y ~ n.L + b is the diffuse reflection (albedo rho) of lighting with a uniform part and a part that
    grows towards the light: radiance a0 + a1 (w.l) gives Y = rho a0 + (2 rho / 3) a1 (n.l), so a0 = b / rho and
    a1 = 1.5 |L| / rho. A share `compact` of the directional part is taken to come from one compact source of
    irradiance pi |L| compact / rho; the rest stays in a1.
      wide:    a mirror shows the lighting itself, F(n.v) * max(0, a0 + a1 (r.l))
      narrow:  the compact source through a GGX lobe, F(v.h) * D(n.h) / (4 n.v) * irradiance
    Both are scaled by the shading-based occlusion. The diffuse image should be multiplied by 1 - F(n.v).
    Two values cannot come from the image: the albedo (it sets the overall strength) and `compact`."""
    A = torch.cat([n[sel], torch.ones(int(sel.sum()), 1)], 1)
    sol = torch.linalg.lstsq(A, y[sel][:, None]).solution[:, 0]
    L, b = float(sol[:3].norm()), float(sol[3].clamp_min(1e-6))
    light, r2 = light_direction(n, y, sel)
    nv = (n * v).sum(-1).clamp(1e-3, 1)
    r = 2 * nv[..., None] * n - v
    h = F.normalize(light + v, dim=-1)
    schlick = lambda c: F0 + (1 - F0) * (1 - c.clamp(0, 1)) ** 5
    wide = schlick(nv) * (b + 1.5 * (1 - compact) * L * (r * light).sum(-1)).clamp_min(0) / albedo
    a2 = roughness**4
    ggx = a2 / (np.pi * ((n * h).sum(-1).clamp(0, 1) ** 2 * (a2 - 1) + 1) ** 2)
    lit = ((n * light).sum(-1) > 0).float()
    narrow = schlick((v * h).sum(-1)) * ggx / (4 * nv) * np.pi * compact * L / albedo * lit
    ys = blur(y[None, None], 1.5)[0, 0]
    occl = (ys / torch.quantile(ys[sel], 0.5).clamp_min(1e-6)).clamp(0, 1) ** 0.5
    return (wide + narrow) * occl, 1 - schlick(nv), dict(directional_over_uniform=L / b, r2=r2, wide=wide * occl,
                                                         narrow=narrow * occl)


def layers(c, feat):
    y = (c["lin"].clamp(0, 1) * LUM[0]).sum(0)
    n, v, sel = geometry(c)
    ys = blur(y[None, None], 2.0)[0, 0]
    t, top = torch.quantile(ys[sel], 0.90), ys[sel].max()
    out = {"shading": ((ys - t) / (top - t).clamp_min(1e-6)).clamp(0, 1) ** 2}
    x = torch.round(linear_to_srgb(c["lin"]) * 255) / 255
    out["P"] = feat.gloss_map(x[None].cuda())[0].cpu()
    A = torch.cat([n[sel], torch.ones(int(sel.sum()), 1)], 1)
    light = F.normalize(torch.linalg.lstsq(A, y[sel][:, None]).solution[:3, 0], dim=0)
    out["normals k=30"] = (n * F.normalize(light + v, dim=-1)).sum(-1).clamp(0, 1) ** 30
    for name, (cov, r) in CUE_SETS.items():
        out[name], _ = cue_layer(n, v, y, sel, cov, r)
    for name, r in GRADED.items():
        out[name] = graded_layer(n, v, y, sel, r)
    for name, r in PHYSICAL.items():
        out[name], out[name + " diffuse"], out[name + " info"] = physical_layer(n, v, y, sel, r)
    return {k: (v_ if isinstance(v_, dict) else v_ * sel) for k, v_ in out.items()}, sel


def score(pred, true, sel):
    p, t = pred[sel].numpy(), true[sel].numpy()
    rho = spearmanr(p, t)[0] if p.std() > 0 else np.nan
    n = max(1, int(0.05 * len(p)))
    top_p, top_t = np.argsort(-p, kind="stable")[:n], set(np.argsort(-t, kind="stable")[:n].tolist())
    return rho, np.mean([i in top_t for i in top_p])


def main():
    out = Path("runs/gloss_edit")
    out.mkdir(parents=True, exist_ok=True)
    feat = GlossFeatures(32, sigma_step=1).cuda()
    lights = sorted({Path(f).stem.split("__")[2] for f in glob.glob("data/serrano_render/*.npz")})
    res, examples, fits, phys_rows = {}, [], [], []
    for shape in SHAPES:
        for light in lights:
            c = load(shape, INPUT, light)
            maps, sel = layers(c, feat)
            n_, _, sel_ = geometry(c)
            l_, r2 = light_direction(n_, (c["lin"].clamp(0, 1) * LUM[0]).sum(0), sel_)
            fits.append(f"{shape:18s}{light[:18]:20s} R2 {r2:.2f}  light (right, up, towards viewer) "
                        + " ".join(f"{t:+.2f}" for t in l_.tolist()))
            truth = {t: load(shape, t, light, c["scale"]) for t in ("semi_blue", "glossy_blue")}
            for target, t in truth.items():
                for k in METHODS:
                    res.setdefault((shape, target, k), []).append(score(maps[k], t["glossy"], sel))
            yobj = (c["lin"] * LUM[0]).sum(0)[sel]
            info = maps["physical semi info"]
            share = {k: float(maps[k][sel].sum() / (maps[k][sel].sum() + yobj.sum())) for k in PHYSICAL}
            real = {t: float(truth[t]["glossy"][sel].sum() / (truth[t]["lin"] * LUM[0]).sum(0)[sel].sum()) for t in truth}
            phys_rows.append(f"{shape:18s}{light[:18]:20s}{info['directional_over_uniform']:12.2f}"
                             + "".join(f"{share[k]:12.2f}" for k in PHYSICAL) + "".join(f"{real[t]:12.2f}" for t in truth))
            if light == lights[0]:
                examples.append((shape, c, maps, truth))
    with open(out / "synthesis.txt", "w") as f:
        for target, lab in (("semi_blue", "semi-glossy (roughness 0.25)"), ("glossy_blue", "glossy (roughness 0.05)")):
            for j, what in enumerate(("Spearman correlation with the true specular luminance",
                                      "share of the layer's top 5% inside the true top 5% (chance 0.05)")):
                print(f"\n== {lab}: {what} ==", file=f)
                print(f"{'shape':20s}" + "".join(f"{k:>15s}" for k in METHODS), file=f)
                for shape in SHAPES:
                    print(f"{shape:20s}" + "".join(f"{np.nanmean([r[j] for r in res[(shape, target, k)]]):15.2f}"
                                                   for k in METHODS), file=f)
        print("\n== layer from the shading fit: directional / uniform light, and specular share of the object's light ==\n"
              + f"{'shape':18s}{'light':20s}{'dir/uniform':>12s}{'ours semi':>12s}{'ours glossy':>12s}{'real semi':>12s}"
              + f"{'real glossy':>12s}\n" + "\n".join(phys_rows), file=f)
        print("\n== light direction: shading fit combined with the prior (camera space) ==\n" + "\n".join(fits), file=f)
    print(open(out / "synthesis.txt").read())

    titles = ["matte input", "hand-set layer (semi)", "from the shading fit (semi)", "real semi-glossy render",
              "from the shading fit (glossy)", "real glossy render", "layer from the fit (semi)", "true specular (semi)"]
    fig, ax = plt.subplots(len(examples), len(titles), figsize=(2.6 * len(titles), 2.7 * len(examples)))
    for r, (shape, c, maps, truth) in enumerate(examples):
        def add(m):
            return linear_to_srgb(c["lin"] + m[None]).permute(1, 2, 0).numpy()
        show = lambda t: linear_to_srgb(t["lin"]).permute(1, 2, 0).numpy()
        ym = (c["lin"] * LUM[0]).sum(0)[c["mask"]].mean()
        graded = {k: maps[k] * STRENGTH * ym / maps[k][c["mask"]].mean().clamp_min(1e-9) for k in GRADED}
        def phys(k):  # diffuse reduced by what is reflected, plus the layer; elsewhere unchanged
            d = torch.where(maps[k + " diffuse"] > 0, maps[k + " diffuse"], torch.ones_like(maps[k]))
            return linear_to_srgb(c["lin"] * d[None] + maps[k][None]).permute(1, 2, 0).numpy()
        ims = [show(c), add(graded["graded semi"]), phys("physical semi"), show(truth["semi_blue"]),
               phys("physical glossy"), show(truth["glossy_blue"]), maps["physical semi"],
               truth["semi_blue"]["glossy"] * c["mask"]]
        for k, im in enumerate(ims):
            if im.ndim == 2:
                ax[r, k].imshow(np.asarray(im), cmap="magma", vmin=0, vmax=float(np.quantile(np.asarray(im), 0.995)) + 1e-6)
            else:
                ax[r, k].imshow(im)
            ax[r, k].set_xticks([])
            ax[r, k].set_yticks([])
            if r == 0:
                ax[r, k].set_title(titles[k], fontsize=8)
        ax[r, 0].set_ylabel(shape[:9], fontsize=9)
    plt.tight_layout()
    plt.savefig(out / "synthesis.png", dpi=90)


if __name__ == "__main__":
    main()
