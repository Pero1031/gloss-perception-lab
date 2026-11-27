"""Fit the 100 MERL measured BRDFs with Blender 4.5 Principled BSDF parameters (Cycles cannot use tabulated BRDFs).

    .\\.venv\\Scripts\\python.exe -I scripts\\merl_fit.py --merl-dir <MERL brdfs folder>
        [--out runs\\merl_fit] [--workers 4] [--n-cells 25000] [--materials chrome red-plastic ...] [--figure-only]

Model (re-implemented from the Cycles 4.5 kernel source shipped in <Blender>/4.5/scripts/addons_core/cycles/source:
svm/closure.h CLOSURE_BSDF_PRINCIPLED_ID, closure/bsdf_microfacet.h), isotropic, default "Multiscatter GGX":
  - specular: GGX, alpha = roughness^2, height-correlated Smith G, times the energy-compensation factor
    1 + Fms (1 - E(mu_view)) / E(mu_view), Fms = Fss E_avg / (1 - Fss (1 - E_avg)) (microfacet_ggx_preserve_energy);
  - dielectric variant (metallic 0): F0 = 0.04 * 2 * specular_ior_level (IOR 1.5), exact dielectric Fresnel with the
    IOR that gives this F0, white specular tint; Lambertian diffuse base_color / pi attenuated by
    (1 - albedo of the specular lobe for the view direction) (closure_layering_weight);
  - metallic variant (metallic 1): F82-tint Fresnel with white tint = Schlick with F0 = base colour, no diffuse;
  - mixed variant: metallic free in 0..1 (Cycles mixes metal lobe and dielectric layer linearly with it), no coat;
    for metallic paints, whose coloured lobe sits on a weak diffuse floor;
  - optional coat (dielectric or metallic variant): GGX lobe with IOR 1.5, weight coat_weight, roughness coat_roughness, energy
    compensated; everything below it is attenuated by (1 - coat_weight * coat albedo(view)).
The view direction is the outgoing direction; every MERL cell is used in both orientations (MERL is reciprocal,
the Cycles model is not quite, because of the view-dependent terms).

Error (Ngan et al. 2005 style, cosine-weighted and log-compressed):
    sqrt( sum_k w_k (ln(1 + f_meas cos_i / eps) - ln(1 + f_fit cos_i / eps))^2 / sum_k w_k ),
averaged over RGB, on a random subsample of valid MERL cells with theta_in and theta_out <= 80 deg. eps = 0.003 1/sr
(--log-eps; the classic form has eps = 1). With eps = 1 the error is linear below ~0.1 1/sr, so the diffuse floor of
dark glossy materials does not count: in a test, black phenolic, chrome, aluminium and two-layer-silver got a diffuse
floor 10-100x above the measured one. eps = 0.003 (about 1 % albedo) makes the error relative down to the level that
is no longer visible. Default weights
(--weighting table): w_k = 1 per MERL cell, so the specular region counts as densely as the table samples it (sqrt
theta_h spacing). With --weighting solid-angle, w_k is the measure of the cell (4 cos theta_d sin theta_h sin theta_d
dtheta_h dtheta_d dphi_d); that makes the peak almost irrelevant (in a test, sharp lobes became unconstrained spikes
and chrome traded F0 against roughness), so it is not the default. `err_rel` divides
the error by the weighted spread of the measured log values (0 = perfect, 1 = no better than a constant).

MERL values are taken as they are (MERL is not on an absolute radiometric scale). The per-direction albedo tables
(E, E_avg, Fresnel-weighted albedo) are recomputed here by quasi-Monte Carlo; Cycles uses its own precomputed tables,
so the energy terms agree only approximately.

Outputs (in --out): merl_fit.csv (chosen model per material), merl_fit_variants.csv (all five variants per material),
merl_fit_ranges.json (parameter ranges and quantiles), merl_fit.log, merl_fit_slices.png (measured vs fitted in-plane
slices), merl_fit_params.png (parameter histograms).
"""
import argparse
import csv
import json
import os
import sys
import time
from multiprocessing import Pool

import numpy as np
from scipy.ndimage import map_coordinates
from scipy.optimize import least_squares
from scipy.stats import qmc

MERL_DIR = os.environ.get("MERL_DIR")  # folder with the 100 MERL *.binary files
SCALE = np.array([1.0, 1.15, 1.66]) / 1500.0
N_TH, N_TD, N_PD = 90, 90, 180
F0_IOR15 = 0.04
R_MIN = 0.005
FIG_MATERIALS = ["white-diffuse-bball", "blue-fabric", "red-plastic", "specular-black-phenolic",
                 "gold-metallic-paint", "aluminium", "chrome", "two-layer-silver"]
VARIANTS = ("dielectric", "dielectric+coat", "metallic", "metallic+coat", "mixed")

_log_lines = []


def say(s=""):
    print(s, flush=True)
    _log_lines.append(s)


# ---------------------------------------------------------------------------------------------
# MERL data
# ---------------------------------------------------------------------------------------------


def load_merl(path):
    """[theta_h, theta_d, phi_d, 3] float64 (scaled), invalid bins (any channel < 0) set to NaN."""
    with open(path, "rb") as f:
        dims = np.fromfile(f, np.int32, 3)
        n = int(np.prod(dims))
        data = np.fromfile(f, np.float64, 3 * n)
    if tuple(dims) != (N_TH, N_TD, N_PD) or data.size != 3 * n:
        raise ValueError(f"{path}: unexpected layout {dims}, {data.size} values")
    arr = np.moveaxis(data.reshape(3, N_TH, N_TD, N_PD), 0, -1) * SCALE
    arr[(arr < 0).any(-1)] = np.nan
    return arr


def cell_centres():
    """Bin centres (MERL BRDFRead.cpp truncates, so bin i covers [i, i+1) in index space)."""
    th = ((np.arange(N_TH) + 0.5) / N_TH) ** 2 * (np.pi / 2)
    td = (np.arange(N_TD) + 0.5) / N_TD * (np.pi / 2)
    pd = (np.arange(N_PD) + 0.5) / N_PD * np.pi
    th_edges = (np.arange(N_TH + 1) / N_TH) ** 2 * (np.pi / 2)
    return th, td, pd, np.diff(th_edges)


def directions(th, td, pd):
    """(theta_h, theta_d, phi_d) -> wi, wo [..., 3]; the half vector lies in the x-z plane (MERL convention)."""
    dx, dy, dz = np.sin(td) * np.cos(pd), np.sin(td) * np.sin(pd), np.cos(td)
    wi = np.stack([dx * np.cos(th) + dz * np.sin(th), dy, -dx * np.sin(th) + dz * np.cos(th)], -1)
    half = np.stack([np.sin(th), np.zeros_like(th), np.cos(th)], -1)
    wo = 2.0 * np.sum(wi * half, -1, keepdims=True) * half - wi
    return wi, wo


def half_diff(wi, wo):
    h = wi + wo
    h = h / np.maximum(np.linalg.norm(h, axis=-1, keepdims=True), 1e-12)
    th = np.arccos(np.clip(h[..., 2], -1, 1))
    ph = np.arctan2(h[..., 1], h[..., 0])
    c, s = np.cos(-ph), np.sin(-ph)
    x1 = wi[..., 0] * c - wi[..., 1] * s
    y1 = wi[..., 0] * s + wi[..., 1] * c
    z1 = wi[..., 2]
    c, s = np.cos(-th), np.sin(-th)
    x2 = x1 * c + z1 * s
    z2 = -x1 * s + z1 * c
    return th, np.arccos(np.clip(z2, -1, 1)), np.arctan2(y1, x2)


def merl_lookup(arr, wi, wo):
    """Nearest-bin lookup as in MERL BRDFRead.cpp (phi_d < 0 folded by reciprocity)."""
    th, td, pd = half_diff(wi, wo)
    pd = np.where(pd < 0, pd + np.pi, pd)
    ih = np.clip((np.sqrt(np.clip(th / (np.pi / 2), 0, 1)) * N_TH).astype(int), 0, N_TH - 1)
    idd = np.clip((td / (np.pi / 2) * N_TD).astype(int), 0, N_TD - 1)
    ip = np.clip((pd / np.pi * N_PD).astype(int), 0, N_PD - 1)
    return arr[ih, idd, ip]


def geometry(wi, wo):
    h = wi + wo
    h = h / np.linalg.norm(h, axis=-1, keepdims=True)
    return dict(ci=wi[..., 2], co=wo[..., 2], ch=h[..., 2], cd=np.clip(np.sum(wi * h, -1), 0.0, 1.0))


def sample_cells(arr, n_cells, max_theta_deg, rng, weighting="table"):
    """Random valid, non-grazing cells, each used in both orientations. Returns geometry, measured values, weights.
    weighting 'table': every MERL cell counts the same (the table is dense near the specular peak, sqrt theta_h);
    'solid-angle': each cell weighted by its measure in (wi, wo) space."""
    th, td, pd, dth = cell_centres()
    TH, TD, PD = np.meshgrid(th, td, pd, indexing="ij")
    wi, wo = directions(TH, TD, PD)
    cmin = np.cos(np.radians(max_theta_deg))
    ok = np.isfinite(arr).all(-1) & (wi[..., 2] >= cmin) & (wo[..., 2] >= cmin)
    idx = np.flatnonzero(ok)
    n_ok = idx.size
    if n_cells < idx.size:
        idx = rng.choice(idx, n_cells, replace=False)
    ih, idd, ip = np.unravel_index(idx, ok.shape)
    if weighting == "solid-angle":
        w = np.cos(td[idd]) * np.sin(th[ih]) * np.sin(td[idd]) * dth[ih]   # constant dtheta_d dphi_d dropped
    else:
        w = np.ones(idx.size)
    a, b = wi.reshape(-1, 3)[idx], wo.reshape(-1, 3)[idx]
    g = geometry(np.concatenate([a, b]), np.concatenate([b, a]))
    meas = np.concatenate([arr.reshape(-1, 3)[idx]] * 2)
    return g, meas, np.concatenate([w, w]), n_ok


# ---------------------------------------------------------------------------------------------
# GGX pieces (Cycles conventions)
# ---------------------------------------------------------------------------------------------


def fresnel_dielectric(c, eta):
    """Unpolarized dielectric Fresnel (Cycles fresnel_dielectric), c = cos of the angle to the microfacet normal."""
    eta = np.asarray(eta, float)
    g2 = eta ** 2 - 1.0 + c ** 2
    g = np.sqrt(np.maximum(g2, 0.0))
    A = (g - c) / np.maximum(g + c, 1e-12)
    B = (c * (g + c) - 1.0) / (c * (g - c) + 1.0)
    return np.where(g2 > 0, 0.5 * A * A * (1.0 + B * B), 1.0)


def fresnel_dielectric_Fss(eta):
    """Cycles' fit of the hemispherical average Fresnel (eta >= 1)."""
    return (eta - 1.0) / (4.08567 + 1.00071 * eta)


def ior_from_F0(f0):
    s = np.sqrt(np.clip(f0, 0.0, 0.99))
    return (1.0 + s) / (1.0 - s)


def D_ggx(ch, a2):
    d = ch * ch * (a2 - 1.0) + 1.0
    return a2 / (np.pi * d * d)


def lambda_ggx(c, a2):
    c2 = np.maximum(c * c, 1e-12)
    return 0.5 * (np.sqrt(1.0 + a2 * (1.0 - c2) / c2) - 1.0)


def _vndf_weights(mu, a, u):
    """GGX VNDF samples (Heitz 2018) for view cosines mu [M]; returns G2/G1 weights [M, S] and cos(view, h)."""
    mu = mu[:, None]
    sx = np.sqrt(1.0 - mu * mu)
    n = np.sqrt((a * sx) ** 2 + mu * mu)
    hx, hz = a * sx / n, mu / n                       # Vh; T1 = (0, 1, 0), T2 = (-hz, 0, hx)
    r = np.sqrt(u[None, :, 0])
    phi = 2 * np.pi * u[None, :, 1]
    t1, t2 = r * np.cos(phi), r * np.sin(phi)
    s = 0.5 * (1.0 + hz)
    t2 = (1.0 - s) * np.sqrt(np.maximum(1.0 - t1 * t1, 0.0)) + s * t2
    nz = np.sqrt(np.maximum(1.0 - t1 * t1 - t2 * t2, 0.0))
    Nx, Ny, Nz = -t2 * hz + nz * hx, t1, t2 * hx + nz * hz
    Ex, Ey, Ez = a * Nx, a * Ny, np.maximum(Nz, 1e-7)
    nn = np.sqrt(Ex * Ex + Ey * Ey + Ez * Ez)
    Ex, Ez = Ex / nn, Ez / nn
    cvh = np.clip(sx * Ex + mu * Ez, 0.0, 1.0)
    oz = 2.0 * cvh * Ez - mu
    a2 = a * a
    lv = lambda_ggx(mu, a2)
    w = np.where(oz > 0, (1.0 + lv) / (1.0 + lv + lambda_ggx(np.maximum(oz, 1e-6), a2)), 0.0)
    return w, cvh


def ggx_tables(n_r=32, n_mu=32, n_z=13, z_max=0.6, n_samp=2 ** 15, seed=1):
    """E(roughness, mu), E_avg(roughness) of single-scattering GGX (F = 1), and the albedo of the energy-compensated
    dielectric lobe A(roughness, mu, z), z = F0^(1/4) = sqrt((eta - 1) / (eta + 1)) (the Cycles table coordinate)."""
    u = qmc.Sobol(2, scramble=True, seed=seed).random(n_samp)
    r_grid, mu_grid = np.linspace(0, 1, n_r), np.linspace(0, 1, n_mu)
    z_grid = np.linspace(0, z_max, n_z)
    eta_z = ior_from_F0(z_grid ** 4)
    mu_fine = (np.arange(64) + 0.5) / 64
    E = np.zeros((n_r, n_mu))
    Eavg = np.zeros(n_r)
    Ass = np.zeros((n_r, n_mu, n_z))
    for i, r in enumerate(r_grid):
        a = max(r * r, 1e-4)
        w, cvh = _vndf_weights(np.concatenate([np.clip(mu_grid, 1e-3, 1.0), mu_fine]), a, u)
        e = w.mean(1)
        E[i] = e[:n_mu]
        Eavg[i] = 2.0 * np.mean(e[n_mu:] * mu_fine)
        for k, eta in enumerate(eta_z):
            Ass[i, :, k] = (fresnel_dielectric(cvh[:n_mu], eta) * w[:n_mu]).mean(1)
    Fss = fresnel_dielectric_Fss(eta_z)[None, None, :]
    Ea = Eavg[:, None, None]
    Fms = Fss * Ea / (1.0 - Fss * (1.0 - Ea))
    A = Ass * (1.0 + Fms * (1.0 - E[:, :, None]) / E[:, :, None])
    return dict(E=E, Eavg=Eavg, A=A, z_max=z_max)


def _E(T, r, mu):
    n = T["E"].shape[0] - 1
    return map_coordinates(T["E"], [np.full_like(mu, r * n), mu * n], order=1, mode="nearest")


def _Eavg(T, r):
    return float(np.interp(r, np.linspace(0, 1, T["Eavg"].size), T["Eavg"]))


def _A(T, r, mu, z):
    nr, nm, nz = T["A"].shape
    zc = np.full_like(mu, min(z / T["z_max"], 1.0) * (nz - 1))
    return map_coordinates(T["A"], [np.full_like(mu, r * (nr - 1)), mu * (nm - 1), zc], order=1, mode="nearest")


def ggx_lobe(T, r, g, F, Fss):
    """Energy-compensated GGX reflection lobe D G F / (4 ci co) * ms, view = outgoing direction. F, Fss broadcast [.., 3]."""
    a2 = max(r, R_MIN) ** 4
    G = 1.0 / (1.0 + lambda_ggx(g["ci"], a2) + lambda_ggx(g["co"], a2))
    base = D_ggx(g["ch"], a2) * G / (4.0 * g["ci"] * g["co"])
    E = _E(T, r, g["co"])[:, None]
    Ea = _Eavg(T, r)
    Fms = Fss * Ea / (1.0 - Fss * (1.0 - Ea))
    return base[:, None] * F * (1.0 + Fms * (1.0 - E) / E)


def principled(T, p, g):
    """Cycles Principled BSDF (subset) for parameters p at geometry g; returns [N, 3]."""
    base = np.asarray(p["base"], float)[None, :]
    r = p["roughness"]
    m = p["metallic"]
    f = 0.0
    if m > 0:
        F = base + (1.0 - base) * ((1.0 - g["cd"]) ** 5)[:, None]
        f = f + m * ggx_lobe(T, r, g, F, base + (1.0 - base) / 21.0)
    if m < 1:
        f0 = F0_IOR15 * 2.0 * p["specular_ior_level"]
        eta = ior_from_F0(f0)
        F = fresnel_dielectric(g["cd"], eta)[:, None]
        spec = ggx_lobe(T, r, g, F, fresnel_dielectric_Fss(eta))
        layer = np.clip(1.0 - _A(T, r, g["co"], f0 ** 0.25), 0.0, 1.0)[:, None]
        f = f + (1.0 - m) * (spec + base / np.pi * layer)
    wc = p.get("coat_weight", 0.0)
    if wc > 0:
        rc = p["coat_roughness"]
        Fc = fresnel_dielectric(g["cd"], 1.5)[:, None]
        coat = wc * ggx_lobe(T, rc, g, Fc, fresnel_dielectric_Fss(1.5))
        layer = np.clip(1.0 - wc * _A(T, rc, g["co"], F0_IOR15 ** 0.25), 0.0, 1.0)[:, None]
        f = coat + layer * f
    return f


# ---------------------------------------------------------------------------------------------
# Fitting
# ---------------------------------------------------------------------------------------------


def unpack(kind, x):
    p = dict(base=list(x[:3]), roughness=float(x[3]), metallic=1.0 if kind.startswith("metallic") else 0.0,
             specular_ior_level=float("nan"), coat_weight=0.0, coat_roughness=float("nan"))
    k = 4
    if kind.startswith("dielectric") or kind == "mixed":
        p["specular_ior_level"] = float(x[4])
        k = 5
    if kind == "mixed":
        p["metallic"] = float(x[5])
    if kind.endswith("+coat"):
        p["coat_weight"], p["coat_roughness"] = float(x[k]), float(x[k + 1])
    return p


def bounds(kind):
    lo, hi = [0.0, 0.0, 0.0, R_MIN], [1.0, 1.0, 1.0, 1.0]
    if kind.startswith("dielectric"):
        lo.append(0.0)
        hi.append(1.0)
    if kind == "mixed":
        lo += [0.0, 0.0]
        hi += [1.0, 1.0]
    if kind.endswith("+coat"):
        lo += [0.0, R_MIN]
        hi += [1.0, 1.0]
    return np.array(lo), np.array(hi)


def fit_variant(T, kind, starts, g, y, sw, ci):
    lo, hi = bounds(kind)

    def resid(x):
        f = principled(T, unpack(kind, x), g)
        return ((np.log1p(np.maximum(f, 0.0) * ci[:, None]) - y) * sw[:, None]).ravel()

    best = None
    for x0 in starts:
        x0 = np.clip(np.asarray(x0, float), lo + 1e-6, hi - 1e-6)
        res = least_squares(resid, x0, bounds=(lo, hi), x_scale="jac", diff_step=1e-4, max_nfev=200,
                            ftol=1e-7, xtol=1e-7)
        if best is None or res.cost < best.cost:
            best = res
    return unpack(kind, best.x), float(np.sqrt(2.0 * best.cost))


def fit_material(args):
    path, T, n_cells, max_theta, gain, seed, weighting, eps = args
    t0 = time.time()
    name = os.path.splitext(os.path.basename(path))[0]
    arr = load_merl(path)
    rng = np.random.default_rng(seed)
    g, meas, w, n_ok = sample_cells(arr, n_cells, max_theta, rng, weighting)
    ci = g["ci"]
    ci = ci / eps                                   # ln(1 + f cos_i / eps)
    y = np.log1p(meas * ci[:, None])
    sw = np.sqrt(w / w.sum() / 3.0)
    ybar = np.sum(sw[:, None] ** 2 * y, axis=0) * 3.0
    spread = float(np.sqrt(np.sum((sw[:, None] * (y - ybar)) ** 2)))

    th_cell = np.arccos(np.clip(g["ch"], -1, 1))
    tail = th_cell > np.radians(45)
    base0 = np.clip(np.pi * np.median(meas[tail] if tail.sum() > 50 else meas, axis=0), 0.01, 0.95)
    out = {}
    starts = [list(base0) + [r0, lv] for r0, lv in ((0.06, 0.5), (0.2, 0.5), (0.5, 0.5), (0.3, 0.15))]
    out["dielectric"] = fit_variant(T, "dielectric", starts, g, y, sw, ci)
    starts = [[b, b, b, r0] for b in (0.3, 0.8) for r0 in (0.06, 0.2, 0.5)]
    out["metallic"] = fit_variant(T, "metallic", starts, g, y, sw, ci)
    for fam in ("dielectric", "metallic"):
        p, _ = out[fam]
        r = p["roughness"]
        head = list(p["base"]) + [r] + ([p["specular_ior_level"]] if fam == "dielectric" else [])
        head_broad = list(p["base"]) + [min(1.0, 2.0 * r + 0.05)] + head[4:]
        head_sharp = list(p["base"]) + [max(R_MIN, 0.5 * r)] + head[4:]
        starts = [head_broad + [0.5, max(R_MIN, 0.5 * r)], head_sharp + [0.3, min(1.0, 2.0 * r + 0.05)],
                  head + [0.5, 0.03]]
        out[fam + "+coat"] = fit_variant(T, fam + "+coat", starts, g, y, sw, ci)
    pd_, pm = out["dielectric"][0], out["metallic"][0]
    starts = [list(pm["base"]) + [pm["roughness"], 0.5, 0.9], list(pm["base"]) + [pm["roughness"], 0.5, 0.5],
              list(pd_["base"]) + [pd_["roughness"], pd_["specular_ior_level"], 0.1]]
    out["mixed"] = fit_variant(T, "mixed", starts, g, y, sw, ci)

    # choice: a coat only if it clearly helps (relative error reduction >= gain); then the better of dielectric and
    # metallic (as fitted, with or without coat); a free metallic fraction (mixed) only if it clearly beats that
    clear = lambda a, b: out[a][1] < (1.0 - gain) * out[b][1]
    pick = {}
    for fam in ("dielectric", "metallic"):
        coat_ok = clear(fam + "+coat", fam) and out[fam + "+coat"][0]["coat_weight"] >= 0.05
        pick[fam] = fam + "+coat" if coat_ok else fam
    chosen = min(pick.values(), key=lambda k: out[k][1])
    if clear("mixed", chosen):
        chosen = "mixed"
    return dict(name=name, variants=out, chosen=chosen, spread=spread, n_valid=n_ok, n_cells=len(ci) // 2,
                seconds=time.time() - t0)


# ---------------------------------------------------------------------------------------------
# Outputs
# ---------------------------------------------------------------------------------------------

CSV_COLS = ["name", "model", "base_r", "base_g", "base_b", "metallic", "roughness", "specular_ior_level",
            "coat_weight", "coat_roughness", "gloss_roughness", "err_chosen", "err_rel", "err_dielectric",
            "err_dielectric_coat", "err_metallic", "err_metallic_coat", "err_mixed", "flags"]


def fmt(v):
    return "" if v is None or (isinstance(v, float) and np.isnan(v)) else (f"{v:.5g}" if isinstance(v, float) else v)


def gloss_roughness(p):
    """Roughness of the dominant specular lobe: the coat's when its F0 (0.04 x weight) exceeds the base lobe's."""
    m = p["metallic"]
    lv = p["specular_ior_level"] if np.isfinite(p["specular_ior_level"]) else 0.0
    f0_base = m * float(np.mean(p["base"])) + (1.0 - m) * F0_IOR15 * 2.0 * lv
    return p["coat_roughness"] if F0_IOR15 * p["coat_weight"] > f0_base else p["roughness"]


def flags_of(p):
    f = []
    if max(p["base"]) > 0.995:
        f.append("base_at_1")
    if p["roughness"] <= R_MIN * 1.01:
        f.append("roughness_at_min")
    if p["roughness"] >= 0.995:
        f.append("roughness_at_1")
    if p["metallic"] < 1 and p["specular_ior_level"] >= 0.995:
        f.append("spec_level_at_1")
    if p["metallic"] < 1 and p["specular_ior_level"] <= 0.005:
        f.append("spec_level_at_0")
    if p["coat_weight"] >= 0.995:
        f.append("coat_weight_at_1")
    return ";".join(f)


def write_outputs(results, out_dir):
    results = sorted(results, key=lambda r: r["name"])
    with open(os.path.join(out_dir, "merl_fit.csv"), "w", newline="", encoding="utf-8") as f:
        wr = csv.writer(f)
        wr.writerow(CSV_COLS)
        for r in results:
            p, e = r["variants"][r["chosen"]]
            errs = [r["variants"][k][1] for k in VARIANTS]
            wr.writerow([r["name"], r["chosen"], *[fmt(float(b)) for b in p["base"]], fmt(p["metallic"]),
                         fmt(p["roughness"]), fmt(p["specular_ior_level"]), fmt(p["coat_weight"]),
                         fmt(p["coat_roughness"]), fmt(gloss_roughness(p)), fmt(e), fmt(e / r["spread"]),
                         *[fmt(x) for x in errs], flags_of(p)])
    with open(os.path.join(out_dir, "merl_fit_variants.csv"), "w", newline="", encoding="utf-8") as f:
        wr = csv.writer(f)
        wr.writerow(["name", "variant", "chosen", "base_r", "base_g", "base_b", "metallic", "roughness",
                     "specular_ior_level", "coat_weight", "coat_roughness", "err", "err_rel", "flags"])
        for r in results:
            for k in VARIANTS:
                p, e = r["variants"][k]
                wr.writerow([r["name"], k, int(k == r["chosen"]), *[fmt(float(b)) for b in p["base"]],
                             fmt(p["metallic"]), fmt(p["roughness"]), fmt(p["specular_ior_level"]),
                             fmt(p["coat_weight"]), fmt(p["coat_roughness"]), fmt(e), fmt(e / r["spread"]), flags_of(p)])


def read_variants(out_dir):
    res = {}
    with open(os.path.join(out_dir, "merl_fit_variants.csv"), encoding="utf-8") as f:
        for row in csv.DictReader(f):
            num = lambda k: float(row[k]) if row[k] != "" else float("nan")
            p = dict(base=[num("base_r"), num("base_g"), num("base_b")], metallic=num("metallic"),
                     roughness=num("roughness"), specular_ior_level=num("specular_ior_level"),
                     coat_weight=num("coat_weight"), coat_roughness=num("coat_roughness"))
            r = res.setdefault(row["name"], dict(name=row["name"], variants={}, chosen=None))
            r["variants"][row["variant"]] = (p, num("err"))
            r["spread"] = num("err") / num("err_rel") if num("err_rel") > 0 else 1.0
            if row["chosen"] == "1":
                r["chosen"] = row["variant"]
    return list(res.values())


def quantiles(v):
    v = np.asarray([x for x in v if np.isfinite(x)], float)
    if v.size == 0:
        return dict(n=0)
    q = np.quantile(v, [0, 0.05, 0.25, 0.5, 0.75, 0.95, 1])
    return dict(n=int(v.size), min=q[0], q05=q[1], q25=q[2], median=q[3], q75=q[4], q95=q[5], max=q[6],
                mean=float(v.mean()))


def summarise(results, out_dir, args):
    ch = [(r["name"], r["chosen"], *r["variants"][r["chosen"]], r["spread"]) for r in results]
    P = lambda key: [p[key] for _, _, p, _, _ in ch]
    base = np.array(P("base"))
    lum = base @ np.array([0.2126, 0.7152, 0.0722])
    diel = [p for _, k, p, _, _ in ch if k.startswith("dielectric")]
    met = [p for _, k, p, _, _ in ch if k.startswith("metallic")]
    mix = [p for _, k, p, _, _ in ch if k == "mixed"]
    coat = [p for _, k, p, _, _ in ch if k.endswith("+coat")]
    dvar = [r["variants"]["dielectric"][0] for r in results]
    summary = dict(
        n_materials=len(results),
        model_counts={k: sum(1 for _, c, _, _, _ in ch if c == k) for k in VARIANTS},
        chosen=dict(
            roughness=quantiles(P("roughness")),
            roughness_dielectric=quantiles([p["roughness"] for p in diel]),
            roughness_metallic=quantiles([p["roughness"] for p in met]),
            roughness_mixed=quantiles([p["roughness"] for p in mix]),
            metallic_mixed=quantiles([p["metallic"] for p in mix]),
            gloss_roughness=quantiles([gloss_roughness(p) for _, _, p, _, _ in ch]),
            alpha=quantiles([p["roughness"] ** 2 for _, _, p, _, _ in ch]),
            base_r=quantiles(base[:, 0]), base_g=quantiles(base[:, 1]), base_b=quantiles(base[:, 2]),
            base_luminance=quantiles(lum),
            base_luminance_dielectric=quantiles([np.dot(p["base"], [0.2126, 0.7152, 0.0722]) for p in diel]),
            base_luminance_metallic=quantiles([np.dot(p["base"], [0.2126, 0.7152, 0.0722]) for p in met]),
            specular_ior_level=quantiles([p["specular_ior_level"] for p in diel]),
            specular_ior_level_mixed=quantiles([p["specular_ior_level"] for p in mix]),
            coat_weight=quantiles([p["coat_weight"] for p in coat]),
            coat_roughness=quantiles([p["coat_roughness"] for p in coat]),
            err=quantiles([e for _, _, _, e, _ in ch]),
            err_rel=quantiles([e / s for _, _, _, e, s in ch]),
        ),
        dielectric_variant_all=dict(
            roughness=quantiles([p["roughness"] for p in dvar]),
            specular_ior_level=quantiles([p["specular_ior_level"] for p in dvar]),
            err=quantiles([r["variants"]["dielectric"][1] for r in results]),
        ),
        flags={f: sum(1 for _, _, p, _, _ in ch if f in flags_of(p).split(";"))
               for f in ("base_at_1", "roughness_at_min", "roughness_at_1", "spec_level_at_1", "spec_level_at_0",
                         "coat_weight_at_1")},
        settings=dict(n_cells=args.n_cells, max_theta_deg=args.max_theta, coat_gain=args.coat_gain,
                      weighting=args.weighting, log_eps=args.log_eps,
                      error="weighted RMS of ln(1 + f cos_i / eps) - ln(1 + f_fit cos_i / eps), RGB mean",
                      model="Cycles 4.5 Principled: Multiscatter GGX, Lambert diffuse, metallic 0 or 1 (free in "
                            "the mixed variant), coat IOR 1.5",
                      selection="coat only if the error drops by at least coat_gain (relative); then the lower "
                                "error of dielectric and metallic; mixed only if it beats that by coat_gain",
                      note="MERL is not on an absolute scale; values are as fitted. Blender roughness = sqrt(GGX alpha)."),
    )
    with open(os.path.join(out_dir, "merl_fit_ranges.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=float)
    return summary


def draw_slices(results, T, merl_dir, out_png, names):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    byname = {r["name"]: r for r in results}
    names = [n for n in names if n in byname]
    th_in = (30, 60)
    ncol = 4
    nrow = int(np.ceil(len(names) * len(th_in) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.2 * ncol, 3.2 * nrow), squeeze=False)
    to = np.radians(np.arange(-89.5, 90.0, 0.5))
    cols = ("tab:red", "tab:green", "tab:blue")
    k = 0
    for name in names:
        r = byname[name]
        arr = load_merl(os.path.join(merl_dir, name + ".binary"))
        p, e = r["variants"][r["chosen"]]
        for t in th_in:
            ax = axes.flat[k]
            k += 1
            ti = np.radians(t)
            wi = np.broadcast_to(np.array([np.sin(ti), 0.0, np.cos(ti)]), (to.size, 3))
            wo = np.stack([-np.sin(to), np.zeros_like(to), np.cos(to)], -1)   # theta_out > 0: specular side
            meas = merl_lookup(arr, wi, wo)
            fit = principled(T, p, geometry(wi, wo))
            for c in range(3):
                ax.plot(np.degrees(to), meas[:, c], ".", ms=2.5, color=cols[c], alpha=0.6)
                ax.plot(np.degrees(to), fit[:, c], "-", lw=1.0, color=cols[c])
            ax.axvspan(-90, -80, color="0.9")
            ax.axvspan(80, 90, color="0.9")
            ax.axvline(t, color="0.6", lw=0.6, ls=":")
            ax.set_yscale("log")
            ax.set_xlim(-90, 90)
            ax.set_title(f"{name}, theta_in {t} deg\n{r['chosen']}, r {p['roughness']:.3f}, err {e:.3f}", fontsize=8)
            ax.grid(alpha=0.3)
            ax.tick_params(labelsize=7)
            if k % ncol == 1:
                ax.set_ylabel("BRDF [1/sr]", fontsize=8)
            ax.set_xlabel("theta_out in the plane of incidence [deg]", fontsize=7)
    for ax in axes.flat[k:]:
        ax.axis("off")
    fig.suptitle("MERL measured (dots) vs fitted Principled BSDF (lines), R/G/B; grey: > 80 deg, not fitted", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_png, dpi=120)
    plt.close(fig)


def draw_params(results, out_png):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ch = [(r["chosen"], *r["variants"][r["chosen"]]) for r in results]
    fig, ax = plt.subplots(1, 4, figsize=(16, 3.4))
    for fam, col in (("dielectric", "tab:blue"), ("metallic", "tab:orange"), ("mixed", "tab:green")):
        rs = [p["roughness"] for k, p, _ in ch if k.startswith(fam)]
        ax[0].hist(rs, bins=np.linspace(0, 1, 26), color=col, alpha=0.7, label=f"{fam} ({len(rs)})")
        lum = [np.dot(p["base"], [0.2126, 0.7152, 0.0722]) for k, p, _ in ch if k.startswith(fam)]
        ax[1].hist(lum, bins=np.linspace(0, 1, 26), color=col, alpha=0.7, label=fam)
    ax[0].set_xlabel("roughness (chosen model)")
    ax[0].legend(fontsize=8)
    ax[1].set_xlabel("base colour luminance")
    lv = [p["specular_ior_level"] for k, p, _ in ch if k.startswith("dielectric")]
    ax[2].hist(lv, bins=np.linspace(0, 1, 26), color="tab:blue", alpha=0.7)
    ax[2].set_xlabel("specular IOR level (dielectric)")
    cw = [(p["coat_weight"], p["coat_roughness"]) for k, p, _ in ch if k.endswith("+coat")]
    if cw:
        ax[3].scatter(*zip(*cw), s=12)
    ax[3].set_xlim(0, 1)
    ax[3].set_ylim(0, 1)
    ax[3].set_xlabel("coat weight")
    ax[3].set_ylabel("coat roughness")
    ax[3].set_title(f"coated materials ({len(cw)})", fontsize=9)
    for a in ax:
        a.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_png, dpi=120)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--merl-dir", default=MERL_DIR, required=MERL_DIR is None)
    ap.add_argument("--out", default=os.path.join("runs", "merl_fit"))
    ap.add_argument("--materials", nargs="*", help="subset of material names (default: all)")
    ap.add_argument("--n-cells", type=int, default=25000, help="MERL cells per material (each used twice)")
    ap.add_argument("--max-theta", type=float, default=80.0, help="drop cells with theta_in or theta_out above this")
    ap.add_argument("--coat-gain", type=float, default=0.10,
                    help="relative error reduction needed to keep a coat or a free metallic (mixed)")
    ap.add_argument("--weighting", choices=("table", "solid-angle"), default="table",
                    help="cell weights of the error (table: uniform per MERL cell, emphasises the specular region)")
    ap.add_argument("--log-eps", type=float, default=0.003,
                    help="error compares ln(1 + f cos_i / eps); eps = 1 is the classic Ngan / Loew form")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--figure-materials", nargs="*", default=FIG_MATERIALS)
    ap.add_argument("--figure-only", action="store_true", help="redraw the figures from merl_fit_variants.csv")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    t0 = time.time()
    say(f"merl_fit: {' '.join(sys.argv[1:])}")
    T = ggx_tables()
    say(f"GGX albedo tables: {time.time() - t0:.1f} s (E at roughness 1, mu 0.5: {_E(T, 1.0, np.array([0.5]))[0]:.3f}, "
        f"E_avg at roughness 1: {_Eavg(T, 1.0):.3f})")

    if args.figure_only:
        results = read_variants(args.out)
    else:
        paths = sorted(os.path.join(args.merl_dir, f) for f in os.listdir(args.merl_dir) if f.endswith(".binary"))
        if args.materials:
            paths = [p for p in paths if os.path.splitext(os.path.basename(p))[0] in set(args.materials)]
        say(f"{len(paths)} materials from {args.merl_dir}, {args.workers} workers, {args.n_cells} cells each")
        jobs = [(p, T, args.n_cells, args.max_theta, args.coat_gain, args.seed, args.weighting,
                 args.log_eps) for p in paths]
        results = []
        with Pool(args.workers) as pool:
            for r in pool.imap_unordered(fit_material, jobs):
                results.append(r)
                p, e = r["variants"][r["chosen"]]
                errs = " ".join(f"{k} {r['variants'][k][1]:.4f}" for k in VARIANTS)
                say(f"[{len(results):3d}/{len(paths)}] {r['name']:26s} {r['chosen']:16s} base "
                    f"({p['base'][0]:.3f} {p['base'][1]:.3f} {p['base'][2]:.3f}) r {p['roughness']:.3f} "
                    f"lvl {p['specular_ior_level']:.3f} coat {p['coat_weight']:.3f}/{p['coat_roughness']:.3f} "
                    f"err_rel {e / r['spread']:.3f} | {errs} | {r['seconds']:.0f} s {flags_of(p)}")
        write_outputs(results, args.out)
    s = summarise(results, args.out, args)
    say(f"model counts: {s['model_counts']}")
    for key in ("roughness", "roughness_dielectric", "roughness_metallic", "roughness_mixed", "metallic_mixed",
                "gloss_roughness", "base_luminance", "specular_ior_level",
                "coat_weight", "coat_roughness", "err", "err_rel"):
        q = s["chosen"][key]
        if q.get("n"):
            say(f"  {key:22s} n {q['n']:3d}  min {q['min']:.3f}  q05 {q['q05']:.3f}  q25 {q['q25']:.3f}  "
                f"median {q['median']:.3f}  q75 {q['q75']:.3f}  q95 {q['q95']:.3f}  max {q['max']:.3f}")
    say(f"bound hits (chosen model): {s['flags']}")
    worst = sorted(results, key=lambda r: -r["variants"][r["chosen"]][1] / r["spread"])[:10]
    say("largest relative errors: " + ", ".join(f"{r['name']} {r['variants'][r['chosen']][1] / r['spread']:.3f}"
                                                for r in worst))
    draw_slices(results, T, args.merl_dir, os.path.join(args.out, "merl_fit_slices.png"), args.figure_materials)
    draw_params(results, os.path.join(args.out, "merl_fit_params.png"))
    say(f"total {time.time() - t0:.0f} s; outputs in {os.path.abspath(args.out)}")
    with open(os.path.join(args.out, "merl_fit.log" if not args.figure_only else "merl_fit_figures.log"), "w",
              encoding="utf-8") as f:
        f.write("\n".join(_log_lines) + "\n")


if __name__ == "__main__":
    main()
