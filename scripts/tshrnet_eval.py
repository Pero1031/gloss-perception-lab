"""Test a published specular-removal network (TSHRNet, Fu et al. ICCV 2023) on the gloss-edit test sets.

Needs the TSHRNet repository in external/TSHRNet and its pretrained weights in external/TSHRNet_weights (not
redistributed; see https://github.com/fu123456/TSHRNet). The network returns a specular-free image D; the specular part
is the linear-light difference between the input and D, and the edit is D + gain * specular. Errors are measured as
in gloss_edit_eval.py (RMSE x100 against the same edit of the true specular component), next to the b1 share edit.
"""
import glob
import importlib.util
from collections import OrderedDict

import numpy as np
import torch

from scripts.gloss_edit import edit_share, linear_to_srgb, srgb_to_linear
from scripts.gloss_edit_eval import load, load_net, net_map, rmse

ROOT = "external"


def load_tshrnet(name):
    spec = importlib.util.spec_from_file_location("tshr_unet", f"{ROOT}/TSHRNet/models/UNet.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    nets = []
    for i, ch in enumerate((3, 3, 6, 9), 1):
        net = mod.UNet(input_channels=ch, output_channels=3)
        sd = torch.load(f"{ROOT}/TSHRNet_weights/checkpoints_{name}/UNet{i}_60.pth", map_location="cpu", weights_only=True)
        net.load_state_dict(OrderedDict((k[7:] if k.startswith("module.") else k, v) for k, v in sd.items()))
        nets.append(net.cuda().eval())
    return nets


@torch.no_grad()
def diffuse(nets, x, mix=False):
    """x: (B,3,256,256) sRGB in [0,1] -> specular-free sRGB image (tone-corrected output of the third stage).

    SSHR weights: the first two networks give albedo and shading. Mixed-data weights (mix=True): they give the diffuse
    image and the specular residue directly (test_4_networks_mix.py)."""
    u1, u2, u3, u4 = nets
    img = x * 2 - 1
    coarse, residue = (u1(img), u2(img)) if mix else (lambda c: (c, img - c))(u1(img) * u2(img))
    refined = u3(torch.cat([coarse, img], 1))
    out = u4(torch.cat([refined, residue, img], 1))
    return (out * 0.5 + 0.5).clamp(0, 1)


def tshr_edit(nets, x, mask, gain, black_bg, mix=False):
    xin = torch.where(mask[None, None], x, torch.zeros_like(x)) if black_bg else x
    d = srgb_to_linear(diffuse(nets, xin, mix))
    lin = srgb_to_linear(x)
    return linear_to_srgb(d + gain * (lin - d).clamp_min(0))


def main():
    rng = np.random.default_rng(0)
    b1 = load_net("spec")
    models = {"SSHR": load_tshrnet("SSHR"), "mix": load_tshrnet("mix_SSHR_SHIQ_PSD")}
    held = sorted(glob.glob("data/stereo_ds/scene_*.npz"))[-1000:]
    sets = {"A: Serrano geometries": sorted(glob.glob("data/serrano_render/*.npz")),
            "B: random scenes": [held[i] for i in rng.choice(len(held), 200, replace=False)]}
    methods = ["none", "share[b1]"] + [f"TSHR {m}{b}" for m in models for b in ("", " black bg")]
    for name, paths in sets.items():
        acc = {}
        for q in paths:
            c = load(q)
            lin, mask = c["lin"][None].cuda(), c["mask"].cuda()
            x = torch.round(linear_to_srgb(lin) * 255) / 255
            glossy = c["glossy"][None, None].cuda()
            share = net_map(b1, x, mask[None])[0] * mask
            g = ("matte " if c["matte"] else "glossy ") + c["shape"]
            for gain in (0, 2):
                oracle = linear_to_srgb(lin + (gain - 1) * glossy)
                e = {"none": x, "share[b1]": edit_share(x, share[None], gain)}
                for m, nets in models.items():
                    for b, flag in (("", False), (" black bg", True)):
                        e[f"TSHR {m}{b}"] = tshr_edit(nets, x, mask, gain, flag, m == "mix")
                for k, v in e.items():
                    acc.setdefault((g, gain), {}).setdefault(k, []).append(rmse(v, oracle, mask))
        print(f"\n== {name}: edit error against the true-specular edit (RMSE x100) ==")
        for gain, lab in ((0, "suppress (gain 0)"), (2, "enhance (gain 2)")):
            print(f"{lab:26s}" + "".join(f"{k:>18s}" for k in methods))
            for g in sorted({k[0] for k in acc}):
                print(f"  {g:24s}" + "".join(f"{np.mean(acc[(g, gain)][k]):18.2f}" for k in methods))


if __name__ == "__main__":
    main()
