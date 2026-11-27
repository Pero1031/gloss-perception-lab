"""Loader for the intrinsic decomposition of Careaga & Aksoy (ACM TOG 2023, 2024).

Their code (academic use only, not redistributed here) goes in external/: https://github.com/compphoto/Intrinsic,
https://github.com/CCareaga/MiDaS (commit fb51e3a) as external/altered_midas_src, and
https://github.com/CCareaga/chrislib (commit 9a4c63f) as external/chrislib_src. Weights are fetched from the releases
of the first repository on first use. The encoders they load through torch.hub are built from timm and
torchvision instead (same networks, weights come from the checkpoints), so no further repository is needed.
"""
import sys

import torch

for d in ("external/Intrinsic", "external/altered_midas_src", "external/chrislib_src"):
    if d not in sys.path:
        sys.path.insert(0, d)


def load(version="v2.1"):
    import timm
    from intrinsic.pipeline import load_models

    hub_load = torch.hub.load

    def from_timm(repo, name, *args, **kwargs):
        if repo == "rwightman/gen-efficientnet-pytorch":
            net = timm.create_model(name, pretrained=False, exportable=kwargs.get("exportable", False))
            if not hasattr(net, "act1"):  # current timm applies the activation inside bn1
                net.act1 = torch.nn.Identity()
            return net
        if repo == "facebookresearch/WSL-Images" and name == "resnext101_32x8d_wsl":
            import torchvision
            return torchvision.models.resnext101_32x8d(weights=None)
        return hub_load(repo, name, *args, **kwargs)

    torch.hub.load = from_timm
    try:
        return load_models(version)
    finally:
        torch.hub.load = hub_load


def decompose(models, lin):
    """lin: (H, W, 3) numpy float image, linear, in [0, 1]. Returns the result dictionary of run_pipeline
    (hr_alb: albedo, dif_shd: diffuse shading, dif_img: their product, residual: image - dif_img, pos_res, ...)."""
    from intrinsic.pipeline import run_pipeline
    return run_pipeline(models, lin, linear=True)
