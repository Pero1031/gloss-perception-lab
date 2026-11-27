"""Gloss manipulation prototype: highlight detectors with false-highlight rejection, and edits of the detected layer.

Detectors return a per-pixel score (B, H, W) for an sRGB batch x of shape (B, 3, H, W) in [0, 1]:
  P          gloss map of Nagata et al. (products of ON-centre LoG responses across scales)
  P_balance  P weighted by the local ON/OFF balance: an isolated highlight is ON-dominated, relief shading (lit ridges
             between shadowed grooves) gives ON and OFF responses of similar strength side by side
  P_skew     P weighted by the local skewness of luminance (highlights are sparse bright outliers)
  P_white    P weighted by how much less saturated the pixel is than its surround (dielectric highlights take the
             colour of the light); blind on achromatic surfaces
Edits work in linear RGB:
  specular_layer   image minus a base interpolated from outside the accepted region (normalised convolution)
  edit             base + gain * layer (gain 0 removes the layer, gain > 1 enhances it)
  edit_share       scale the specular part given a per-pixel specular share of the luminance
  edit_additive    I + gain * signal, the additive rule described for Nagata et al.'s gloss control
"""
import torch
import torch.nn.functional as F

from src.models.features import GlossFeatures

LUM = torch.tensor([0.2126, 0.7152, 0.0722]).view(1, 3, 1, 1)


def srgb_to_linear(x):
    return torch.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(x):
    x = x.clamp(0, 1)
    return torch.where(x <= 0.0031308, x * 12.92, 1.055 * x ** (1 / 2.4) - 0.055)


def blur(v, sigma):
    """Gaussian blur of (B, C, H, W) with replicate padding."""
    r = max(1, int(3 * sigma))
    k = torch.exp(-torch.arange(-r, r + 1, device=v.device, dtype=v.dtype) ** 2 / (2 * sigma**2))
    k = k / k.sum()
    c = v.shape[1]
    v = F.conv2d(F.pad(v, (r, r, 0, 0), mode="replicate"), k.view(1, 1, 1, -1).repeat(c, 1, 1, 1), groups=c)
    return F.conv2d(F.pad(v, (0, 0, r, r), mode="replicate"), k.view(1, 1, -1, 1).repeat(c, 1, 1, 1), groups=c)


def masked_blur(v, m, sigma):
    """Blur of v using only pixels where m (B, 1, H, W) is 1."""
    return blur(v * m, sigma) / blur(m, sigma).clamp_min(1e-4)


class Detector:
    def __init__(self, sigma_max=32, window=6.0, device="cuda"):
        self.feat = GlossFeatures(sigma_max, sigma_step=1).to(device)
        self.window = window  # px at 256: neighbourhood over which ON and OFF responses are compared

    @torch.no_grad()
    def maps(self, x, obj=None):
        """All detector scores for x. obj: (B, H, W) bool object mask; statistics ignore pixels outside it."""
        m = torch.ones_like(x[:, :1]) if obj is None else obj[:, None].float()
        P = self.feat.gloss_map(x)
        N = self.feat.gloss_map(1 - x)  # OFF-centre responses: the LoG is linear, so they are the ON responses of 1 - x
        Pn, Nn = (masked_blur(v[:, None], m, self.window)[:, 0] for v in (P, N))
        balance = ((Pn - Nn) / (Pn + Nn).clamp_min(1e-12)).clamp(0, 1)

        y = (srgb_to_linear(x) * LUM.to(x.device)).sum(1, keepdim=True)
        mu = masked_blur(y, m, self.window)
        d = y - mu
        var = masked_blur(d**2, m, self.window)
        skew = (masked_blur(d**3, m, self.window) / var.clamp_min(1e-8) ** 1.5)[:, 0]

        sat = (x.amax(1, keepdim=True) - x.amin(1, keepdim=True)) / x.amax(1, keepdim=True).clamp_min(1e-3)
        white = (masked_blur(sat, m, 2 * self.window) - sat)[:, 0].clamp(0, 1)
        return {"P": P, "P_balance": P * balance, "P_skew": P * (skew / 2).clamp(0, 1), "P_white": P * white,
                "balance": balance}


@torch.no_grad()
def specular_layer(x, accept, grow=2, sigma=5.0):
    """Split linear RGB into base + layer. accept: (B, H, W) bool, pixels taken as highlight.

    The accepted region is grown by `grow` px, and the base under it is interpolated from the surround with a
    normalised Gaussian convolution; the layer is what exceeds the base there (never negative)."""
    lin = srgb_to_linear(x)
    a = F.max_pool2d(accept[:, None].float(), 2 * grow + 1, 1, grow)
    keep = 1 - a
    base = lin
    for s in (sigma, 2 * sigma, 4 * sigma):  # widen until every accepted pixel has support
        est = masked_blur(lin, keep, s)
        ok = blur(keep, s) > 0.05
        base = torch.where((a > 0) & ok & (base == lin), torch.minimum(est, lin), base)
    return base, (lin - base).clamp_min(0)


def edit(base, layer, gain):
    return linear_to_srgb(base + gain * layer)


def edit_additive(x, signal, gain):
    """I + gain * signal on the sRGB image; signal (B, H, W) scaled to 0-1 by its 99.9th percentile per image."""
    s = signal / torch.quantile(signal.flatten(1), 0.999, dim=1).clamp_min(1e-12)[:, None, None]
    return (x + gain * s.clamp(0, 1)[:, None]).clamp(0, 1)


def edit_share(x, share, gain):
    """Edit with a per-pixel specular share (B, H, W) of the luminance: the specular part, share * luminance, is taken
    as achromatic (the colour of the light) and scaled by gain."""
    lin = srgb_to_linear(x)
    spec = share[:, None] * (lin * LUM.to(x.device)).sum(1, keepdim=True)
    return linear_to_srgb(lin + (gain - 1) * spec)
