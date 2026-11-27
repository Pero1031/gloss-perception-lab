import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class GlossFeatures(nn.Module):
    """Multi-scale input features of the gloss detector: luminance plus rectified LoG responses (after Nagata et al.).

    For each scale sigma a centre-positive Laplacian-of-Gaussian kernel is applied to the luminance
    image and multiplied by sigma. Negative responses are set to 0. The responses of all
    scales and the luminance itself are stacked as channels. Each kernel is truncated at 3 sigma and
    shifted to sum to zero.
    Output: (B, 1 + len(sigmas), H, W). Not learnable; the kernels are applied by FFT because sigma goes up to ~40 px.
    """

    def __init__(self, sigma_max=39, sigma_step=2):
        super().__init__()
        self.sigmas = list(range(1, sigma_max + 1, sigma_step))
        self.radius = math.ceil(3 * self.sigmas[-1])
        self._cache = {}

    @property
    def out_channels(self):
        return 1 + len(self.sigmas)

    def _kernels(self, n, device):
        key = (n, str(device))
        if key not in self._cache:
            # signed distance from the origin, which sits at index 0 for circular convolution
            d = (torch.arange(n, device=device) + n // 2) % n - n // 2
            r2 = (d[:, None] ** 2 + d[None, :] ** 2).double()
            ks = []
            for s in self.sigmas:
                mask = r2 <= (3 * s) ** 2
                k = (2 * s**2 - r2) / (2 * math.pi * s**6) * torch.exp(-r2 / (2 * s**2)) * mask
                # truncating at 3 sigma leaves a small non-zero sum; remove it so flat regions give no response
                k = (k - k.sum() / mask.sum()) * mask
                ks.append(s * k)
            self._cache[key] = torch.fft.rfft2(torch.stack(ks).float())
        return self._cache[key]

    @torch.no_grad()
    def forward(self, x):
        lum = 0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]
        h, w = lum.shape[-2:]
        r = self.radius
        n = 1 << (max(h, w) + 2 * r - 1).bit_length()
        spec = torch.fft.rfft2(F.pad(lum, (r, r, r, r), mode="replicate"), s=(n, n))
        resp = torch.fft.irfft2(spec * self._kernels(n, x.device), s=(n, n))
        resp = resp[..., r:r + h, r:r + w].clamp_min(0)
        return torch.cat([lum, resp], dim=1)

    @torch.no_grad()
    def gloss_map(self, x, product=True):
        """Gloss map of Nagata et al.'s detector: (B, H, W).

        product=True: mean over all scale pairs i<j of p(sigma_i) * p(sigma_j), which suppresses edges and noise
        that respond at only some scales. product=False: plain mean of p(sigma) over scales.
        """
        return self.gloss_map_from(self(x)[:, 1:], product)

    @staticmethod
    def gloss_map_from(p, product=True):
        """Same as gloss_map, from already computed scale responses p of shape (B, n_scales, H, W)."""
        m = p.shape[1]
        if not product:
            return p.sum(1) / m
        # sum over pairs i<j of p_i p_j = ((sum p)^2 - sum p^2) / 2, divided by the number of pairs m(m-1)/2
        return (p.sum(1) ** 2 - (p**2).sum(1)) / (m * (m - 1))
