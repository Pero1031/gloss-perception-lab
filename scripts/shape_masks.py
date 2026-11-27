"""Object, background and depth-from-silhouette masks for any geometry of the Serrano images.

For a fixed geometry the camera is fixed, so the silhouette is the same for every material and illumination. Under one
illumination the background is identical for all materials, hence the pixels whose values change with the material belong
to the object.
"""
import re
from collections import defaultdict

import numpy as np
import torch
from scipy import ndimage


def geometry_of(path):
    return re.match(r"\[.+?\]\[(.+?)\]\[", path).group(1)


def illum_of(path):
    return path.split("]")[0][1:]


def geometry_masks(ds, indices, n_per_illum=60, thresh=0.03, erode=10, bg_margin=8, max_hole=50):
    """Masks of the geometry shown by the rows `indices` of `ds`.

    Returns (obj, bg, rho, silhouette, agreement): obj = silhouette eroded by `erode` pixels, bg = pixels farther than
    `bg_margin` from the silhouette, rho = 1 - distance to the silhouette / its maximum (0 at the deepest point, 1 at
    the silhouette; equals the normalised radius for a sphere), agreement = fraction of pixels on which the silhouettes
    found under the individual illuminations agree with the final one.
    Only holes of at most `max_hole` pixels (at 256 x 256) are filled: larger enclosed regions are background seen
    through the object (between the arms of happy_vrip_vt, inside the teapot handle). bg is empty when the object
    fills the frame (surface2-1000x1000part).
    """
    by_illum = defaultdict(list)
    for i in indices:
        by_illum[illum_of(ds.rows[i][0])].append(i)
    votes = []
    for idx in by_illum.values():
        sel = idx[::max(1, len(idx) // n_per_illum)][:n_per_illum]
        votes.append((torch.stack([ds[i][0] for i in sel]).std(0).mean(0) > thresh).numpy())
    votes = np.stack(votes)
    sil = votes.mean(0) > 0.5
    sil = ndimage.binary_closing(sil, iterations=2)
    holes, n_holes = ndimage.label(ndimage.binary_fill_holes(sil) & ~sil)
    if n_holes:
        limit = max_hole * sil.size / 256 ** 2
        small = np.flatnonzero(ndimage.sum(np.ones_like(sil), holes, range(1, n_holes + 1)) <= limit) + 1
        sil = sil | np.isin(holes, small)
    labels, n = ndimage.label(sil)
    if n > 1:  # keep the largest component
        sil = labels == 1 + int(np.argmax(ndimage.sum(sil, labels, range(1, n + 1))))
    agreement = float((votes == sil[None]).mean())
    depth = ndimage.distance_transform_edt(sil)
    obj = depth > erode
    rho = 1 - depth / depth.max()
    bg = ndimage.distance_transform_edt(~sil) > bg_margin
    return torch.from_numpy(obj), torch.from_numpy(bg), torch.from_numpy(rho).float(), sil, agreement
