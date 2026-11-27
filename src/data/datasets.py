import csv
import re
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


class SyntheticGlossDataset(Dataset):
    """Blinn-Phong spheres. The target is the specular strength in [0, 1].

    Only meant for checking that the training pipeline runs and can learn something.
    """

    def __init__(self, n=2000, size=128, seed=0):
        self.n, self.size, self.seed = n, size, seed
        ys, xs = np.mgrid[-1:1:size * 1j, -1:1:size * 1j]
        r2 = xs**2 + ys**2
        self.mask = r2 <= 0.8**2
        z = np.sqrt(np.clip(0.8**2 - r2, 0, None))
        normal = np.stack([xs, -ys, z], axis=-1)
        self.normal = normal / np.linalg.norm(normal, axis=-1, keepdims=True).clip(1e-6)

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        rng = np.random.default_rng(self.seed * 1_000_003 + idx)
        ks = rng.uniform(0, 1)
        shininess = rng.uniform(30, 200)
        albedo = rng.uniform(0.2, 0.9, size=3)

        light = np.array([rng.uniform(-0.8, 0.8), rng.uniform(0.2, 0.9), 1.0])
        light /= np.linalg.norm(light)
        half = light + np.array([0, 0, 1.0])
        half /= np.linalg.norm(half)

        diffuse = np.clip(self.normal @ light, 0, None)[..., None] * albedo
        specular = ks * np.clip(self.normal @ half, 0, None)[..., None] ** shininess
        img = np.clip(diffuse + specular, 0, 1) ** (1 / 2.2)
        img = np.where(self.mask[..., None], img, 0.5)

        x = torch.from_numpy(img.transpose(2, 0, 1).astype(np.float32))
        return x, torch.tensor(ks, dtype=torch.float32)


def material_of(path):
    """Material part of a Serrano file name: [illumination][geometry][material].jpg"""
    return re.match(r"\[.+?\]\[.+?\]\[(.+)\]\.jpg$", path).group(1)


def material_groups(materials):
    """Map each material to a group of materials that share a source.

    A split that keeps groups together never puts a material in validation while a near-copy is in training.
    - MERL_EDIT-<base>[_diff_mix_<partner>]_...: edits of measured MERL materials. They are grouped with the
      plain MERL-<base> / MERL-<partner> materials they are built from (one big group).
    - UTIA_ISO-isotropic_mNNN_...: isotropic version of UTIA-mNNN, grouped with it.
    """
    edit_re = re.compile(r"MERL_EDIT-(.+?)(?:_diff_mix_(.+?)_spec)?_hue_")
    sources = set()
    for m in materials:
        e = edit_re.match(m)
        if e:
            sources.update(s for s in e.groups() if s)

    groups = {}
    for m in materials:
        if m.startswith("MERL_EDIT") or (m.startswith("MERL-") and m[len("MERL-"):] in sources):
            groups[m] = "MERL_EDIT"
        elif m.startswith("UTIA"):
            groups[m] = "UTIA-m" + re.search(r"m(\d{3})", m).group(1)
        else:
            groups[m] = m
    return groups


class ManifestDataset(Dataset):
    """Reads a CSV with columns `path,gloss`. `path` is relative to `root`.

    With cache=True all images are decoded and resized once up front and kept as uint8.
    """

    def __init__(self, csv_path, root, size=256, cache=False):
        self.root, self.size = Path(root), size
        with open(csv_path, newline="", encoding="utf-8") as f:
            self.rows = [(r["path"], float(r["gloss"])) for r in csv.DictReader(f)]
        self.images = [self._load(p) for p, _ in self.rows] if cache else None

    def _load(self, path):
        img = Image.open(self.root / path).convert("RGB").resize((self.size, self.size))
        return np.asarray(img)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        path, gloss = self.rows[idx]
        arr = self.images[idx] if self.images is not None else self._load(path)
        x = torch.from_numpy(arr.astype(np.float32).transpose(2, 0, 1) / 255.0)
        return x, torch.tensor(gloss, dtype=torch.float32)
