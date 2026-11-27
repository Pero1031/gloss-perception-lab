import argparse
import random
import re
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler

from src.data.datasets import ManifestDataset, SyntheticGlossDataset, material_groups, material_of
from src.models.features import GlossFeatures
from src.models.tiny import TinyGlossNet
from src.models.unet import UNet

N_BINS = 7  # ratings are 1-7, rescaled to 0-1


def pearson(a, b):
    a, b = a - a.mean(), b - b.mean()
    return (a * b).sum() / (a.norm() * b.norm()).clamp_min(1e-8)


def rating_bins(y):
    return (y * (N_BINS - 1)).round().long().clamp(0, N_BINS - 1)


def build_dataset(args):
    if args.dataset == "synthetic":
        return SyntheticGlossDataset(n=args.n_synthetic, size=args.size)
    return ManifestDataset(args.manifest, args.root, size=args.size, cache=args.cache)


def split_indices(ds, val_frac, seed):
    """Hold out whole material groups for manifest data; plain random split otherwise.

    A group larger than the validation budget cannot be held out and always goes to training.
    """
    rng = random.Random(seed)
    if not hasattr(ds, "rows"):
        idx = list(range(len(ds)))
        rng.shuffle(idx)
        n_val = max(1, int(len(idx) * val_frac))
        return idx[n_val:], idx[:n_val]
    group_of = material_groups({material_of(p) for p, _ in ds.rows})
    members = {}
    for i, (p, _) in enumerate(ds.rows):
        members.setdefault(group_of[material_of(p)], []).append(i)
    budget = int(len(ds.rows) * val_frac)
    names = sorted(members)
    rng.shuffle(names)
    val_groups, n_val = set(), 0
    for g in names:
        if n_val + len(members[g]) <= budget:
            val_groups.add(g)
            n_val += len(members[g])
    train = [i for g, idx in members.items() if g not in val_groups for i in idx]
    val = [i for g in val_groups for i in members[g]]
    return sorted(train), sorted(val)


def metrics(preds, targets):
    """mse and r on the natural distribution, plus mse averaged over rating bins."""
    bins = rating_bins(targets)
    per_bin = [F.mse_loss(preds[bins == b], targets[bins == b]).item()
               for b in range(N_BINS) if (bins == b).any()]
    return {
        "mse": F.mse_loss(preds, targets).item(),
        "bal_mse": sum(per_bin) / len(per_bin),
        "r": pearson(preds, targets).item(),
    }


def predict(model, loader, device):
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for x, y in loader:
            preds.append(model(x.to(device)).cpu())
            targets.append(y)
    return torch.cat(preds), torch.cat(targets)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=["synthetic", "manifest"], default="synthetic")
    p.add_argument("--manifest", help="CSV with columns path,gloss")
    p.add_argument("--root", help="directory the manifest paths are relative to")
    p.add_argument("--cache", action=argparse.BooleanOptionalAction, default=True,
                   help="keep all manifest images in memory")
    p.add_argument("--n-synthetic", type=int, default=2000)
    p.add_argument("--size", type=int, default=128)
    p.add_argument("--model", choices=["unet", "tiny"], default="unet",
                   help="tiny: --kernels linear filters + global max pooling + linear regression")
    p.add_argument("--kernels", type=int, default=1)
    p.add_argument("--kernel-size", type=int, default=15)
    p.add_argument("--lr-decay-every", type=int, default=0, help="multiply the learning rate by 0.1 every N epochs")
    p.add_argument("--input", choices=["rgb", "multiscale"], default="rgb",
                   help="multiscale: luminance + LoG responses at several scales (21 channels)")
    p.add_argument("--sigma-max", type=int, default=39,
                   help="largest scale of the multiscale input; scales are 1,3,...,sigma-max (39 gives 21 channels)")
    p.add_argument("--head", choices=["scalar", "map"], default="scalar")
    p.add_argument("--base", type=int, default=32)
    p.add_argument("--depth", type=int, default=4)
    p.add_argument("--balance", choices=["sampler", "none"], default="sampler",
                   help="sampler: draw training images with probability inversely proportional to their rating bin's size")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--val-frac", type=float, default=0.2)
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--holdout-geometry", help="train on the other geometries of the manifest (training materials only) "
                                              "and validate on this one (validation materials)")
    p.add_argument("--illum-fold", type=int, choices=[0, 1, 2], default=None,
                   help="also hold out the illuminations k, k+3, k+6 (sorted names): train on the training materials "
                        "under the other illuminations, validate on the validation materials under these three")
    p.add_argument("--out", default="runs/default")
    args = p.parse_args()

    if args.head == "map":
        raise SystemExit("--head map needs map targets; the datasets here only provide scalar ratings")

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    ds = build_dataset(args)
    train_idx, val_idx = split_indices(ds, args.val_frac, args.seed)
    if args.illum_fold is not None:
        illum = [pth.split("]")[0][1:] for pth, _ in ds.rows]
        names = sorted(set(illum))
        held = {names[i] for i in range(args.illum_fold, len(names), 3)}
        train_idx = [i for i in train_idx if illum[i] not in held]
        val_idx = [i for i in val_idx if illum[i] in held]
        print(f"held-out illuminations: {sorted(held)}")
    if args.holdout_geometry:
        geometry = [re.match(r"\[.+?\]\[(.+?)\]\[", pth).group(1) for pth, _ in ds.rows]
        train_idx = [i for i in train_idx if geometry[i] != args.holdout_geometry]
        val_idx = [i for i in val_idx if geometry[i] == args.holdout_geometry]
        print(f"held-out geometry: {args.holdout_geometry}")
    train_ds, val_ds = Subset(ds, train_idx), Subset(ds, val_idx)

    sampler = None
    if args.balance == "sampler" and hasattr(ds, "rows"):
        bins = rating_bins(torch.tensor([ds.rows[i][1] for i in train_idx]))
        weights = 1.0 / torch.bincount(bins, minlength=N_BINS).float()[bins]
        sampler = WeightedRandomSampler(weights, num_samples=len(train_idx), replacement=True)
    train_loader = DataLoader(train_ds, args.batch_size, shuffle=sampler is None, sampler=sampler,
                              num_workers=args.workers)
    val_loader = DataLoader(val_ds, args.batch_size, num_workers=args.workers)

    if args.model == "tiny":
        model = TinyGlossNet(args.kernels, args.kernel_size)
    elif args.input == "multiscale":
        features = GlossFeatures(args.sigma_max)
        model = nn.Sequential(features, UNet(features.out_channels, 1, args.base, args.depth, args.head))
    else:
        model = UNet(base=args.base, depth=args.depth, head=args.head)
    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.StepLR(opt, args.lr_decay_every, 0.1) if args.lr_decay_every else None
    print(f"device={device} train={len(train_ds)} val={len(val_ds)} balance={'sampler' if sampler else 'none'} "
          f"params={sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

    if hasattr(ds, "rows"):
        train_y = torch.tensor([ds.rows[i][1] for i in train_idx])
        val_y = torch.tensor([ds.rows[i][1] for i in val_idx])
        base = metrics(torch.full_like(val_y, train_y.mean().item()), val_y)
        print(f"baseline (always predict train mean {train_y.mean():.3f}): "
              f"val_mse={base['mse']:.4f} val_bal_mse={base['bal_mse']:.4f}")

    best = float("inf")
    for epoch in range(1, args.epochs + 1):
        model.train()
        t0, total = time.time(), 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            loss = F.mse_loss(model(x), y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * len(x)
        if sched:
            sched.step()
        m = metrics(*predict(model, val_loader, device))
        print(f"epoch {epoch:3d}  train_mse={total / len(train_ds):.4f}  val_mse={m['mse']:.4f}  "
              f"val_bal_mse={m['bal_mse']:.4f}  val_r={m['r']:.3f}  ({time.time() - t0:.0f}s)", flush=True)
        if m["bal_mse"] < best:
            best = m["bal_mse"]
            torch.save({"model": model.state_dict(), "args": vars(args)}, out_dir / "best.pt")
        torch.save({"model": model.state_dict(), "args": vars(args)}, out_dir / "last.pt")

    model.load_state_dict(torch.load(out_dir / "best.pt")["model"])
    preds, targets = predict(model, val_loader, device)
    bins = rating_bins(targets)
    print("best model, mean prediction per rating bin (rating: n, mean pred, target):")
    for b in range(N_BINS):
        if (bins == b).any():
            print(f"  {b + 1}: n={(bins == b).sum().item():4d}  pred={preds[bins == b].mean():.3f}  "
                  f"target={targets[bins == b].mean():.3f}")


if __name__ == "__main__":
    main()
