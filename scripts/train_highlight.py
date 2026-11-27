"""Monocular highlight network trained on the stereo scenes (targets from stereo_targets.py).

  --target residual   b2: predict from one (left) image what the two eyes reveal (left-right residual); no physics label
  --target spec       b1: predict the renderer's specular share of the luminance

The network sees a tone-mapped left image (random exposure: the object's 99.5th luminance percentile is mapped to
0.6-1.6, then clipped and gamma-encoded, like an 8-bit photo) and is trained on object pixels where the target is
defined. Output: sigmoid map trained against sqrt(target) (the residual is heavy-tailed). Scenes are split by index
(last --val share for validation).
"""
import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from src.models.unet import UNet

LUM = torch.tensor([0.2126, 0.7152, 0.0722]).view(1, 3, 1, 1)
GREY = 128 / 255  # background value (sRGB) when --bg grey


def tone_map(rgb, mask, gen=None, fixed=None, lo=0.6, hi=1.6):
    """Linear (B,3,H,W) -> 8-bit-like sRGB in [0,1]. The object's 99.5th luminance percentile is mapped to a random
    value, log-uniform in [lo, hi] (fixed: that value instead)."""
    y = (rgb * LUM.to(rgb.device)).sum(1)
    q = torch.stack([torch.quantile(yi[mi], 0.995) if mi.any() else yi.max() for yi, mi in zip(y, mask)])
    if fixed:
        target = torch.full_like(q, fixed)
    else:
        target = torch.exp(torch.empty_like(q).uniform_(np.log(lo), np.log(hi), generator=gen))
    x = (rgb * (target / q.clamp_min(1e-6)).view(-1, 1, 1, 1)).clamp(0, 1) ** (1 / 2.2)
    return torch.round(x * 255) / 255


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/stereo_ds_targets.npz")
    p.add_argument("--target", choices=["residual", "spec"], default="residual")
    p.add_argument("--out", default=None)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--base", type=int, default=16)
    p.add_argument("--depth", type=int, default=4)
    p.add_argument("--val", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--bg-sup", type=int, default=1, help="1: supervise the background with target 0")
    p.add_argument("--bg", choices=["scene", "grey"], default="grey",
                   help="grey: replace the background by uniform grey, so the net need not find the object itself")
    p.add_argument("--exp-lo", type=float, default=0.08,
                   help="exposure range (object 99.5th luminance percentile after scaling); Serrano matte images "
                        "have 0.13-0.25, glossy ones 0.4-0.8")
    p.add_argument("--exp-hi", type=float, default=1.6)
    p.add_argument("--zoom", type=float, default=0.45,
                   help="random crops of this to 1 times the frame side (objects fill 17-30%% of the training "
                        "frames but the whole frame in many Serrano images); 1 = off")
    args = p.parse_args()
    out = args.out or f"runs/highlight_{args.target}"
    os.makedirs(out, exist_ok=True)
    torch.manual_seed(args.seed)

    d = np.load(args.data)
    rgb = torch.from_numpy(d["rgb"])  # float16 (N,3,H,W)
    tgt = torch.from_numpy(d[args.target].astype(np.float32)).clamp(0, 1).sqrt()
    sup = torch.from_numpy(d["valid"] if args.target == "residual" else d["mask"])
    mask = torch.from_numpy(d["mask"])
    if args.bg_sup:  # background: target 0 (no highlight), so the net must find the object itself
        tgt = torch.where(mask, tgt, torch.zeros_like(tgt))
        sup = sup | ~mask
    n = len(rgb)
    n_val = max(1, int(n * args.val))
    tr, va = np.arange(n - n_val), np.arange(n - n_val, n)
    print(f"{n} scenes (train {len(tr)}, val {len(va)}), target {args.target}, supervised pixels "
          f"{sup.float().mean():.3f}", flush=True)

    dev = "cuda"
    model = UNet(3, 1, args.base, args.depth, head="map").to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)
    gen = torch.Generator().manual_seed(args.seed)

    def zoom(x, t, s, m):
        """Random square crop (side zoom..1 of the frame, kept inside it) resized back to the full size."""
        B = len(x)
        sc = torch.empty(B).uniform_(args.zoom, 1, generator=gen)
        off = (torch.rand(B, 2, generator=gen) * 2 - 1) * (1 - sc)[:, None]
        theta = torch.zeros(B, 2, 3)
        theta[:, 0, 0] = theta[:, 1, 1] = sc
        theta[:, :, 2] = off
        grid = F.affine_grid(theta, list(x.shape), align_corners=False)
        g = lambda a, mode: F.grid_sample(a, grid, mode=mode, align_corners=False)
        return (g(x, "bilinear"), g(t, "bilinear"), g(s, "nearest"),
                g(m[:, None].float(), "nearest")[:, 0] > 0.5)

    def batch(idx, train):
        x = rgb[idx].float()
        m = mask[idx]
        t, s = tgt[idx][:, None], sup[idx][:, None].float()
        if train and args.zoom < 1:
            x, t, s, m = zoom(x, t, s, m)
        x = tone_map(x, m, gen, lo=args.exp_lo, hi=args.exp_hi) if train else tone_map(x, m, fixed=1.0)
        if args.bg == "grey":  # background removed; Serrano test images get the same treatment from their masks
            x = torch.where(m[:, None], x, torch.full_like(x, GREY))
        if train and torch.rand(1, generator=gen) < 0.5:
            x, t, s = x.flip(-1), t.flip(-1), s.flip(-1)
        return x.to(dev), t.to(dev), s.to(dev)

    log = []
    for ep in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        perm = tr[torch.randperm(len(tr), generator=gen).numpy()]
        tot = 0.0
        for i in range(0, len(perm), args.batch):
            x, t, s = batch(perm[i:i + args.batch], True)
            pred = torch.sigmoid(model(x))
            loss = ((pred - t) ** 2 * s).sum() / s.sum().clamp_min(1)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item() * len(x)
        sched.step()
        model.eval()
        se, cnt, preds, tgts = 0.0, 0.0, [], []
        with torch.no_grad():
            for i in range(0, len(va), 64):
                x, t, s = batch(va[i:i + 64], False)
                pred = torch.sigmoid(model(x))
                se += ((pred - t) ** 2 * s).sum().item()
                cnt += s.sum().item()
                obj = (s > 0) & mask[va[i:i + 64]][:, None].to(dev)  # object pixels only (comparable to bg-sup=0)
                preds.append(pred[obj].cpu())
                tgts.append(t[obj].cpu())
        pv, tv = torch.cat(preds), torch.cat(tgts)
        r = torch.corrcoef(torch.stack([pv, tv]))[0, 1].item()
        log.append(dict(epoch=ep, train_mse=tot / len(tr), val_mse=se / cnt, val_r=r))
        print(f"epoch {ep:3d}  train_mse={tot / len(tr):.5f}  val_mse={se / cnt:.5f}  val_r(object pixels)={r:.3f}  "
              f"({time.time() - t0:.0f}s)", flush=True)
        torch.save(dict(model=model.state_dict(), args=vars(args), epoch=ep), os.path.join(out, "last.pt"))
    json.dump(log, open(os.path.join(out, "log.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
