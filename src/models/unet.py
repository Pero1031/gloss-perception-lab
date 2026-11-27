import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class UNet(nn.Module):
    """U-Net with a switchable output head.

    head="map":    returns (B, out_ch, H, W), e.g. a per-pixel gloss map.
    head="scalar": averages the output map spatially and returns (B,), e.g. a gloss rating.
    """

    def __init__(self, in_ch=3, out_ch=1, base=32, depth=4, head="scalar"):
        super().__init__()
        assert head in ("map", "scalar")
        self.head = head

        chans = [base * 2**i for i in range(depth + 1)]
        self.down = nn.ModuleList()
        prev = in_ch
        for c in chans[:-1]:
            self.down.append(DoubleConv(prev, c))
            prev = c
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = DoubleConv(chans[-2], chans[-1])

        self.up = nn.ModuleList()
        self.dec = nn.ModuleList()
        for c_hi, c_lo in zip(chans[:0:-1], chans[-2::-1]):
            self.up.append(nn.ConvTranspose2d(c_hi, c_lo, 2, stride=2))
            self.dec.append(DoubleConv(c_lo * 2, c_lo))
        self.out_conv = nn.Conv2d(chans[0], out_ch, 1)

    def forward(self, x):
        skips = []
        for down in self.down:
            x = down(x)
            skips.append(x)
            x = self.pool(x)
        x = self.bottleneck(x)
        for up, dec, skip in zip(self.up, self.dec, reversed(skips)):
            x = up(x)
            x = dec(torch.cat([skip, x], dim=1))
        x = self.out_conv(x)
        if self.head == "scalar":
            return x.mean(dim=(1, 2, 3))
        return x
