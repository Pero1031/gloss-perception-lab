import torch.nn as nn


class TinyGlossNet(nn.Module):
    """Tiny gloss network in the style of Morimoto et al. (2026).

    `kernels` linear 15x15 filters (no ReLU, no normalization), global max pooling, linear regression.
    Convolution is "valid" so the response map has no border artifacts; response() returns it for visualisation.
    """

    def __init__(self, kernels=1, kernel_size=15, in_ch=3):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, kernels, kernel_size, bias=False)
        self.fc = nn.Linear(kernels, 1)

    def response(self, x):
        return self.conv(x)

    def forward(self, x):
        return self.fc(self.response(x).amax(dim=(2, 3)))[:, 0]
