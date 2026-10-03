"""
Fourier Neural Operator for the thermal arch-sweep comparison.

Built from scratch (no `neuraloperator` dependency), per the project's
minimal-dependency convention — physics motivation: HotSpot's steady-state
heat diffusion is a global spectral smoothing operator, which a Fourier
layer represents directly, unlike a U-Net's local convolutions.

Input:  (B, in_channels, 64, 64)
Output: (B, 1, 64, 64) in (0, 1) (sigmoid, matching HeatmapHead's range)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SpectralConv2d(nn.Module):
    """Global convolution via truncated 2-D FFT: multiply the two
    low-frequency corners of the spectrum by learned complex weights,
    zero elsewhere."""

    def __init__(self, in_channels: int, out_channels: int, modes1: int, modes2: int):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2
        scale = 1.0 / (in_channels * out_channels)
        self.weights1 = nn.Parameter(
            scale * torch.rand(in_channels, out_channels, modes1, modes2, dtype=torch.cfloat)
        )
        self.weights2 = nn.Parameter(
            scale * torch.rand(in_channels, out_channels, modes1, modes2, dtype=torch.cfloat)
        )

    @staticmethod
    def _compl_mul2d(x, weights):
        return torch.einsum("bixy,ioxy->boxy", x, weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, _, h, w = x.shape
        x_ft = torch.fft.rfft2(x)
        out_ft = torch.zeros(
            batch, self.out_channels, h, w // 2 + 1, dtype=torch.cfloat, device=x.device
        )
        out_ft[:, :, : self.modes1, : self.modes2] = self._compl_mul2d(
            x_ft[:, :, : self.modes1, : self.modes2], self.weights1
        )
        out_ft[:, :, -self.modes1 :, : self.modes2] = self._compl_mul2d(
            x_ft[:, :, -self.modes1 :, : self.modes2], self.weights2
        )
        return torch.fft.irfft2(out_ft, s=(h, w))


class FNO2d(nn.Module):
    """4-layer FNO: lift to `width` channels (with coordinate grid appended),
    `n_layers` Fourier blocks, project back to a single sigmoid output."""

    def __init__(
        self,
        in_channels: int = 5,
        width: int = 32,
        modes: int = 12,
        n_layers: int = 4,
        padding: int = 8,
    ):
        super().__init__()
        self.width = width
        self.n_layers = n_layers
        self.padding = padding

        self.lift = nn.Linear(in_channels + 2, width)
        self.spectral_convs = nn.ModuleList(
            [SpectralConv2d(width, width, modes, modes) for _ in range(n_layers)]
        )
        self.pointwise_convs = nn.ModuleList(
            [nn.Conv2d(width, width, 1) for _ in range(n_layers)]
        )
        self.fc1 = nn.Linear(width, 128)
        self.fc2 = nn.Linear(128, 1)

    @staticmethod
    def _coord_grid(batch: int, h: int, w: int, device) -> torch.Tensor:
        gy = torch.linspace(0, 1, h, device=device).view(1, 1, h, 1).expand(batch, 1, h, w)
        gx = torch.linspace(0, 1, w, device=device).view(1, 1, 1, w).expand(batch, 1, h, w)
        return torch.cat([gy, gx], dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, _, h, w = x.shape
        grid = self._coord_grid(batch, h, w, x.device)
        x = torch.cat([x, grid], dim=1).permute(0, 2, 3, 1)
        x = self.lift(x).permute(0, 3, 1, 2)
        x = F.pad(x, [0, self.padding, 0, self.padding])

        for i, (spec, pw) in enumerate(zip(self.spectral_convs, self.pointwise_convs)):
            x = spec(x) + pw(x)
            if i < self.n_layers - 1:
                x = F.gelu(x)

        x = x[:, :, :h, :w].permute(0, 2, 3, 1)
        x = F.gelu(self.fc1(x))
        x = self.fc2(x).permute(0, 3, 1, 2)
        return torch.sigmoid(x)
