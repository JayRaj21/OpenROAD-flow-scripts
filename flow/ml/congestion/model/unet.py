"""
U-Net architecture for congestion prediction.

Input:  (B, 1, H, W)  -- normalised cell-density grid
Output: (B, 10, H, W) -- predicted congestion per routing layer (10 layers)

The spatial dimensions H and W are determined by the grid_size used during
data collection (default 64). The model uses padding=1 throughout so input
and output spatial dimensions match exactly.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class Down(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = ConvBlock(in_ch, out_ch)

    def forward(self, x):
        return self.conv(self.pool(x))


class Up(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, in_ch // 2, kernel_size=2, stride=2)
        self.conv = ConvBlock(in_ch, out_ch)

    def forward(self, x, skip):
        x = self.up(x)
        if x.shape != skip.shape:
            x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=False)
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


class CongestionUNet(nn.Module):
    """
    U-Net that predicts per-layer routing congestion from a placement density map.

    Args:
        in_channels:   Number of input channels (1 for a single density grid,
                       or more if additional features are stacked)
        out_channels:  Number of output channels (10 for metal1-metal10)
        base_features: Width of the first encoder block; doubles at each depth
    """

    def __init__(self, in_channels: int = 1, out_channels: int = 10, base_features: int = 32):
        super().__init__()
        f = base_features

        self.enc1 = ConvBlock(in_channels, f)
        self.enc2 = Down(f, f * 2)
        self.enc3 = Down(f * 2, f * 4)
        self.enc4 = Down(f * 4, f * 8)

        self.bottleneck = Down(f * 8, f * 16)

        self.dec4 = Up(f * 16, f * 8)
        self.dec3 = Up(f * 8, f * 4)
        self.dec2 = Up(f * 4, f * 2)
        self.dec1 = Up(f * 2, f)

        self.head = nn.Conv2d(f, out_channels, kernel_size=1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)

        b = self.bottleneck(e4)

        d4 = self.dec4(b, e4)
        d3 = self.dec3(d4, e3)
        d2 = self.dec2(d3, e2)
        d1 = self.dec1(d2, e1)

        return torch.sigmoid(self.head(d1))


if __name__ == "__main__":
    model = CongestionUNet()
    x = torch.randn(2, 1, 64, 64)
    y = model(x)
    print(f"Input:  {x.shape}")
    print(f"Output: {y.shape}")
    total = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {total:,}")
