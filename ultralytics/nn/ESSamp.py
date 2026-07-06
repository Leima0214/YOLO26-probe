import torch
import torch.nn as nn

from ultralytics.nn.modules import (
    Conv,
    DSConv,
    GhostConv,
)


class DSConv(nn.Module):
    def __init__(self, c1, c2, k=3, s=1, act=True, depth_multiplier=2):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(c1, c1 * depth_multiplier, kernel_size=k, stride=s, padding=k // 2, groups=c1, bias=False),
            nn.BatchNorm2d(c1 * depth_multiplier),
            nn.GELU() if act else nn.Identity(),
            nn.Conv2d(c1 * depth_multiplier, c2, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(c2),
            nn.GELU() if act else nn.Identity(),
        )

    def forward(self, x):
        return self.block(x)


class GhostConv(nn.Module):
    """Ghost Convolution https://github.com/huawei-noah/ghostnet."""

    def __init__(self, c1, c2, k=1, s=1, g=1, act=True):
        """Initializes Ghost Convolution module with primary and cheap operations for efficient feature learning."""
        super().__init__()
        c_ = c2 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, k, s, None, g, act=act)
        self.cv2 = Conv(c_, c_, 5, 1, None, c_, act=act)

    def forward(self, x):
        """Forward propagation through a Ghost Bottleneck layer with skip connection."""
        y = self.cv1(x)
        return torch.cat((y, self.cv2(y)), 1)


class ESSamp(nn.Module):
    def __init__(self, c1, c2, k=3, s=1, act=True, depth_multiplier=2):
        super().__init__()
        self.dsconv = DSConv(c1 * 4, c2, k=k, s=s, act=act, depth_multiplier=depth_multiplier)
        # self.GC=GhostConv(c1=64,c2=c1)
        self.slices = nn.PixelUnshuffle(2)

    def forward(self, x):
        # print(x.shape)
        # x=self.GC(x)
        x = self.slices(x)
        return self.dsconv(x)
