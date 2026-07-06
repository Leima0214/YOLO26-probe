import torch.nn as nn

# 详细改进流程和操作，请关注B站博主：AI学术叫叫兽

# from mmcv.cnn.bricks import DropPath


class AFF(nn.Module):
    def __init__(self, channels=64, r=4):
        super().__init__()
        inter_channels = int(channels // r)
        # 详细改进流程和操作，请关注B站博主：AI学术叫叫兽
        self.local_att = nn.Sequential(
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )

        self.global_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),  # 详细改进流程和操作，请关注B站博主：AI学术叫叫兽
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x, residual):
        xa = x + residual
        xl = self.local_att(xa)
        xg = self.global_att(xa)
        xlg = xl + xg
        wei = self.sigmoid(xlg)

        xo = 2 * x * wei + 2 * residual * (1 - wei)
        return xo


# 详细改进流程和操作，请关注B站博主：AI学术叫叫兽
