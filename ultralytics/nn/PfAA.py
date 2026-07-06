import torch
import torch.nn as nn


# 详细改进流程和操作，请关注B站博主：AI学术叫叫兽
class PfAAMLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        _b, _c, _h, _w = x.size()
        y = self.avg_pool(x)
        z = torch.mean(x, dim=1, keepdim=True)
        attention = self.sigmoid(y * z)
        return x * attention


# 详细改进流程和操作，请关注B站博主：AI学术叫叫兽
# 详细改进流程和操作，请关注B站博主：AI学术叫叫兽
