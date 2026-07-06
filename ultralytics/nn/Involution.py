import torch
import torch.nn as nn
from torch.nn.init import constant_, xavier_uniform_
from mmcv.cnn import ConvModule

from ultralytics.nn.modules import Conv


def caffe2_xavier_init(module, **kwargs):
    """Replace mmcv.cnn.caffe2_xavier_init with pure torch.nn.init.xavier_uniform_. Compatible with mmcv-lite 2.x."""
    if hasattr(module, "weight") and module.weight is not None:
        xavier_uniform_(module.weight)
    if hasattr(module, "bias") and module.bias is not None:
        module.bias.data.zero_()


def constant_init(module, val, bias=0):
    """Replace mmcv.cnn.constant_init with pure torch.nn.init.constant_. Compatible with mmcv-lite 2.x."""
    if hasattr(module, "weight") and module.weight is not None:
        constant_(module.weight, val)
    if hasattr(module, "bias") and module.bias is not None:
        constant_(module.bias, bias)

class Involution(nn.Module):

    def __init__(self,c1,c2,kernel_size,stride):
        super(Involution, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.c1 = c1
        reduction_ratio = 4
        self.group_channels = 16
        self.groups = self.c1 // self.group_channels
        self.conv1 = Conv(c1, c1 // reduction_ratio,1)
        self.conv2 = Conv(c1 // reduction_ratio,kernel_size**2 * self.groups,1,1)
           
        if stride > 1:
            self.avgpool = nn.AvgPool2d(stride, stride)
        self.unfold = nn.Unfold(kernel_size, 1, (kernel_size-1)//2, stride)    

    def forward(self, x):

        weight = self.conv2(self.conv1(x if self.stride == 1 else self.avgpool(x)))
        b, c, h, w = weight.shape
        weight = weight.view(b, self.groups, self.kernel_size**2, h, w).unsqueeze(2)
       #out = _involution_cuda(x, weight, stride=self.stride, padding=(self.kernel_size-1)//2)
        #print("weight shape:",weight.shape)
        out = self.unfold(x).view(b, self.groups, self.group_channels, self.kernel_size**2, h, w)
        #print("new out:",(weight*out).shape)
        out = (weight * out).sum(dim=3).view(b, self.c1, h, w)
     
        return out