# Ultralytics NN compatibility layer
# Drop-in replacements for mmcv, timm, einops — no external deps beyond torch.
# Import from here instead of the original packages.

import math
import torch
import torch.nn as nn
from torch.nn.init import constant_ as _torch_constant, xavier_uniform_

# ── einops.rearrange ──────────────────────────────────────────────────────────

try:
    from einops import rearrange  # noqa: F401
except ImportError:
    def rearrange(tensor, pattern, **axes_lengths):
        """Minimal einops.rearrange fallback for commonly used patterns."""
        if pattern == "b c h w -> b (h w) c":
            return tensor.flatten(2).transpose(1, 2)
        if pattern == "b (h w) c -> b c h w":
            h, w = axes_lengths["h"], axes_lengths["w"]
            return tensor.transpose(1, 2).reshape(tensor.size(0), -1, h, w)
        if pattern == "b (head c) h w -> b head c (h w)":
            head = axes_lengths["head"]
            B, HC, H, W = tensor.shape
            C = HC // head
            return tensor.view(B, head, C, H, W).flatten(3)
        if pattern == "b head c (h w) -> b (head c) h w":
            head = axes_lengths["head"]
            h, w = axes_lengths["h"], axes_lengths["w"]
            B, Hd, C, HW = tensor.shape
            return tensor.view(B, Hd, C, h, w).reshape(B, Hd * C, h, w)
        if pattern == "b n (h d) -> b h n d":
            h = axes_lengths["h"]
            B, N, HD = tensor.shape
            D = HD // h
            return tensor.view(B, N, h, D).transpose(1, 2)
        if pattern == "b (h d) n -> b h n d":
            h = axes_lengths.get("h")
            B, HD, N = tensor.shape
            if h is None:
                h = HD  # assume h=HD, d=1
            D = HD // h
            return tensor.view(B, h, D, N).transpose(2, 3)
        if pattern == "b n c h w -> (b n) c h w":
            return tensor.reshape(-1, *tensor.shape[2:])
        raise NotImplementedError(
            f"rearrange fallback does not support pattern '{pattern}'. "
            f"Install einops: pip install einops"
        )

# ── timm DropPath ─────────────────────────────────────────────────────────────

class DropPath(nn.Module):
    """Stochastic Depth / DropPath layer. Drops entire samples with probability drop_prob."""

    def __init__(self, drop_prob=0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x.div(keep_prob) * random_tensor


try:
    from timm.models.layers import DropPath as _timm_DropPath  # noqa: F401
    DropPath = _timm_DropPath
except ImportError:
    pass  # use our DropPath above

# ── timm SqueezeExcite ────────────────────────────────────────────────────────

try:
    from timm.models.layers import SqueezeExcite  # noqa: F401
except ImportError:
    class SqueezeExcite(nn.Module):
        """Pure-PyTorch Squeeze-and-Excitation block, matches timm API."""

        def __init__(self, in_chs, rd_ratio=0.25, rd_channels=None,
                     act_layer=nn.ReLU, gate_layer=nn.Sigmoid):
            super().__init__()
            rd_channels = rd_channels or max(1, int(in_chs * rd_ratio))
            self.fc1 = nn.Conv2d(in_chs, rd_channels, 1)
            self.act = act_layer(inplace=True) if act_layer == nn.ReLU else act_layer()
            self.fc2 = nn.Conv2d(rd_channels, in_chs, 1)
            self.gate = gate_layer()

        def forward(self, x):
            x_se = x.mean((2, 3), keepdim=True)
            x_se = self.fc1(x_se)
            x_se = self.act(x_se)
            x_se = self.fc2(x_se)
            return x * self.gate(x_se)

# ── timm trunc_normal_ / to_2tuple / register_model ─────────────────────────

try:
    from timm.models.layers import trunc_normal_  # noqa: F401
except ImportError:
    def trunc_normal_(tensor, mean=0.0, std=1.0, a=-2.0, b=2.0):
        """Truncated normal initializer fallback."""
        return torch.nn.init.trunc_normal_(tensor, mean, std, a, b)


def to_2tuple(x):
    """Convert scalar to 2-tuple if needed."""
    return x if isinstance(x, (tuple, list)) and len(x) == 2 else (x, x)


try:
    from timm.models.registry import register_model  # noqa: F401
except ImportError:
    def register_model(func):
        """No-op register_model fallback when timm is not installed."""
        return func

# ── mmcv ConvModule ──────────────────────────────────────────────────────────

class ConvModule(nn.Module):
    """Minimal drop-in replacement for mmcv.cnn.ConvModule (no mmcv dependency).

    Exposes a .conv attribute (the inner Conv2d) for weight init compatibility.
    """

    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1,
                 padding=0, conv_cfg=None, act_cfg=None, norm_cfg=None, **kwargs):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, **kwargs)

    def forward(self, x):
        return self.conv(x)

# ── mmcv weight init helpers ─────────────────────────────────────────────────

try:
    from mmengine.model.weight_init import trunc_normal_init, normal_init
except ImportError:
    def trunc_normal_init(module, mean=0.0, std=1.0, bias=0.0):
        """Drop-in for mmengine's trunc_normal_init."""
        for m in module.modules():
            if hasattr(m, "weight") and m.weight is not None:
                torch.nn.init.trunc_normal_(m.weight, mean, std)
            if hasattr(m, "bias") and m.bias is not None:
                torch.nn.init.constant_(m.bias, bias)

    def normal_init(module, mean=0.0, std=1.0, bias=0.0):
        """Drop-in for mmengine's normal_init."""
        for m in module.modules():
            if hasattr(m, "weight") and m.weight is not None:
                torch.nn.init.normal_(m.weight, mean, std)
            if hasattr(m, "bias") and m.bias is not None:
                torch.nn.init.constant_(m.bias, bias)


def constant_init(module, val, bias=0):
    """Drop-in for mmcv/mmengine constant_init."""
    if hasattr(module, "weight") and module.weight is not None:
        _torch_constant(module.weight, val)
    if hasattr(module, "bias") and module.bias is not None:
        _torch_constant(module.bias, bias)


def caffe2_xavier_init(module, **kwargs):
    """Drop-in for mmcv caffe2_xavier_init."""
    if hasattr(module, "weight") and module.weight is not None:
        xavier_uniform_(module.weight)
    if hasattr(module, "bias") and module.bias is not None:
        module.bias.data.zero_()
