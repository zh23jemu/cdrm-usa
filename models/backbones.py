import math
from typing import Optional, List

import torch
import torch.nn as nn
import torch.nn.functional as F


def _conv1d(in_c: int, out_c: int, k: int, s: int = 1, p: Optional[int] = None) -> nn.Conv1d:
    if p is None:
        p = k // 2
    return nn.Conv1d(in_c, out_c, kernel_size=k, stride=s, padding=p, bias=False)


class BasicBlock1D(nn.Module):
    expansion = 1

    def __init__(self, in_c: int, out_c: int, stride: int = 1, dropout: float = 0.1) -> None:
        super().__init__()
        self.conv1 = _conv1d(in_c, out_c, k=3, s=stride)
        self.bn1 = nn.BatchNorm1d(out_c)
        self.conv2 = _conv1d(out_c, out_c, k=3, s=1)
        self.bn2 = nn.BatchNorm1d(out_c)
        self.drop = nn.Dropout(dropout)
        if stride != 1 or in_c != out_c:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_c, out_c, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_c),
            )
        else:
            self.shortcut = nn.Identity()
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.act(self.bn1(self.conv1(x)))
        y = self.drop(y)
        y = self.bn2(self.conv2(y))
        y = y + self.shortcut(x)
        return self.act(y)


class ResNet1D(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        base_channels: int = 32,
        layers: List[int] = (2, 2, 2, 2),
        feat_dim: int = 256,
        dropout: float = 0.1,
        large_kernel: int = 15,
    ) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, base_channels, kernel_size=large_kernel, stride=2, padding=large_kernel // 2, bias=False),
            nn.BatchNorm1d(base_channels),
            nn.GELU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
        )
        channels = [base_channels, base_channels * 2, base_channels * 4, base_channels * 8]
        strides = [1, 2, 2, 2]
        self.stages = nn.ModuleList()
        prev = base_channels
        for c, s, n in zip(channels, strides, layers):
            blocks = []
            for i in range(n):
                blocks.append(BasicBlock1D(prev, c, stride=s if i == 0 else 1, dropout=dropout))
                prev = c
            self.stages.append(nn.Sequential(*blocks))
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.proj = nn.Sequential(
            nn.Linear(channels[-1], feat_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.out_dim = feat_dim

    def forward(self, x: torch.Tensor, return_stage_feats: bool = False) -> torch.Tensor:
        h = self.stem(x)
        feats = []
        for stage in self.stages:
            h = stage(h)
            feats.append(h)
        z = self.pool(h).squeeze(-1)
        z = self.proj(z)
        if return_stage_feats:
            return z, feats
        return z


class WDCNN(nn.Module):
    def __init__(self, in_channels: int = 1, feat_dim: int = 256) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(in_channels, 16, kernel_size=64, stride=16, padding=24),
            nn.BatchNorm1d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2, 2),
            nn.Conv1d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2, 2),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2, 2),
            nn.Conv1d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2, 2),
            nn.Conv1d(64, 64, kernel_size=3, padding=0),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1),
        )
        self.proj = nn.Linear(64, feat_dim)
        self.out_dim = feat_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.features(x).squeeze(-1)
        return self.proj(h)


class _LayerNorm2d(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / (s + self.eps).sqrt()
        return x * self.weight[None, :, None, None] + self.bias[None, :, None, None]


class _ConvNeXtBlock2D(nn.Module):
    def __init__(self, dim: int, drop_path: float = 0.0) -> None:
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = _LayerNorm2d(dim)
        self.pwconv1 = nn.Conv2d(dim, 4 * dim, kernel_size=1)
        self.act = nn.GELU()
        self.pwconv2 = nn.Conv2d(4 * dim, dim, kernel_size=1)
        self.drop_path = drop_path

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        inp = x
        x = self.dwconv(x)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.training and self.drop_path > 0.0:
            keep = 1.0 - self.drop_path
            mask = torch.bernoulli(torch.full((x.size(0), 1, 1, 1), keep, device=x.device))
            x = x * mask / keep
        return inp + x


class TFConvNeXtTiny(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        depths: List[int] = (2, 2, 4, 2),
        dims: List[int] = (48, 96, 192, 384),
        feat_dim: int = 256,
        drop_path: float = 0.1,
    ) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, dims[0], kernel_size=4, stride=4),
            _LayerNorm2d(dims[0]),
        )
        self.stages = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        for i in range(4):
            stage_blocks = nn.Sequential(
                *[_ConvNeXtBlock2D(dims[i], drop_path=drop_path) for _ in range(depths[i])]
            )
            self.stages.append(stage_blocks)
            if i < 3:
                self.downsamples.append(
                    nn.Sequential(
                        _LayerNorm2d(dims[i]),
                        nn.Conv2d(dims[i], dims[i + 1], kernel_size=2, stride=2),
                    )
                )
        self.norm = _LayerNorm2d(dims[-1])
        self.head = nn.Linear(dims[-1], feat_dim)
        self.out_dim = feat_dim

    def forward_tokens(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        for i, stage in enumerate(self.stages):
            x = stage(x)
            if i < len(self.downsamples):
                x = self.downsamples[i](x)
        x = self.norm(x)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.forward_tokens(x)
        x = x.mean(dim=(-2, -1))
        return self.head(x)


def build_backbone(name: str, in_channels: int = 1, feat_dim: int = 256, **kwargs) -> nn.Module:
    name = name.lower()
    if name == "resnet1d":
        return ResNet1D(in_channels=in_channels, feat_dim=feat_dim, **kwargs)
    if name == "wdcnn":
        return WDCNN(in_channels=in_channels, feat_dim=feat_dim)
    if name == "tfconvnext":
        return TFConvNeXtTiny(in_channels=in_channels, feat_dim=feat_dim, **kwargs)
    raise ValueError(f"Unknown backbone: {name}")
