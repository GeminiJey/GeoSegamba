from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _num_groups(channels: int, max_groups: int = 8) -> int:
    for groups in range(min(max_groups, channels), 0, -1):
        if channels % groups == 0:
            return groups
    return 1


class ConvGNAct(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int | None = None,
        groups: int = 1,
        activation: bool = True,
    ) -> None:
        super().__init__()
        if padding is None:
            padding = kernel_size // 2

        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            bias=False,
        )
        self.norm = nn.GroupNorm(_num_groups(out_channels), out_channels)
        self.act = nn.GELU() if activation else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.norm(self.conv(x)))


class BottConv(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        mid_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        bias: bool = False,
    ) -> None:
        super().__init__()
        mid_channels = max(4, mid_channels)
        self.pointwise_1 = nn.Conv2d(in_channels, mid_channels, 1, bias=bias)
        self.depthwise = nn.Conv2d(
            mid_channels,
            mid_channels,
            kernel_size,
            stride,
            padding,
            groups=mid_channels,
            bias=False,
        )
        self.pointwise_2 = nn.Conv2d(mid_channels, out_channels, 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pointwise_1(x)
        x = self.depthwise(x)
        x = self.pointwise_2(x)
        return x


class GBC(nn.Module):
    def __init__(self, in_channels: int) -> None:
        super().__init__()
        mid_channels = max(4, in_channels // 8)

        self.block1 = nn.Sequential(
            BottConv(in_channels, in_channels, mid_channels, 3, 1, 1),
            nn.GroupNorm(_num_groups(in_channels), in_channels),
            nn.GELU(),
        )
        self.block2 = nn.Sequential(
            BottConv(in_channels, in_channels, mid_channels, 3, 1, 1),
            nn.GroupNorm(_num_groups(in_channels), in_channels),
            nn.GELU(),
        )
        self.block3 = nn.Sequential(
            BottConv(in_channels, in_channels, mid_channels, 1, 1, 0),
            nn.GroupNorm(_num_groups(in_channels), in_channels),
            nn.GELU(),
        )
        self.block4 = nn.Sequential(
            BottConv(in_channels, in_channels, mid_channels, 1, 1, 0),
            nn.GroupNorm(_num_groups(in_channels), in_channels),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x1 = self.block2(self.block1(x))
        x2 = self.block3(x)
        x = self.block4(x1 * x2)
        return x + residual


class ECAAttention(nn.Module):
    def __init__(self, channels: int, kernel_size: int = 3) -> None:
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(
            1,
            1,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            bias=False,
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.pool(x).squeeze(-1).transpose(1, 2)
        y = self.conv(y)
        y = self.sigmoid(y).transpose(1, 2).unsqueeze(-1)
        return x * y


class GeoPriorGate(nn.Module):
    def __init__(self, channels: int, geo_prior_channels: int) -> None:
        super().__init__()
        self.geo_prior_channels = geo_prior_channels
        self.gate = nn.Conv2d(channels + geo_prior_channels, channels, kernel_size=1)

    def forward(
        self,
        x: torch.Tensor,
        geo_prior: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if geo_prior is None or self.geo_prior_channels == 0:
            return x

        geo_prior = F.interpolate(
            geo_prior,
            size=x.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        weight = torch.sigmoid(self.gate(torch.cat([x, geo_prior], dim=1)))
        return x * (1.0 + weight)


class LightPAF(nn.Module):
    def __init__(self, channels: int, mid_channels: int | None = None) -> None:
        super().__init__()
        mid_channels = mid_channels or max(8, channels // 2)
        self.feature_transform = nn.Sequential(
            BottConv(channels, mid_channels, max(4, mid_channels // 2), 1),
            nn.GroupNorm(_num_groups(mid_channels), mid_channels),
        )
        self.channel_adapter = nn.Sequential(
            BottConv(mid_channels, channels, max(4, mid_channels // 2), 1),
            nn.GroupNorm(_num_groups(channels), channels),
        )

    def forward(
        self,
        base_feat: torch.Tensor,
        guidance_feat: torch.Tensor,
    ) -> torch.Tensor:
        if guidance_feat.shape[-2:] != base_feat.shape[-2:]:
            guidance_feat = F.interpolate(
                guidance_feat,
                size=base_feat.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

        guidance_query = self.feature_transform(guidance_feat)
        base_key = self.feature_transform(base_feat)
        similarity_map = torch.sigmoid(self.channel_adapter(base_key * guidance_query))
        return (1 - similarity_map) * base_feat + similarity_map * guidance_feat


class GeoSemanticPositionEncoding(nn.Module):
    def __init__(self, channels: int, geo_prior_channels: int) -> None:
        super().__init__()
        in_channels = 8 + geo_prior_channels
        self.proj = nn.Sequential(
            nn.Conv2d(in_channels, channels, kernel_size=1, bias=False),
            nn.GroupNorm(_num_groups(channels), channels),
            nn.GELU(),
        )
        self.geo_prior_channels = geo_prior_channels

    def forward(
        self,
        x: torch.Tensor,
        geo_prior: torch.Tensor | None = None,
    ) -> torch.Tensor:
        b, _, h, w = x.shape
        device = x.device
        dtype = x.dtype

        yy, xx = torch.meshgrid(
            torch.linspace(-1.0, 1.0, h, device=device, dtype=dtype),
            torch.linspace(-1.0, 1.0, w, device=device, dtype=dtype),
            indexing="ij",
        )
        base = [
            xx,
            yy,
            torch.sin(torch.pi * xx),
            torch.cos(torch.pi * xx),
            torch.sin(torch.pi * yy),
            torch.cos(torch.pi * yy),
            torch.sin(torch.pi * (xx + yy)),
            torch.cos(torch.pi * (xx - yy)),
        ]
        pos = torch.stack(base, dim=0).unsqueeze(0).expand(b, -1, -1, -1)

        if self.geo_prior_channels > 0:
            if geo_prior is None:
                geo_feat = torch.zeros(
                    b,
                    self.geo_prior_channels,
                    h,
                    w,
                    device=device,
                    dtype=dtype,
                )
            else:
                geo_feat = F.interpolate(
                    geo_prior,
                    size=(h, w),
                    mode="bilinear",
                    align_corners=False,
                )
            pos = torch.cat([pos, geo_feat], dim=1)

        return x + self.proj(pos)

