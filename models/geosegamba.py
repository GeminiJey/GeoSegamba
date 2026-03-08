from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import (
    BottConv,
    ConvGNAct,
    ECAAttention,
    GBC,
    GeoPriorGate,
    GeoSemanticPositionEncoding,
    LightPAF,
    _num_groups,
)


@lru_cache(maxsize=64)
def _scan_order(path_type: str, height: int, width: int) -> tuple[tuple[int, ...], tuple[int, ...]]:
    order: list[int] = []

    if path_type == "horizontal":
        for row in range(height):
            cols = range(width) if row % 2 == 0 else range(width - 1, -1, -1)
            for col in cols:
                order.append(row * width + col)
    elif path_type == "vertical":
        for col in range(width):
            rows = range(height) if col % 2 == 0 else range(height - 1, -1, -1)
            for row in rows:
                order.append(row * width + col)
    elif path_type == "diagonal":
        for diag in range(height + width - 1):
            cells = []
            row_start = max(0, diag - width + 1)
            row_end = min(height - 1, diag)
            for row in range(row_start, row_end + 1):
                col = diag - row
                if 0 <= col < width:
                    cells.append(row * width + col)
            if diag % 2 == 1:
                cells.reverse()
            order.extend(cells)
    else:
        raise ValueError(f"Unsupported scan path: {path_type}")

    inverse = [0] * len(order)
    for seq_idx, flat_idx in enumerate(order):
        inverse[flat_idx] = seq_idx

    return tuple(order), tuple(inverse)


class ScanPath(nn.Module):
    def __init__(self, channels: int, kernel_size: int = 5) -> None:
        super().__init__()
        self.seq_conv = nn.Conv1d(
            channels,
            channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=channels,
            bias=False,
        )
        self.mix = nn.Conv1d(channels, channels, kernel_size=1, bias=False)
        self.norm = nn.GroupNorm(_num_groups(channels), channels)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor, path_type: str) -> torch.Tensor:
        b, c, h, w = x.shape
        flat = x.flatten(2)
        order, inverse = _scan_order(path_type, h, w)
        order_t = torch.tensor(order, device=x.device, dtype=torch.long)
        inverse_t = torch.tensor(inverse, device=x.device, dtype=torch.long)

        seq = flat.index_select(-1, order_t)
        seq = self.mix(self.seq_conv(seq))
        restored = seq.index_select(-1, inverse_t).view(b, c, h, w)
        return self.act(self.norm(restored))


class GeoPathMixer(nn.Module):
    def __init__(self, channels: int, geo_prior_channels: int) -> None:
        super().__init__()
        self.geo_gate = GeoPriorGate(channels, geo_prior_channels)
        self.paths = nn.ModuleDict(
            {
                "horizontal": ScanPath(channels),
                "vertical": ScanPath(channels),
                "diagonal": ScanPath(channels),
            }
        )
        self.paf = LightPAF(channels)
        self.weight_head = nn.Sequential(
            nn.Conv2d(channels + geo_prior_channels, channels, kernel_size=1, bias=False),
            nn.GELU(),
            nn.Conv2d(channels, 3, kernel_size=1, bias=True),
        )
        self.out_proj = ConvGNAct(channels, channels, kernel_size=3)
        self.geo_prior_channels = geo_prior_channels

    def forward(
        self,
        x: torch.Tensor,
        geo_prior: torch.Tensor | None = None,
        return_details: bool = False,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]] | torch.Tensor:
        gated = self.geo_gate(x, geo_prior)
        path_outputs: list[torch.Tensor] = []
        path_heatmaps: list[torch.Tensor] = []

        for path_name, path_module in self.paths.items():
            scanned = path_module(gated, path_name)
            fused = self.paf(x, scanned)
            path_outputs.append(fused)
            path_heatmaps.append(fused.mean(dim=1, keepdim=True))

        pooled_x = F.adaptive_avg_pool2d(gated, 1)
        if self.geo_prior_channels > 0:
            if geo_prior is None:
                pooled_geo = torch.zeros(
                    pooled_x.shape[0],
                    self.geo_prior_channels,
                    1,
                    1,
                    device=x.device,
                    dtype=x.dtype,
                )
            else:
                pooled_geo = F.adaptive_avg_pool2d(geo_prior, 1)
            pooled = torch.cat([pooled_x, pooled_geo], dim=1)
        else:
            pooled = pooled_x

        weights = torch.softmax(self.weight_head(pooled).flatten(1), dim=1)
        fused = sum(
            path_outputs[idx] * weights[:, idx].view(-1, 1, 1, 1)
            for idx in range(len(path_outputs))
        )
        fused = self.out_proj(fused) + x

        if not return_details:
            return fused

        details = {
            "path_weights": weights,
            "horizontal_heatmap": path_heatmaps[0],
            "vertical_heatmap": path_heatmaps[1],
            "diagonal_heatmap": path_heatmaps[2],
        }
        return fused, details


class GeoVSSBlock(nn.Module):
    def __init__(self, channels: int, geo_prior_channels: int, expansion: int = 2) -> None:
        super().__init__()
        hidden_channels = channels * expansion
        self.gbc = GBC(channels)
        self.geo_mixer = GeoPathMixer(channels, geo_prior_channels)
        self.local_refine = ConvGNAct(channels, channels, kernel_size=3, groups=channels)
        self.ffn = nn.Sequential(
            ConvGNAct(channels, hidden_channels, kernel_size=1),
            nn.Conv2d(hidden_channels, channels, kernel_size=1, bias=False),
            nn.GroupNorm(_num_groups(channels), channels),
        )

    def forward(
        self,
        x: torch.Tensor,
        geo_prior: torch.Tensor | None = None,
        return_details: bool = False,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]] | torch.Tensor:
        residual = x
        x = self.gbc(x)
        if return_details:
            x, details = self.geo_mixer(x, geo_prior, return_details=True)
        else:
            x = self.geo_mixer(x, geo_prior, return_details=False)
            details = {}
        x = residual + self.local_refine(x)
        x = x + self.ffn(x)
        if return_details:
            return x, details
        return x


class EncoderStage(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        depth: int,
        geo_prior_channels: int,
        downsample: bool,
    ) -> None:
        super().__init__()
        if downsample:
            self.downsample = ConvGNAct(in_channels, out_channels, kernel_size=3, stride=2)
        elif in_channels != out_channels:
            self.downsample = ConvGNAct(in_channels, out_channels, kernel_size=1, padding=0)
        else:
            self.downsample = nn.Identity()

        self.blocks = nn.ModuleList(
            [GeoVSSBlock(out_channels, geo_prior_channels) for _ in range(depth)]
        )

    def forward(
        self,
        x: torch.Tensor,
        geo_prior: torch.Tensor | None = None,
        collect_details: bool = False,
    ) -> tuple[torch.Tensor, list[dict[str, torch.Tensor]]] | torch.Tensor:
        x = self.downsample(x)
        details_list: list[dict[str, torch.Tensor]] = []
        for block in self.blocks:
            if collect_details:
                x, details = block(x, geo_prior, return_details=True)
                details_list.append(details)
            else:
                x = block(x, geo_prior, return_details=False)
        if collect_details:
            return x, details_list
        return x


class LiteASPP(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dilations: tuple[int, ...] = (1, 3, 5, 7)) -> None:
        super().__init__()
        branches = []
        for dilation in dilations:
            if dilation == 1:
                branches.append(ConvGNAct(in_channels, out_channels, kernel_size=1, padding=0))
            else:
                branches.append(
                    nn.Sequential(
                        nn.Conv2d(
                            in_channels,
                            in_channels,
                            kernel_size=3,
                            padding=dilation,
                            dilation=dilation,
                            groups=in_channels,
                            bias=False,
                        ),
                        nn.GroupNorm(_num_groups(in_channels), in_channels),
                        nn.GELU(),
                        nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                        nn.GroupNorm(_num_groups(out_channels), out_channels),
                        nn.GELU(),
                    )
                )
        self.branches = nn.ModuleList(branches)
        self.global_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            ConvGNAct(in_channels, out_channels, kernel_size=1, padding=0),
        )
        self.fuse = ConvGNAct(out_channels * (len(dilations) + 1), out_channels, kernel_size=1, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outputs = [branch(x) for branch in self.branches]
        pooled = self.global_pool(x)
        pooled = F.interpolate(pooled, size=x.shape[-2:], mode="bilinear", align_corners=False)
        outputs.append(pooled)
        return self.fuse(torch.cat(outputs, dim=1))


class MFFH(nn.Module):
    def __init__(
        self,
        in_channels_list: list[int],
        decoder_channels: int,
        input_channels: int,
    ) -> None:
        super().__init__()
        self.proj_layers = nn.ModuleList(
            [ConvGNAct(in_channels, decoder_channels, kernel_size=1, padding=0) for in_channels in in_channels_list]
        )
        self.aspp_layers = nn.ModuleList(
            [LiteASPP(decoder_channels, decoder_channels) for _ in in_channels_list]
        )
        self.fuse = nn.Sequential(
            BottConv(
                decoder_channels * len(in_channels_list),
                decoder_channels,
                max(8, decoder_channels // 2),
                kernel_size=1,
            ),
            nn.GroupNorm(_num_groups(decoder_channels), decoder_channels),
            nn.GELU(),
        )
        self.eca = ECAAttention(decoder_channels)
        self.spectrum_proj = (
            ConvGNAct(input_channels, decoder_channels, kernel_size=1, padding=0)
            if input_channels > 3
            else None
        )

    def forward(
        self,
        features: list[torch.Tensor],
        raw_input: torch.Tensor | None = None,
    ) -> torch.Tensor:
        target_size = features[0].shape[-2:]
        fused_features = []

        for feature, proj, aspp in zip(features, self.proj_layers, self.aspp_layers):
            x = aspp(proj(feature))
            if x.shape[-2:] != target_size:
                x = F.interpolate(x, size=target_size, mode="bilinear", align_corners=False)
            fused_features.append(x)

        x = self.fuse(torch.cat(fused_features, dim=1))
        x = self.eca(x)

        if self.spectrum_proj is not None and raw_input is not None:
            spectrum = self.spectrum_proj(raw_input)
            spectrum = F.interpolate(spectrum, size=target_size, mode="bilinear", align_corners=False)
            x = x + spectrum

        return x


class GeoSS(nn.Module):
    def __init__(self, channels: int, geo_prior_channels: int) -> None:
        super().__init__()
        self.geo_gate = GeoPriorGate(channels, geo_prior_channels)
        self.paths = nn.ModuleDict(
            {
                "horizontal": ScanPath(channels, kernel_size=7),
                "vertical": ScanPath(channels, kernel_size=7),
                "diagonal": ScanPath(channels, kernel_size=7),
            }
        )
        self.weight_head = nn.Sequential(
            nn.Conv2d(channels + geo_prior_channels, channels, kernel_size=1, bias=False),
            nn.GELU(),
            nn.Conv2d(channels, 3, kernel_size=1, bias=True),
        )
        self.refine = nn.Sequential(
            ConvGNAct(channels, channels, kernel_size=3, groups=channels),
            ConvGNAct(channels, channels, kernel_size=1, padding=0),
        )
        self.geo_prior_channels = geo_prior_channels

    def forward(
        self,
        x: torch.Tensor,
        geo_prior: torch.Tensor | None = None,
        return_details: bool = False,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]] | torch.Tensor:
        base = self.geo_gate(x, geo_prior)
        path_outputs = [module(base, name) for name, module in self.paths.items()]

        pooled_x = F.adaptive_avg_pool2d(base, 1)
        if self.geo_prior_channels > 0:
            if geo_prior is None:
                pooled_geo = torch.zeros(
                    pooled_x.shape[0],
                    self.geo_prior_channels,
                    1,
                    1,
                    device=x.device,
                    dtype=x.dtype,
                )
            else:
                pooled_geo = F.adaptive_avg_pool2d(geo_prior, 1)
            pooled = torch.cat([pooled_x, pooled_geo], dim=1)
        else:
            pooled = pooled_x

        weights = torch.softmax(self.weight_head(pooled).flatten(1), dim=1)
        refined = sum(
            path_outputs[idx] * weights[:, idx].view(-1, 1, 1, 1)
            for idx in range(len(path_outputs))
        )
        refined = self.refine(refined) + x

        if not return_details:
            return refined

        return refined, {
            "path_weights": weights,
            "horizontal_heatmap": path_outputs[0].mean(dim=1, keepdim=True),
            "vertical_heatmap": path_outputs[1].mean(dim=1, keepdim=True),
            "diagonal_heatmap": path_outputs[2].mean(dim=1, keepdim=True),
        }


@dataclass
class GeoSegambaConfig:
    in_channels: int = 3
    num_classes: int = 6
    geo_prior_channels: int = 1
    dims: tuple[int, int, int, int] = (32, 64, 128, 192)
    depths: tuple[int, int, int, int] = (1, 1, 2, 2)
    decoder_channels: int = 64


class GeoSegamba(nn.Module):
    def __init__(self, config: GeoSegambaConfig) -> None:
        super().__init__()
        self.config = config
        dims = config.dims
        depths = config.depths

        self.stem = nn.Sequential(
            ConvGNAct(config.in_channels, dims[0], kernel_size=3, stride=2),
            ConvGNAct(dims[0], dims[0], kernel_size=3),
        )
        self.pos_encoding = GeoSemanticPositionEncoding(dims[0], config.geo_prior_channels)

        self.stage1 = EncoderStage(dims[0], dims[0], depths[0], config.geo_prior_channels, downsample=False)
        self.stage2 = EncoderStage(dims[0], dims[1], depths[1], config.geo_prior_channels, downsample=True)
        self.stage3 = EncoderStage(dims[1], dims[2], depths[2], config.geo_prior_channels, downsample=True)
        self.stage4 = EncoderStage(dims[2], dims[3], depths[3], config.geo_prior_channels, downsample=True)

        self.mffh = MFFH(list(dims), config.decoder_channels, config.in_channels)
        self.geoss = GeoSS(config.decoder_channels, config.geo_prior_channels)
        self.seg_head = nn.Conv2d(config.decoder_channels, config.num_classes, kernel_size=1)

    def forward(
        self,
        x: torch.Tensor,
        geo_prior: torch.Tensor | None = None,
        return_details: bool = False,
    ) -> dict[str, torch.Tensor] | torch.Tensor:
        raw_input = x
        input_size = raw_input.shape[-2:]
        x = self.stem(raw_input)
        x = self.pos_encoding(x, geo_prior)

        if return_details:
            f1, stage1_details = self.stage1(x, geo_prior, collect_details=True)
            f2, stage2_details = self.stage2(f1, geo_prior, collect_details=True)
            f3, stage3_details = self.stage3(f2, geo_prior, collect_details=True)
            f4, stage4_details = self.stage4(f3, geo_prior, collect_details=True)
        else:
            f1 = self.stage1(x, geo_prior, collect_details=False)
            f2 = self.stage2(f1, geo_prior, collect_details=False)
            f3 = self.stage3(f2, geo_prior, collect_details=False)
            f4 = self.stage4(f3, geo_prior, collect_details=False)

        fused = self.mffh(
            [f1, f2, f3, f4],
            raw_input=raw_input if self.config.in_channels > 3 else None,
        )
        if return_details:
            refined, geoss_details = self.geoss(fused, geo_prior, return_details=True)
        else:
            refined = self.geoss(fused, geo_prior, return_details=False)
            geoss_details = {}

        logits = self.seg_head(refined)
        logits = F.interpolate(logits, size=input_size, mode="bilinear", align_corners=False)

        if not return_details:
            return logits

        return {
            "logits": logits,
            "f1": f1,
            "f2": f2,
            "f3": f3,
            "f4": f4,
            "geoss": geoss_details,
            "stages": {
                "stage1": stage1_details,
                "stage2": stage2_details,
                "stage3": stage3_details,
                "stage4": stage4_details,
            },
        }


def build_geosegamba(
    in_channels: int = 3,
    num_classes: int = 6,
    geo_prior_channels: int = 1,
    dims: tuple[int, int, int, int] = (32, 64, 128, 192),
    depths: tuple[int, int, int, int] = (1, 1, 2, 2),
    decoder_channels: int = 64,
) -> GeoSegamba:
    config = GeoSegambaConfig(
        in_channels=in_channels,
        num_classes=num_classes,
        geo_prior_channels=geo_prior_channels,
        dims=dims,
        depths=depths,
        decoder_channels=decoder_channels,
    )
    return GeoSegamba(config)
