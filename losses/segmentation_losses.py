from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


def _one_hot(target: torch.Tensor, num_classes: int, ignore_index: int) -> torch.Tensor:
    valid_mask = target != ignore_index
    safe_target = torch.where(valid_mask, target, torch.zeros_like(target))
    one_hot = F.one_hot(safe_target, num_classes=num_classes).permute(0, 3, 1, 2).float()
    one_hot = one_hot * valid_mask.unsqueeze(1)
    return one_hot


class DiceLoss(nn.Module):
    def __init__(self, smooth: float = 1.0, ignore_index: int = 255) -> None:
        super().__init__()
        self.smooth = smooth
        self.ignore_index = ignore_index

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        num_classes = logits.shape[1]
        probs = torch.softmax(logits, dim=1)
        target_one_hot = _one_hot(target, num_classes, self.ignore_index)
        valid_mask = (target != self.ignore_index).unsqueeze(1)
        probs = probs * valid_mask

        intersection = (probs * target_one_hot).sum(dim=(0, 2, 3))
        denominator = probs.sum(dim=(0, 2, 3)) + target_one_hot.sum(dim=(0, 2, 3))
        dice = (2 * intersection + self.smooth) / (denominator + self.smooth)
        return 1.0 - dice.mean()


class SoftCrossEntropyLoss(nn.Module):
    def __init__(self, ignore_index: int = 255, label_smoothing: float = 0.0) -> None:
        super().__init__()
        self.ignore_index = ignore_index
        self.label_smoothing = label_smoothing

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.cross_entropy(
            logits,
            target,
            ignore_index=self.ignore_index,
            label_smoothing=self.label_smoothing,
        )


class FocalLoss(nn.Module):
    def __init__(
        self,
        gamma: float = 2.0,
        alpha: float | None = None,
        ignore_index: int = 255,
    ) -> None:
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.ignore_index = ignore_index

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(logits, target, reduction="none", ignore_index=self.ignore_index)
        valid_mask = target != self.ignore_index
        pt = torch.exp(-ce)
        focal = (1 - pt) ** self.gamma * ce
        if self.alpha is not None:
            focal = self.alpha * focal

        valid_focal = focal[valid_mask]
        if valid_focal.numel() == 0:
            return logits.new_tensor(0.0)
        return valid_focal.mean()


@dataclass
class LossConfig:
    ce_weight: float = 1.0
    dice_weight: float = 0.5
    focal_weight: float = 0.0
    label_smoothing: float = 0.0
    focal_gamma: float = 2.0
    focal_alpha: float | None = None
    ignore_index: int = 255


class CompositeSegmentationLoss(nn.Module):
    def __init__(self, config: LossConfig) -> None:
        super().__init__()
        self.config = config
        self.ce = SoftCrossEntropyLoss(
            ignore_index=config.ignore_index,
            label_smoothing=config.label_smoothing,
        )
        self.dice = DiceLoss(ignore_index=config.ignore_index)
        self.focal = FocalLoss(
            gamma=config.focal_gamma,
            alpha=config.focal_alpha,
            ignore_index=config.ignore_index,
        )

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
        losses: dict[str, torch.Tensor] = {}
        total = logits.new_tensor(0.0)

        if self.config.ce_weight > 0:
            ce_loss = self.ce(logits, target)
            losses["ce"] = ce_loss
            total = total + self.config.ce_weight * ce_loss

        if self.config.dice_weight > 0:
            dice_loss = self.dice(logits, target)
            losses["dice"] = dice_loss
            total = total + self.config.dice_weight * dice_loss

        if self.config.focal_weight > 0:
            focal_loss = self.focal(logits, target)
            losses["focal"] = focal_loss
            total = total + self.config.focal_weight * focal_loss

        stats = {name: float(value.detach().item()) for name, value in losses.items()}
        stats["total"] = float(total.detach().item())
        return total, stats


def build_loss(
    ce_weight: float = 1.0,
    dice_weight: float = 0.5,
    focal_weight: float = 0.0,
    label_smoothing: float = 0.0,
    focal_gamma: float = 2.0,
    focal_alpha: float | None = None,
    ignore_index: int = 255,
) -> CompositeSegmentationLoss:
    return CompositeSegmentationLoss(
        LossConfig(
            ce_weight=ce_weight,
            dice_weight=dice_weight,
            focal_weight=focal_weight,
            label_smoothing=label_smoothing,
            focal_gamma=focal_gamma,
            focal_alpha=focal_alpha,
            ignore_index=ignore_index,
        )
    )

