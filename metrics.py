from __future__ import annotations

import torch


class SegmentationMetric:
    def __init__(self, num_classes: int, ignore_index: int = 255) -> None:
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.confusion_matrix = torch.zeros((num_classes, num_classes), dtype=torch.float64)

    @torch.no_grad()
    def update(self, logits: torch.Tensor, target: torch.Tensor) -> None:
        pred = logits.argmax(dim=1)
        valid = target != self.ignore_index
        pred = pred[valid]
        target = target[valid]
        if target.numel() == 0:
            return

        indices = target * self.num_classes + pred
        hist = torch.bincount(indices, minlength=self.num_classes ** 2)
        self.confusion_matrix += hist.reshape(self.num_classes, self.num_classes).cpu()

    def compute(self) -> dict[str, float]:
        hist = self.confusion_matrix
        eps = 1e-6

        tp = hist.diag()
        fp = hist.sum(dim=0) - tp
        fn = hist.sum(dim=1) - tp

        iou = tp / (tp + fp + fn + eps)
        precision = tp / (tp + fp + eps)
        recall = tp / (tp + fn + eps)
        f1 = 2 * precision * recall / (precision + recall + eps)

        pixel_acc = tp.sum() / (hist.sum() + eps)
        mean_acc = (tp / (hist.sum(dim=1) + eps)).mean()

        return {
            "pixel_acc": float(pixel_acc.item()),
            "mean_acc": float(mean_acc.item()),
            "miou": float(iou.mean().item()),
            "mf1": float(f1.mean().item()),
            "precision": float(precision.mean().item()),
            "recall": float(recall.mean().item()),
        }

