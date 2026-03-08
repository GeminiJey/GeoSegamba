from __future__ import annotations

from contextlib import nullcontext
from typing import Any

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from metrics import SegmentationMetric


def _move_batch(batch: dict[str, Any], device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    image = batch["image"].to(device, non_blocking=True)
    mask = batch["mask"].to(device, non_blocking=True)
    geo_prior = batch.get("geo_prior")
    if isinstance(geo_prior, torch.Tensor):
        geo_prior = geo_prior.to(device, non_blocking=True)
    else:
        geo_prior = None
    return image, mask, geo_prior


def train_one_epoch(
    model: torch.nn.Module,
    criterion,
    data_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    scaler: torch.cuda.amp.GradScaler | None = None,
    use_amp: bool = False,
    grad_clip: float | None = None,
) -> dict[str, float]:
    model.train()
    metric_logger = {"loss": 0.0}
    num_batches = max(1, len(data_loader))
    progress = tqdm(data_loader, desc=f"Train {epoch}", leave=False)
    for batch in progress:
        image, mask, geo_prior = _move_batch(batch, device)
        optimizer.zero_grad(set_to_none=True)

        autocast_context = (
            torch.autocast(device_type=device.type, dtype=torch.float16)
            if use_amp and device.type == "cuda"
            else nullcontext()
        )
        with autocast_context:
            logits = model(image, geo_prior=geo_prior)
            loss, loss_dict = criterion(logits, mask)

        if scaler is not None and use_amp and device.type == "cuda":
            scaler.scale(loss).backward()
            if grad_clip is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        metric_logger["loss"] += float(loss.item())
        progress.set_postfix(loss=f"{loss.item():.4f}", ce=f"{loss_dict.get('ce', 0.0):.4f}")

    metric_logger["loss"] /= num_batches
    return metric_logger


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    criterion,
    data_loader: DataLoader,
    device: torch.device,
    num_classes: int,
    ignore_index: int,
) -> dict[str, float]:
    model.eval()
    metric = SegmentationMetric(num_classes=num_classes, ignore_index=ignore_index)
    total_loss = 0.0
    num_batches = max(1, len(data_loader))

    progress = tqdm(data_loader, desc="Validate", leave=False)
    for batch in progress:
        image, mask, geo_prior = _move_batch(batch, device)
        logits = model(image, geo_prior=geo_prior)
        loss, _ = criterion(logits, mask)
        total_loss += float(loss.item())
        metric.update(logits, mask)
        progress.set_postfix(loss=f"{loss.item():.4f}")

    stats = metric.compute()
    stats["loss"] = total_loss / num_batches
    return stats
