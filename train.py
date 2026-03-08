from __future__ import annotations

import argparse
from pathlib import Path

import torch

from datasets import build_dataloaders
from datasets.segmentation_dataset import DatasetConfig
from engine import evaluate, train_one_epoch
from losses import build_loss
from models import build_geosegamba
from utils import ensure_dir, load_checkpoint, now_str, save_checkpoint, save_json, seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Train GeoSegamba for geographic segmentation")

    parser.add_argument("--data-root", type=str, required=True)
    parser.add_argument("--train-split", type=str, default="train")
    parser.add_argument("--val-split", type=str, default="val")
    parser.add_argument("--image-dirname", type=str, default="images")
    parser.add_argument("--mask-dirname", type=str, default="masks")
    parser.add_argument("--geo-prior-dirname", type=str, default="geo_prior")
    parser.add_argument("--use-geo-prior", action="store_true")
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-val-samples", type=int, default=None)

    parser.add_argument("--num-classes", type=int, default=6)
    parser.add_argument("--in-channels", type=int, default=3)
    parser.add_argument("--geo-prior-channels", type=int, default=1)
    parser.add_argument("--dims", type=int, nargs=4, default=(32, 64, 128, 192))
    parser.add_argument("--depths", type=int, nargs=4, default=(1, 1, 2, 2))
    parser.add_argument("--decoder-channels", type=int, default=64)

    parser.add_argument("--image-size", type=int, nargs=2, default=(512, 512))
    parser.add_argument("--mean", type=float, nargs="+", default=(0.485, 0.456, 0.406))
    parser.add_argument("--std", type=float, nargs="+", default=(0.229, 0.224, 0.225))
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--val-batch-size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--grad-clip", type=float, default=None)

    parser.add_argument("--optimizer", type=str, choices=["adamw", "sgd"], default="adamw")
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--scheduler", type=str, choices=["poly", "cosine", "step", "none"], default="poly")
    parser.add_argument("--min-lr", type=float, default=1e-6)
    parser.add_argument("--step-size", type=int, default=30)
    parser.add_argument("--gamma", type=float, default=0.1)

    parser.add_argument("--ce-weight", type=float, default=1.0)
    parser.add_argument("--dice-weight", type=float, default=0.5)
    parser.add_argument("--focal-weight", type=float, default=0.0)
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    parser.add_argument("--focal-alpha", type=float, default=None)
    parser.add_argument("--ignore-index", type=int, default=255)

    parser.add_argument("--output-dir", type=str, default="./outputs")
    parser.add_argument("--experiment-name", type=str, default=None)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--save-every", type=int, default=10)

    return parser.parse_args()


def build_optimizer(args: argparse.Namespace, model: torch.nn.Module) -> torch.optim.Optimizer:
    if args.optimizer == "sgd":
        return torch.optim.SGD(
            model.parameters(),
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
        )
    return torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)


def build_scheduler(
    args: argparse.Namespace,
    optimizer: torch.optim.Optimizer,
) -> torch.optim.lr_scheduler._LRScheduler | None:
    if args.scheduler == "step":
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
    if args.scheduler == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.min_lr)
    if args.scheduler == "poly":
        lambda_fn = lambda epoch: max((1 - epoch / max(1, args.epochs)) ** 0.9, args.min_lr / args.lr)
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_fn)
    return None


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)

    device = torch.device(args.device)
    exp_name = args.experiment_name or f"geosegamba_{now_str()}"
    output_dir = ensure_dir(Path(args.output_dir) / exp_name)

    dataset_config = DatasetConfig(
        root=args.data_root,
        train_split=args.train_split,
        val_split=args.val_split,
        image_dirname=args.image_dirname,
        mask_dirname=args.mask_dirname,
        geo_prior_dirname=args.geo_prior_dirname,
        image_size=tuple(args.image_size),
        mean=tuple(args.mean),
        std=tuple(args.std),
        use_geo_prior=args.use_geo_prior,
        ignore_index=args.ignore_index,
        num_classes=args.num_classes,
        max_train_samples=args.max_train_samples,
        max_val_samples=args.max_val_samples,
    )

    save_json(output_dir / "config.json", {"args": vars(args), "dataset": dataset_config})

    train_loader, val_loader = build_dataloaders(
        config=dataset_config,
        batch_size=args.batch_size,
        val_batch_size=args.val_batch_size,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    model = build_geosegamba(
        in_channels=args.in_channels,
        num_classes=args.num_classes,
        geo_prior_channels=args.geo_prior_channels if args.use_geo_prior else 0,
        dims=tuple(args.dims),
        depths=tuple(args.depths),
        decoder_channels=args.decoder_channels,
    ).to(device)

    criterion = build_loss(
        ce_weight=args.ce_weight,
        dice_weight=args.dice_weight,
        focal_weight=args.focal_weight,
        label_smoothing=args.label_smoothing,
        focal_gamma=args.focal_gamma,
        focal_alpha=args.focal_alpha,
        ignore_index=args.ignore_index,
    )
    optimizer = build_optimizer(args, model)
    scheduler = build_scheduler(args, optimizer)
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp and device.type == "cuda")

    start_epoch = 0
    best_miou = 0.0
    if args.resume:
        start_epoch, best_miou = load_checkpoint(
            args.resume,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            map_location=device,
        )
        start_epoch += 1

    log_path = output_dir / "train_log.txt"
    with log_path.open("a", encoding="utf-8") as log_file:
        for epoch in range(start_epoch, args.epochs):
            train_stats = train_one_epoch(
                model=model,
                criterion=criterion,
                data_loader=train_loader,
                optimizer=optimizer,
                device=device,
                epoch=epoch,
                scaler=scaler,
                use_amp=args.amp,
                grad_clip=args.grad_clip,
            )
            val_stats = evaluate(
                model=model,
                criterion=criterion,
                data_loader=val_loader,
                device=device,
                num_classes=args.num_classes,
                ignore_index=args.ignore_index,
            )

            if scheduler is not None:
                scheduler.step()

            current_lr = optimizer.param_groups[0]["lr"]
            message = (
                f"epoch={epoch} "
                f"train_loss={train_stats['loss']:.4f} "
                f"val_loss={val_stats['loss']:.4f} "
                f"miou={val_stats['miou']:.4f} "
                f"mf1={val_stats['mf1']:.4f} "
                f"pixel_acc={val_stats['pixel_acc']:.4f} "
                f"lr={current_lr:.8f}"
            )
            print(message)
            log_file.write(message + "\n")
            log_file.flush()

            save_checkpoint(
                output_dir / "latest.pth",
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                best_score=best_miou,
                args=vars(args),
            )

            if val_stats["miou"] >= best_miou:
                best_miou = val_stats["miou"]
                save_checkpoint(
                    output_dir / "best.pth",
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    epoch=epoch,
                    best_score=best_miou,
                    args=vars(args),
                )

            if args.save_every > 0 and (epoch + 1) % args.save_every == 0:
                save_checkpoint(
                    output_dir / f"epoch_{epoch:03d}.pth",
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    epoch=epoch,
                    best_score=best_miou,
                    args=vars(args),
                )


if __name__ == "__main__":
    main()
