from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import random

import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset


IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}


def _is_image_file(path: Path) -> bool:
    return path.suffix.lower() in IMG_EXTENSIONS


def _collect_files(directory: Path, required_dirname: str) -> list[Path]:
    if not directory.exists():
        raise FileNotFoundError(f"Directory does not exist: {directory}")
    files = [
        path
        for path in directory.rglob("*")
        if path.is_file() and _is_image_file(path) and required_dirname in path.parts
    ]
    if not files:
        raise FileNotFoundError(f"No image files found in: {directory} with dirname '{required_dirname}'")
    return sorted(files)


def _to_tensor(image: np.ndarray) -> torch.Tensor:
    if image.ndim == 2:
        image = image[:, :, None]
    tensor = torch.from_numpy(np.ascontiguousarray(image.transpose(2, 0, 1))).float()
    if tensor.max() > 1:
        tensor = tensor / 255.0
    return tensor


def _normalize(image: torch.Tensor, mean: list[float], std: list[float]) -> torch.Tensor:
    mean_t = torch.tensor(mean, dtype=image.dtype, device=image.device).view(-1, 1, 1)
    std_t = torch.tensor(std, dtype=image.dtype, device=image.device).view(-1, 1, 1)
    if mean_t.shape[0] != image.shape[0]:
        if mean_t.shape[0] == 1:
            mean_t = mean_t.expand(image.shape[0], -1, -1)
            std_t = std_t.expand(image.shape[0], -1, -1)
        elif mean_t.shape[0] < image.shape[0]:
            extra_channels = image.shape[0] - mean_t.shape[0]
            mean_t = torch.cat([mean_t, mean_t[-1:].expand(extra_channels, -1, -1)], dim=0)
            std_t = torch.cat([std_t, std_t[-1:].expand(extra_channels, -1, -1)], dim=0)
        else:
            raise ValueError(
                f"Normalize channels mismatch: image has {image.shape[0]} channels, "
                f"mean/std define {mean_t.shape[0]} channels."
            )
    return (image - mean_t) / std_t


def _load_image(path: Path) -> np.ndarray:
    image = Image.open(path)
    array = np.array(image)
    if array.ndim == 2:
        return array[:, :, None]
    return array


def _load_mask(path: Path) -> np.ndarray:
    mask = Image.open(path)
    array = np.array(mask)
    if array.ndim == 3:
        array = array[..., 0]
    return array.astype(np.int64)


def _resize_image(image: np.ndarray, size: tuple[int, int], is_mask: bool) -> np.ndarray:
    pil = Image.fromarray(image.squeeze(-1) if image.ndim == 3 and image.shape[-1] == 1 else image)
    resample = Image.NEAREST if is_mask else Image.BILINEAR
    resized = pil.resize((size[1], size[0]), resample=resample)
    out = np.array(resized)
    if out.ndim == 2 and not is_mask:
        out = out[:, :, None]
    return out


def _random_scale(image: np.ndarray, mask: np.ndarray, geo_prior: np.ndarray | None, scale: float) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    if abs(scale - 1.0) < 1e-6:
        return image, mask, geo_prior

    h, w = image.shape[:2]
    new_size = (max(1, int(h * scale)), max(1, int(w * scale)))
    image = _resize_image(image, new_size, is_mask=False)
    mask = _resize_image(mask, new_size, is_mask=True)
    if geo_prior is not None:
        geo_prior = _resize_image(geo_prior, new_size, is_mask=False)
    return image, mask, geo_prior


def _pad_if_needed(
    image: np.ndarray,
    mask: np.ndarray,
    geo_prior: np.ndarray | None,
    size: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    target_h, target_w = size
    h, w = image.shape[:2]
    pad_h = max(0, target_h - h)
    pad_w = max(0, target_w - w)
    if pad_h == 0 and pad_w == 0:
        return image, mask, geo_prior

    image = np.pad(image, ((0, pad_h), (0, pad_w), (0, 0)), mode="reflect")
    mask = np.pad(mask, ((0, pad_h), (0, pad_w)), mode="constant", constant_values=0)
    if geo_prior is not None:
        channels = 1 if geo_prior.ndim == 2 else geo_prior.shape[2]
        geo_prior = np.pad(
            geo_prior,
            ((0, pad_h), (0, pad_w), (0, 0)) if channels > 1 or geo_prior.ndim == 3 else ((0, pad_h), (0, pad_w)),
            mode="reflect",
        )
    return image, mask, geo_prior


def _random_crop(
    image: np.ndarray,
    mask: np.ndarray,
    geo_prior: np.ndarray | None,
    crop_size: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    image, mask, geo_prior = _pad_if_needed(image, mask, geo_prior, crop_size)
    target_h, target_w = crop_size
    h, w = image.shape[:2]
    top = random.randint(0, h - target_h)
    left = random.randint(0, w - target_w)
    image = image[top : top + target_h, left : left + target_w]
    mask = mask[top : top + target_h, left : left + target_w]
    if geo_prior is not None:
        geo_prior = geo_prior[top : top + target_h, left : left + target_w]
    return image, mask, geo_prior


def _center_crop(
    image: np.ndarray,
    mask: np.ndarray,
    geo_prior: np.ndarray | None,
    crop_size: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    image, mask, geo_prior = _pad_if_needed(image, mask, geo_prior, crop_size)
    target_h, target_w = crop_size
    h, w = image.shape[:2]
    top = max(0, (h - target_h) // 2)
    left = max(0, (w - target_w) // 2)
    image = image[top : top + target_h, left : left + target_w]
    mask = mask[top : top + target_h, left : left + target_w]
    if geo_prior is not None:
        geo_prior = geo_prior[top : top + target_h, left : left + target_w]
    return image, mask, geo_prior


@dataclass
class DatasetConfig:
    root: str
    train_split: str = "train"
    val_split: str = "val"
    image_dirname: str = "images"
    mask_dirname: str = "masks"
    geo_prior_dirname: str = "geo_prior"
    image_size: tuple[int, int] = (512, 512)
    mean: tuple[float, ...] = (0.485, 0.456, 0.406)
    std: tuple[float, ...] = (0.229, 0.224, 0.225)
    use_geo_prior: bool = False
    ignore_index: int = 255
    num_classes: int = 6
    max_train_samples: int | None = None
    max_val_samples: int | None = None


class SegmentationDataset(Dataset):
    def __init__(
        self,
        root: str,
        split: str,
        image_dirname: str = "images",
        mask_dirname: str = "masks",
        geo_prior_dirname: str = "geo_prior",
        image_size: tuple[int, int] = (512, 512),
        mean: tuple[float, ...] = (0.485, 0.456, 0.406),
        std: tuple[float, ...] = (0.229, 0.224, 0.225),
        use_geo_prior: bool = False,
        training: bool = True,
        random_flip: bool = True,
        random_scale_range: tuple[float, float] = (0.75, 1.25),
        max_samples: int | None = None,
    ) -> None:
        super().__init__()
        self.root = Path(root)
        self.split = split
        self.training = training
        self.image_size = image_size
        self.mean = list(mean)
        self.std = list(std)
        self.use_geo_prior = use_geo_prior
        self.random_flip = random_flip
        self.random_scale_range = random_scale_range

        self.split_root = self.root / split
        image_dir = self.split_root
        self.geo_prior_dir = self.split_root
        self.image_paths = _collect_files(image_dir, image_dirname)
        self.image_dirname = image_dirname
        self.mask_dirname = mask_dirname
        self.geo_prior_dirname = geo_prior_dirname
        self.mask_paths = [self._resolve_related_path(path, image_dirname, mask_dirname) for path in self.image_paths]

        if max_samples is not None:
            max_samples = max(0, int(max_samples))
            self.image_paths = self.image_paths[:max_samples]
            self.mask_paths = self.mask_paths[:max_samples]

        for mask_path in self.mask_paths:
            if not mask_path.exists():
                raise FileNotFoundError(f"Missing mask file: {mask_path}")

        if self.use_geo_prior:
            geo_prior_root = self.split_root / self.geo_prior_dirname
            if not geo_prior_root.exists():
                raise FileNotFoundError(f"Missing geo prior directory: {geo_prior_root}")

    def _resolve_related_path(self, image_path: Path, src_dirname: str, dst_dirname: str) -> Path:
        parts = list(image_path.parts)
        if src_dirname not in parts:
            raise ValueError(
                f"Cannot map file {image_path} because '{src_dirname}' is not present in its path."
            )
        idx = len(parts) - 1 - parts[::-1].index(src_dirname)
        parts[idx] = dst_dirname
        return Path(*parts)

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        image_path = self.image_paths[index]
        mask_path = self.mask_paths[index]

        image = _load_image(image_path)
        mask = _load_mask(mask_path)
        geo_prior = None

        if self.use_geo_prior:
            geo_prior_path = self._resolve_related_path(
                image_path,
                self.image_dirname,
                self.geo_prior_dirname,
            )
            if not geo_prior_path.exists():
                raise FileNotFoundError(f"Missing geo prior file: {geo_prior_path}")
            geo_prior = _load_image(geo_prior_path)

        if self.training:
            scale = random.uniform(*self.random_scale_range)
            image, mask, geo_prior = _random_scale(image, mask, geo_prior, scale)
            image, mask, geo_prior = _random_crop(image, mask, geo_prior, self.image_size)
            if self.random_flip and random.random() < 0.5:
                image = np.flip(image, axis=1).copy()
                mask = np.flip(mask, axis=1).copy()
                if geo_prior is not None:
                    geo_prior = np.flip(geo_prior, axis=1).copy()
        else:
            image = _resize_image(image, self.image_size, is_mask=False)
            mask = _resize_image(mask, self.image_size, is_mask=True)
            if geo_prior is not None:
                geo_prior = _resize_image(geo_prior, self.image_size, is_mask=False)
            image, mask, geo_prior = _center_crop(image, mask, geo_prior, self.image_size)

        image_tensor = _normalize(_to_tensor(image), self.mean, self.std)
        mask_tensor = torch.from_numpy(mask.copy()).long()

        sample: dict[str, torch.Tensor | str] = {
            "image": image_tensor,
            "mask": mask_tensor,
            "image_path": str(image_path),
            "mask_path": str(mask_path),
        }

        if geo_prior is not None:
            sample["geo_prior"] = _to_tensor(geo_prior)

        return sample


def build_dataloaders(
    config: DatasetConfig,
    batch_size: int,
    val_batch_size: int,
    num_workers: int,
    pin_memory: bool = True,
) -> tuple[DataLoader, DataLoader]:
    train_dataset = SegmentationDataset(
        root=config.root,
        split=config.train_split,
        image_dirname=config.image_dirname,
        mask_dirname=config.mask_dirname,
        geo_prior_dirname=config.geo_prior_dirname,
        image_size=config.image_size,
        mean=config.mean,
        std=config.std,
        use_geo_prior=config.use_geo_prior,
        training=True,
        max_samples=config.max_train_samples,
    )
    val_dataset = SegmentationDataset(
        root=config.root,
        split=config.val_split,
        image_dirname=config.image_dirname,
        mask_dirname=config.mask_dirname,
        geo_prior_dirname=config.geo_prior_dirname,
        image_size=config.image_size,
        mean=config.mean,
        std=config.std,
        use_geo_prior=config.use_geo_prior,
        training=False,
        random_flip=False,
        max_samples=config.max_val_samples,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=val_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )
    return train_loader, val_loader
