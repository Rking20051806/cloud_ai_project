from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF


SUPPORTED_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


@dataclass(frozen=True)
class RiceSample:
    image_path: Path
    target_path: Path
    mask_path: Optional[Path]


class RiceDataset(Dataset):
    """RICE dataset loader for cloud removal and cloud detection experiments."""

    def __init__(self, root_dir: str | Path, split: str = "RICE2", image_size: int = 256):
        self.root_dir = Path(root_dir)
        self.split = split
        self.image_size = image_size

        split_dir = self.root_dir / split
        self.cloud_dir = split_dir / "cloud"
        self.label_dir = split_dir / "label"
        self.mask_dir = split_dir / "mask"

        if not self.cloud_dir.exists():
            raise FileNotFoundError(f"Cloud directory not found: {self.cloud_dir}")
        if not self.label_dir.exists():
            raise FileNotFoundError(f"Label directory not found: {self.label_dir}")

        self.samples = self._build_samples()

    def _build_samples(self) -> list[RiceSample]:
        samples: list[RiceSample] = []
        for image_path in sorted(self.cloud_dir.iterdir()):
            if image_path.suffix.lower() not in SUPPORTED_IMAGE_EXTENSIONS:
                continue

            target_path = self.label_dir / image_path.name
            if not target_path.exists():
                target_candidates = list(self.label_dir.glob(f"{image_path.stem}.*"))
                target_path = target_candidates[0] if target_candidates else image_path

            mask_path: Optional[Path] = None
            if self.mask_dir.exists():
                candidate = self.mask_dir / image_path.name
                if candidate.exists():
                    mask_path = candidate
                else:
                    mask_candidates = list(self.mask_dir.glob(f"{image_path.stem}.*"))
                    mask_path = mask_candidates[0] if mask_candidates else None

            samples.append(RiceSample(image_path=image_path, target_path=target_path, mask_path=mask_path))
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def _load_rgb(self, path: Path) -> Image.Image:
        return Image.open(path).convert("RGB")

    def _load_mask(self, path: Optional[Path]) -> Optional[Image.Image]:
        if path is None:
            return None
        return Image.open(path).convert("L")

    def __getitem__(self, index: int):
        sample = self.samples[index]

        image = self._load_rgb(sample.image_path)
        target = self._load_rgb(sample.target_path)
        mask = self._load_mask(sample.mask_path)

        image = TF.resize(image, [self.image_size, self.image_size], interpolation=InterpolationMode.BILINEAR)
        target = TF.resize(target, [self.image_size, self.image_size], interpolation=InterpolationMode.BILINEAR)
        if mask is not None:
            mask = TF.resize(mask, [self.image_size, self.image_size], interpolation=InterpolationMode.NEAREST)

        image_tensor = TF.to_tensor(image)
        target_tensor = TF.to_tensor(target)
        mask_tensor = TF.to_tensor(mask) if mask is not None else torch.zeros((1, self.image_size, self.image_size))

        return {
            "image": image_tensor,
            "target": target_tensor,
            "mask": mask_tensor,
            "has_mask": mask is not None,
            "image_path": str(sample.image_path),
        }
