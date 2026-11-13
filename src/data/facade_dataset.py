"""
ArchiGen 3D - Real CMP Facades dataset (pix2pix "facades" archive)

Each JPEG in data/raw/facades/{train,val,test}/ is 512x256 RGB:
    left half  = photograph      (TARGET)
    right half = semantic map    (INPUT / condition)

Preprocessing follows the project scoping document (Cadrage, p.8):
  * split each pair in two, RGB, 256x256 (no resize needed for this archive),
  * normalise to [-1, 1],
  * a horizontal flip applied identically to map + photo, TRAIN split only.
Validation and test have no random augmentation.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from src.config import RAW_DATA_DIR

FACADES_DIR = RAW_DATA_DIR / "facades"
SPLITS = ("train", "val", "test")


def _sort_key(p: Path):
    return (0, int(p.stem)) if p.stem.isdigit() else (1, p.stem)


def list_split(split: str, root: Path = FACADES_DIR) -> List[Path]:
    split_dir = Path(root) / split
    if not split_dir.is_dir():
        raise FileNotFoundError(
            f"CMP Facades split not found: {split_dir}. Run `dvc pull data/raw/facades.dvc`."
        )
    files = sorted(split_dir.glob("*.jpg"), key=_sort_key)
    if not files:
        raise FileNotFoundError(f"No .jpg files in {split_dir}")
    return files


def load_pair(path: Path, image_size: int = 256) -> Tuple[np.ndarray, np.ndarray]:
    """Returns (semantic_map, photo) as uint8 HxWx3 arrays."""
    img = Image.open(path).convert("RGB")
    w, h = img.size
    half = w // 2
    photo = img.crop((0, 0, half, h))
    sem_map = img.crop((half, 0, w, h))
    if photo.size != (image_size, image_size):
        photo = photo.resize((image_size, image_size), Image.Resampling.BICUBIC)
        # Nearest keeps label colours intact when resizing a semantic map.
        sem_map = sem_map.resize((image_size, image_size), Image.Resampling.NEAREST)
    return np.asarray(sem_map, dtype=np.uint8), np.asarray(photo, dtype=np.uint8)


def to_tensor(arr: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(arr.transpose(2, 0, 1).astype(np.float32) / 127.5 - 1.0)


def to_uint8(t: torch.Tensor) -> np.ndarray:
    """[-1,1] CHW tensor -> uint8 HWC array."""
    arr = ((t.detach().cpu().clamp(-1, 1) + 1.0) * 127.5).round().byte().numpy()
    return arr.transpose(1, 2, 0)


class CMPFacadesDataset(Dataset):
    def __init__(
        self,
        split: str,
        root: Path = FACADES_DIR,
        image_size: int = 256,
        augment: Optional[bool] = None,
        limit: Optional[int] = None,
    ):
        assert split in SPLITS, split
        self.split = split
        self.image_size = image_size
        self.augment = (split == "train") if augment is None else augment
        self.files = list_split(split, root)
        if limit:
            self.files = self.files[:limit]

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int):
        sem_map, photo = load_pair(self.files[idx], self.image_size)
        m, p = to_tensor(sem_map), to_tensor(photo)
        if self.augment and torch.rand(1).item() < 0.5:
            m, p = m.flip(-1), p.flip(-1)
        return m, p


def make_facade_loaders(
    batch_size: int = 4,
    root: Path = FACADES_DIR,
    num_workers: int = 0,
    limit: Optional[int] = None,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader]:
    """Train (shuffled, flipped) and validation (fixed order) loaders. Test is never loaded here."""
    g = torch.Generator().manual_seed(seed)
    train = CMPFacadesDataset("train", root, limit=limit)
    val = CMPFacadesDataset("val", root, limit=limit)
    pin = torch.cuda.is_available()
    return (
        DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=num_workers, generator=g,
                   pin_memory=pin, persistent_workers=num_workers > 0),
        DataLoader(val, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                   pin_memory=pin, persistent_workers=num_workers > 0),
    )
