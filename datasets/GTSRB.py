import os
from typing import Callable, Optional
from PIL import Image
import torch
from torch.utils.data import Dataset


class GTSRB(Dataset):
    """
    Minimal GTSRB dataset wrapper when torchvision.datasets.GTSRB is unavailable.
    Expects directory structure:
      root/train/<class_id>/*.png
      root/test/<class_id>/*.png
    """

    def __init__(self, root: str, train: bool = True, transform: Optional[Callable] = None):
        self.root = root
        self.split = "train" if train else "test"
        self.transform = transform
        self.samples = []
        split_dir = os.path.join(root, self.split)
        if not os.path.isdir(split_dir):
            raise RuntimeError(f"GTSRB split not found: {split_dir}")
        for cls_name in sorted(os.listdir(split_dir)):
            cls_dir = os.path.join(split_dir, cls_name)
            if not os.path.isdir(cls_dir):
                continue
            try:
                cls_id = int(cls_name)
            except Exception:
                continue
            for fname in sorted(os.listdir(cls_dir)):
                if not fname.lower().endswith((".png", ".jpg", ".jpeg")):
                    continue
                self.samples.append((os.path.join(cls_dir, fname), cls_id))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, label = self.samples[index]
        img = Image.open(path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, label
