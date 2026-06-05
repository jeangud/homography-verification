"""Utilities for custom datasets"""

import inspect
import multiprocessing
import os
import sys
from functools import partial
from pathlib import Path

import torch
import pandas as pd
from PIL import Image
from torchvision import datasets
from torchvision.transforms import v2
from tqdm import tqdm

DIR_DATASETS = Path(__file__).parent
DIR_DOWNLOADS = DIR_DATASETS / ".data"


def get_dataset_choices():
    """
    Returns a list of dataset class names defined in the `datasets` module.
    """
    datasets_module = sys.modules[__name__]  # Dynamically get the current module
    return [
        name
        for name, obj in inspect.getmembers(datasets_module, inspect.isclass)
        if isinstance(obj, type) and obj.__module__ == datasets_module.__name__
    ]


class MNIST(datasets.MNIST):
    """MNIST dataset of handwritten digits."""

    def __init__(self):
        super().__init__(
            root=DIR_DOWNLOADS,
            train=False,
            download=True,
            transform=v2.Compose(
                [
                    v2.ToImage(),
                    v2.ToDtype(torch.float32, scale=True),  # Scale to [0,1]
                ]
            ),
        )


class CIFAR10(datasets.CIFAR10):
    """CIFAR10 dataset of 10 classes of objects."""

    def __init__(self):
        super().__init__(
            root=DIR_DOWNLOADS,
            train=False,
            download=True,
            transform=v2.Compose(
                [
                    v2.ToImage(),
                    v2.ToDtype(torch.float32, scale=True),
                    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.225, 0.225, 0.225]),
                ]
            ),
        )


class GTSRB(datasets.GTSRB):
    """German Traffic Sign Recognition Benchmark (GTSRB) dataset."""

    def __init__(self, size=30):
        super().__init__(
            root=DIR_DOWNLOADS,
            split="test",
            download=True,
            transform=v2.Compose(
                [
                    v2.ToImage(),
                    v2.Resize((size, size)),
                    v2.ToDtype(torch.float32, scale=True),
                ]
            ),
        )


class MetaRoom(datasets.VisionDataset):
    """Indoor object recognition dataset."""

    PATH_INSTANCES_CSV = DIR_DATASETS / "metaroom.csv"

    def __init__(self, root=Path("/data/metaroom")):
        if not root.is_dir() and not (root / "metaroom_tz").is_dir():
            raise RuntimeError(f"""Data not found. Please download the MetaRoom dataset under \"{root}\".
The directory structure should follow:
{root}
    ├── metaroom_rx
    ├── metaroom_rx_vanilla
    ├── metaroom_ry
    ├── metaroom_ry_vanilla
    ├── metaroom_rz
    ├── metaroom_rz_vanilla
    ├── metaroom_tx
    ├── metaroom_tx_vanilla
    ├── metaroom_ty
    ├── metaroom_ty_vanilla
    ├── metaroom_tz
    └── metaroom_tz_vanilla""")
        super().__init__(root)

        self.transform = v2.Compose(
            [v2.ToImage(), v2.Resize((32, 56)), v2.ToDtype(torch.float32, scale=True)]
        )

        df = pd.read_csv(self.PATH_INSTANCES_CSV)

        self._data = []
        for _, row in df.iterrows():
            label_idx = row["model"].split("_")[3]
            self._data.append((row["image"], label_idx))

    def __len__(self):
        return len(self._data)

    def __getitem__(self, idx):
        row = self._data[idx]
        img = Image.open(row[0]).convert("RGB")
        img = self.transform(img)
        return img, row[1]


class LARD(datasets.VisionDataset):
    """A modified version of the Landing Approach Runway Detection (LARD) dataset.
    We simplify the task to image classification with two classes: runway, and no runway."""

    PATH_CSV = DIR_DATASETS / "lard.csv"

    def __init__(self, root=Path("/data/lard"), size=32, split="test", transform=None):
        super().__init__(root)

        if transform is not None:
            self.transform = transform
        else:
            self.transform = v2.Compose(
                [
                    v2.ToImage(),
                    v2.Resize((size, size)),
                    v2.ToDtype(torch.float32, scale=True),
                ]
            )

        df = pd.read_csv(self.PATH_CSV, sep=";")

        if not root.is_dir():
            raise FileNotFoundError(
                f"LARD dataset not found at {root}.\n\n"
                f"To use the LARD dataset, please:\n"
                f"1. Download the dataset from: https://github.com/deel-ai/LARD/tree/LARD_V1\n"
                f"2. Extract it to {root} (or use a custom path: LARD(root='path/to/your/lard/data'))\n"
            )

        # Check if resized images exist
        dir_resized = root / f"resized_{size}"
        if not dir_resized.exists():
            print("Dataset not found, generating resized images...")
            self._resize_images(df, size)

        if split == "train":
            df = df[df["split"] == "train"]
        elif split == "val":
            df = df[df["split"] == "val"]
        elif split == "test":
            df = df[df["split"] == "test"]
        else:
            raise ValueError(f"Unknown split: {split}")

        self._paths_images = (dir_resized / df["image"]).values
        self._labels = df["label"].values

    def __len__(self):
        return len(self._labels)

    def __getitem__(self, idx):
        img = Image.open(self._paths_images[idx])
        img = self.transform(img)
        return img, self._labels[idx]

    def _resize_images(self, df, size):
        root_origin = self.root
        root_resized = self.root / f"resized_{size}"
        root_resized.mkdir(parents=True, exist_ok=True)

        processing_func = partial(
            LARD.resize_image,
            root_origin=root_origin,
            root_resized=root_resized,
            size=size,
        )
        processing_queue = tqdm(df["image"].values, total=len(df))

        with multiprocessing.Pool(os.cpu_count()) as pool:
            pool.map(processing_func, processing_queue)

    @staticmethod
    def resize_image(relative_path, root_origin, root_resized, size):
        original_path = root_origin / relative_path
        img = Image.open(original_path)
        img = img.resize((size, size))

        new_path = root_resized / relative_path
        new_path.parent.mkdir(parents=True, exist_ok=True)
        img.save(new_path)
