"""Tests for dataset utilities."""

import pytest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch
from PIL import Image

from vnn.datasets import get_dataset_choices


def test_get_dataset_choices():
    choices = get_dataset_choices()
    assert isinstance(choices, list)
    assert "MNIST" in choices
    assert "CIFAR10" in choices
    assert "GTSRB" in choices
    assert "MetaRoom" in choices
    assert "LARD" in choices


def test_get_dataset_choices_no_extra():
    """Should not include non-dataset classes."""
    choices = get_dataset_choices()
    for name in choices:
        assert name[0].isupper()  # All class names start with uppercase


def test_mnist_init():
    """Test MNIST initialization without downloading."""
    from vnn.datasets import MNIST

    with patch("torchvision.datasets.MNIST.__init__", return_value=None):
        m = MNIST()
        assert m is not None


def test_cifar10_init():
    """Test CIFAR10 initialization without downloading."""
    from vnn.datasets import CIFAR10

    with patch("torchvision.datasets.CIFAR10.__init__", return_value=None):
        c = CIFAR10()
        assert c is not None


def test_gtsrb_init():
    """Test GTSRB initialization without downloading."""
    from vnn.datasets import GTSRB

    with patch("torchvision.datasets.GTSRB.__init__", return_value=None):
        g = GTSRB()
        assert g is not None


def test_metaroom_missing_data():
    """Test MetaRoom raises when data directory is missing."""
    from vnn.datasets import MetaRoom

    with pytest.raises(RuntimeError, match="Data not found"):
        MetaRoom(root=Path("/nonexistent/path/metaroom"))


def test_metaroom_init_and_access(tmp_path):
    """Test MetaRoom init, __len__, __getitem__ with mocked data."""
    from vnn.datasets import MetaRoom

    # Create a directory that satisfies the check
    metaroom_dir = tmp_path / "metaroom_tz"
    metaroom_dir.mkdir()

    # Create a small test image
    img_path = tmp_path / "test_image.png"
    Image.fromarray(np.random.randint(0, 255, (64, 112, 3), dtype=np.uint8)).save(
        img_path
    )

    # Create mock CSV data
    mock_df = pd.DataFrame(
        {
            "image": [str(img_path)],
            "model": ["metaroom_rx_vanilla_0"],
        }
    )

    with (
        patch("torchvision.datasets.VisionDataset.__init__", return_value=None),
        patch.object(MetaRoom, "PATH_INSTANCES_CSV", tmp_path / "metaroom.csv"),
        patch("vnn.datasets.pd.read_csv", return_value=mock_df),
    ):
        ds = MetaRoom(root=tmp_path)
        assert len(ds) == 1
        img, label = ds[0]
        assert isinstance(img, torch.Tensor)
        assert label == "0"


def test_lard_init_with_splits(tmp_path):
    """Test LARD init with different splits."""
    from vnn.datasets import LARD

    # Create resized directory so _resize_images is not called
    resized_dir = tmp_path / "resized_32"
    resized_dir.mkdir()

    mock_df = pd.DataFrame(
        {
            "image": ["img1.png", "img2.png", "img3.png"],
            "split": ["train", "val", "test"],
            "label": [0, 1, 0],
        }
    )

    for split in ["train", "val", "test"]:
        with (
            patch("torchvision.datasets.VisionDataset.__init__", return_value=None),
            patch("vnn.datasets.pd.read_csv", return_value=mock_df),
        ):
            ds = LARD(root=tmp_path, split=split)
            assert len(ds) == 1  # each split has 1 entry


def test_lard_invalid_split(tmp_path):
    """Test LARD with invalid split raises ValueError."""
    from vnn.datasets import LARD

    resized_dir = tmp_path / "resized_32"
    resized_dir.mkdir()

    mock_df = pd.DataFrame(
        {
            "image": ["img1.png"],
            "split": ["test"],
            "label": [0],
        }
    )

    with (
        patch("torchvision.datasets.VisionDataset.__init__", return_value=None),
        patch("vnn.datasets.pd.read_csv", return_value=mock_df),
    ):
        with pytest.raises(ValueError, match="Unknown split"):
            LARD(root=tmp_path, split="invalid")


def test_lard_getitem(tmp_path):
    """Test LARD __getitem__."""
    from vnn.datasets import LARD

    resized_dir = tmp_path / "resized_32"
    resized_dir.mkdir()

    # Create a test image in resized dir
    img_path = resized_dir / "img1.png"
    Image.fromarray(np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)).save(
        img_path
    )

    mock_df = pd.DataFrame(
        {
            "image": ["img1.png"],
            "split": ["test"],
            "label": [1],
        }
    )

    with (
        patch("torchvision.datasets.VisionDataset.__init__", return_value=None),
        patch("vnn.datasets.pd.read_csv", return_value=mock_df),
    ):
        ds = LARD(root=tmp_path, split="test")
        img, label = ds[0]
        assert isinstance(img, torch.Tensor)
        assert label == 1


def test_lard_custom_transform(tmp_path):
    """Test LARD with custom transform."""
    from vnn.datasets import LARD
    from torchvision.transforms import v2

    resized_dir = tmp_path / "resized_32"
    resized_dir.mkdir()

    mock_df = pd.DataFrame(
        {
            "image": ["img1.png"],
            "split": ["test"],
            "label": [0],
        }
    )

    custom_transform = v2.Compose([v2.ToImage()])

    with (
        patch("torchvision.datasets.VisionDataset.__init__", return_value=None),
        patch("vnn.datasets.pd.read_csv", return_value=mock_df),
    ):
        ds = LARD(root=tmp_path, split="test", transform=custom_transform)
        assert ds.transform is custom_transform


def test_lard_resize_images(tmp_path):
    """Test LARD._resize_images triggers when resized dir missing."""
    from vnn.datasets import LARD

    # Don't create resized dir - should trigger _resize_images
    mock_df = pd.DataFrame(
        {
            "image": ["img1.png"],
            "split": ["test"],
            "label": [0],
        }
    )

    with (
        patch("torchvision.datasets.VisionDataset.__init__", return_value=None),
        patch("vnn.datasets.pd.read_csv", return_value=mock_df),
        patch.object(LARD, "_resize_images") as mock_resize,
    ):
        LARD(root=tmp_path, split="test")
        mock_resize.assert_called_once()


def test_lard_resize_image_static(tmp_path):
    """Test LARD.resize_image static method."""
    from vnn.datasets import LARD

    origin = tmp_path / "origin"
    origin.mkdir()
    resized = tmp_path / "resized"
    resized.mkdir()

    # Create a source image
    img = Image.fromarray(np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8))
    img.save(origin / "test.png")

    LARD.resize_image("test.png", root_origin=origin, root_resized=resized, size=32)

    result = Image.open(resized / "test.png")
    assert result.size == (32, 32)


def test_lard_resize_images_integration(tmp_path):
    """Test LARD._resize_images triggers multiprocessing pool."""
    from vnn.datasets import LARD

    # Create the origin image
    origin = tmp_path
    img = Image.fromarray(np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8))
    img.save(origin / "img1.png")

    mock_df = pd.DataFrame(
        {
            "image": ["img1.png"],
            "split": ["test"],
            "label": [0],
        }
    )

    def mock_vd_init(self, root, *args, **kwargs):
        self.root = root

    with (
        patch("torchvision.datasets.VisionDataset.__init__", mock_vd_init),
        patch("vnn.datasets.pd.read_csv", return_value=mock_df),
    ):
        LARD(root=tmp_path, split="test")
        # _resize_images was called, check resized dir exists
        assert (tmp_path / "resized_32").exists()
