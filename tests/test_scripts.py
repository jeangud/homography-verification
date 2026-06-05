"""Tests for CLI scripts."""

import pickle
from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np
import pytest
import torch

from vnn.transforms.rotation import Rotation
from vnn.transforms.transform_with_bounds import TransformWithBounds
from vnn import datasets
from vnn.transforms import TransformType


def _make_mock_dataset(cls):
    """Create a mock dataset that passes isinstance checks for cls."""
    mock_img = np.random.rand(1, 4, 4).astype(np.float32)
    tensor_img = torch.tensor(mock_img)

    # Create a dynamic subclass that overrides __getitem__ and __len__
    # Special methods (__getitem__, __len__) must be on the type, not instance
    MockCls = type(
        f"Mock{cls.__name__}",
        (cls,),
        {
            "__init__": lambda self: None,  # Skip parent __init__
            "__getitem__": lambda self, idx: (tensor_img, 0),
            "__len__": lambda self: 1,
        },
    )
    return MockCls()


# ============================================================================
# scripts/calculate_bounds.py
# ============================================================================
class TestCalculateBounds:
    def test_get_experiment_name(self):
        from scripts.calculate_bounds import get_experiment_name

        ts = "2026-04-23-14-30-00"
        assert (
            get_experiment_name(ts, "MNIST", [TransformType.ROTATE], [0.0], [0.35], 0.5)
            == "2026-04-23-14-30-00_MNIST_ROTATE_0.00_0.35_pad0.5"
        )
        assert (
            get_experiment_name(ts, "LARD", [TransformType.H_ROLL], [0.0], [0.0], 0.0)
            == "2026-04-23-14-30-00_LARD_H_ROLL_0.00_0.00_pad0.0"
        )

    def test_parse_arguments_basic(self):
        from scripts.calculate_bounds import parse_arguments

        args = parse_arguments(
            [
                "--transformations",
                "ROTATE",
                "--lower",
                "0",
                "--upper",
                "0.1",
                "--dataset",
                "MNIST",
                "--image-number",
                "0",
            ]
        )
        assert args.dataset == "MNIST"
        assert args.image_number == 0
        assert args.padding == 0.5  # default, parsed as float

    def test_parse_arguments_custom_padding(self):
        from scripts.calculate_bounds import parse_arguments

        args = parse_arguments(
            [
                "--transformations",
                "ROTATE",
                "--lower",
                "0",
                "--upper",
                "0.1",
                "--dataset",
                "MNIST",
                "--image-number",
                "0",
                "--padding",
                "BORDER_REPLICATE",
            ]
        )
        assert args.padding == "BORDER_REPLICATE"

    def test_parse_arguments_invalid_padding(self):
        from scripts.calculate_bounds import parse_arguments

        with pytest.raises(ValueError, match="Invalid OpenCV padding"):
            parse_arguments(
                [
                    "--transformations",
                    "ROTATE",
                    "--lower",
                    "0",
                    "--upper",
                    "0.1",
                    "--dataset",
                    "MNIST",
                    "--image-number",
                    "0",
                    "--padding",
                    "NOT_A_PADDING",
                ]
            )

    def test_parse_arguments_csv_path(self):
        from scripts.calculate_bounds import parse_arguments

        args = parse_arguments(
            [
                "--transformations",
                "ROTATE",
                "--lower",
                "0",
                "--upper",
                "0.1",
                "--dataset",
                "MNIST",
                "--path-images-csv",
                "images.csv",
            ]
        )
        assert args.path_images_csv == Path("images.csv")
        assert args.image_number is None

    def test_parse_transform_types(self):
        from scripts.calculate_bounds import parse_transform
        from vnn.transforms import (
            Scale,
            ShearX,
            TranslateX,
            TranslateY,
            HomographyRoll,
            HomographyPitch,
            HomographyYaw,
            HomographyX,
            HomographyY,
            HomographyZ,
        )

        type_to_cls = {
            TransformType.ROTATE: Rotation,
            TransformType.SCALE: Scale,
            TransformType.SHEAR_X: ShearX,
            TransformType.TRANSLATE_X: TranslateX,
            TransformType.TRANSLATE_Y: TranslateY,
            TransformType.H_ROLL: HomographyRoll,
            TransformType.H_PITCH: HomographyPitch,
            TransformType.H_YAW: HomographyYaw,
            TransformType.H_X: HomographyX,
            TransformType.H_Y: HomographyY,
            TransformType.H_Z: HomographyZ,
        }

        for tf_type, expected_cls in type_to_cls.items():
            tfwb = parse_transform(
                [tf_type], [0.0], [0.1], _make_mock_dataset(datasets.MNIST)
            )
            assert isinstance(tfwb, TransformWithBounds)
            assert isinstance(tfwb.transform, expected_cls)

    def test_parse_transform_unsupported_type(self):
        """Test that an unknown transform type raises NotImplementedError."""
        from scripts.calculate_bounds import parse_transform
        from unittest.mock import MagicMock

        fake_type = MagicMock()
        fake_type.__eq__ = lambda self, other: False  # Won't match any branch

        with pytest.raises(NotImplementedError, match="not supported"):
            parse_transform(
                [fake_type], [0.0], [0.1], _make_mock_dataset(datasets.MNIST)
            )

    def test_parse_transform_dataset_types(self):
        """Test camera parameter selection for different dataset types."""
        from scripts.calculate_bounds import parse_transform

        for ds_cls in [
            datasets.MNIST,
            datasets.CIFAR10,
            datasets.GTSRB,
            datasets.MetaRoom,
            datasets.LARD,
        ]:
            tfwb = parse_transform(
                [TransformType.ROTATE], [0.0], [0.1], _make_mock_dataset(ds_cls)
            )
            assert isinstance(tfwb, TransformWithBounds)

    def test_parse_transform_unsupported_dataset(self):
        """Test that unsupported dataset type raises NotImplementedError."""
        from scripts.calculate_bounds import parse_transform

        # Create a mock with an unknown dataset class that doesn't match any isinstance checks
        class UnknownDataset:
            def __getitem__(self, idx):
                return torch.tensor(np.random.rand(1, 4, 4).astype(np.float32)), 0

            def __len__(self):
                return 1

        with pytest.raises(NotImplementedError, match="no custom camera"):
            parse_transform([TransformType.ROTATE], [0.0], [0.1], UnknownDataset())

    def test_main_with_image_number(self, tmp_path, monkeypatch):
        """Test the main function with --image-number."""
        from scripts.calculate_bounds import main
        from unittest.mock import patch as mock_patch
        import scipy.optimize as opt

        def mock_solve_gurobi(c, A, b):
            res = opt.linprog(c, A_ub=A, b_ub=b)
            if res.success:
                return res.x[0], res.x[1]
            raise RuntimeError("No solution found.")

        mock_img = torch.rand(1, 3, 3)
        mock_dataset_instance = MagicMock()
        mock_dataset_instance.__getitem__ = MagicMock(return_value=(mock_img, 0))

        # Use monkeypatch for cwd so timing CSVs are written to tmp_path
        monkeypatch.chdir(tmp_path)

        # Mock at a higher level: mock parse_transform and the dataset loading
        mock_tfwb = TransformWithBounds(Rotation(x0=1, y0=1), 0.0, 0.05)

        with (
            mock_patch("scripts.calculate_bounds.DIR_RESULTS", tmp_path),
            mock_patch(
                "scripts.calculate_bounds.parse_transform", return_value=mock_tfwb
            ),
            mock_patch("scripts.calculate_bounds.getattr", create=True) as mock_getattr,
            mock_patch("vnn.pwl.lp.solve_gurobi", side_effect=mock_solve_gurobi),
        ):
            # Patch getattr(datasets, args.dataset) to return mock dataset class
            mock_getattr.return_value = MagicMock(return_value=mock_dataset_instance)

            with mock_patch("scripts.calculate_bounds.datasets") as mock_ds_mod:
                mock_ds_mod.get_dataset_choices.return_value = ["MNIST"]
                mock_cls = MagicMock(return_value=mock_dataset_instance)
                setattr(mock_ds_mod, "MNIST", mock_cls)

                main(
                    [
                        "--transformations",
                        "ROTATE",
                        "--lower",
                        "0",
                        "--upper",
                        "0.05",
                        "--dataset",
                        "MNIST",
                        "--image-number",
                        "0",
                        "--num-samples",
                        "5",
                        "--num-init-splits",
                        "2",
                        "--num-splits",
                        "2",
                        "--num-subdomain-samples",
                        "3",
                        "--max-bab-iter",
                        "50",
                        "--lipschitz-error",
                        "0.1",
                    ]
                )

    def test_main_with_csv(self, tmp_path, monkeypatch):
        """Test the main function with --path-images-csv."""
        from scripts.calculate_bounds import main
        from unittest.mock import patch as mock_patch
        import scipy.optimize as opt

        def mock_solve_gurobi(c, A, b):
            res = opt.linprog(c, A_ub=A, b_ub=b)
            if res.success:
                return res.x[0], res.x[1]
            raise RuntimeError("No solution found.")

        # Create a CSV with image indices
        csv_path = tmp_path / "images.csv"
        csv_path.write_text("0\n")

        mock_img = torch.rand(1, 3, 3)
        mock_dataset_instance = MagicMock()
        mock_dataset_instance.__getitem__ = MagicMock(return_value=(mock_img, 0))

        monkeypatch.chdir(tmp_path)
        mock_tfwb = TransformWithBounds(Rotation(x0=1, y0=1), 0.0, 0.05)

        with (
            mock_patch("scripts.calculate_bounds.DIR_RESULTS", tmp_path),
            mock_patch(
                "scripts.calculate_bounds.parse_transform", return_value=mock_tfwb
            ),
            mock_patch("vnn.pwl.lp.solve_gurobi", side_effect=mock_solve_gurobi),
        ):
            with mock_patch("scripts.calculate_bounds.datasets") as mock_ds_mod:
                mock_ds_mod.get_dataset_choices.return_value = ["MNIST"]
                mock_cls = MagicMock(return_value=mock_dataset_instance)
                setattr(mock_ds_mod, "MNIST", mock_cls)

                main(
                    [
                        "--transformations",
                        "ROTATE",
                        "--lower",
                        "0",
                        "--upper",
                        "0.05",
                        "--dataset",
                        "MNIST",
                        "--path-images-csv",
                        str(csv_path),
                        "--num-samples",
                        "5",
                        "--num-init-splits",
                        "2",
                        "--num-splits",
                        "2",
                        "--num-subdomain-samples",
                        "3",
                        "--max-bab-iter",
                        "50",
                        "--lipschitz-error",
                        "0.1",
                    ]
                )

    def test_main_load_bounds(self, tmp_path, monkeypatch):
        """Test the main function with --load-bounds."""
        from scripts.calculate_bounds import main, get_experiment_name
        from unittest.mock import patch as mock_patch

        mock_img = torch.rand(1, 3, 3)
        mock_dataset_instance = MagicMock()
        mock_dataset_instance.__getitem__ = MagicMock(return_value=(mock_img, 0))

        monkeypatch.chdir(tmp_path)
        mock_tfwb = TransformWithBounds(Rotation(x0=1, y0=1), 0.0, 0.05)

        # Pre-create the bounds pickle file with a fixed timestamp
        fixed_ts = "2026-01-01-00-00-00"
        bounds_dir = tmp_path / get_experiment_name(
            fixed_ts, "MNIST", [TransformType.ROTATE], [0.0], [0.05], 0.5
        )
        bounds_dir.mkdir(parents=True)

        import pickle

        bounds_data = {
            "linear": {
                "sound": {
                    "lower": np.zeros((1, 3, 3, 2)),
                    "upper": np.ones((1, 3, 3, 2)),
                    "num_bab_lb": [],
                    "num_bab_ub": [],
                }
            },
            "pwl": {
                "sound": {
                    "lower": np.zeros((1, 3, 3, 2, 4)),
                    "upper": np.ones((1, 3, 3, 2, 4)),
                    "num_bab_lb": [],
                    "num_bab_ub": [],
                }
            },
        }
        bounds_pkl = bounds_dir / "0.pkl"
        with bounds_pkl.open("wb") as f:
            pickle.dump({"bounds": bounds_data}, f)

        with (
            mock_patch("scripts.calculate_bounds.DIR_RESULTS", tmp_path),
            mock_patch(
                "scripts.calculate_bounds.parse_transform", return_value=mock_tfwb
            ),
            mock_patch("scripts.calculate_bounds.datetime") as mock_dt,
        ):
            mock_dt.now.return_value.strftime.return_value = fixed_ts
            with mock_patch("scripts.calculate_bounds.datasets") as mock_ds_mod:
                mock_ds_mod.get_dataset_choices.return_value = ["MNIST"]
                mock_cls = MagicMock(return_value=mock_dataset_instance)
                setattr(mock_ds_mod, "MNIST", mock_cls)

                main(
                    [
                        "--transformations",
                        "ROTATE",
                        "--lower",
                        "0",
                        "--upper",
                        "0.05",
                        "--dataset",
                        "MNIST",
                        "--image-number",
                        "0",
                        "--load-bounds",
                        "--num-samples",
                        "5",
                    ]
                )

    def test_main_with_plot(self, tmp_path, monkeypatch):
        """Test the main function with --plot flag."""
        from scripts.calculate_bounds import main
        from unittest.mock import patch as mock_patch
        import scipy.optimize as opt

        def mock_solve_gurobi(c, A, b):
            res = opt.linprog(c, A_ub=A, b_ub=b)
            if res.success:
                return res.x[0], res.x[1]
            raise RuntimeError("No solution found.")

        mock_img = torch.rand(1, 3, 3)
        mock_dataset_instance = MagicMock()
        mock_dataset_instance.__getitem__ = MagicMock(return_value=(mock_img, 0))

        monkeypatch.chdir(tmp_path)
        mock_tfwb = TransformWithBounds(Rotation(x0=1, y0=1), 0.0, 0.05)

        with (
            mock_patch("scripts.calculate_bounds.DIR_RESULTS", tmp_path),
            mock_patch(
                "scripts.calculate_bounds.parse_transform", return_value=mock_tfwb
            ),
            mock_patch("vnn.pwl.lp.solve_gurobi", side_effect=mock_solve_gurobi),
            mock_patch("scripts.calculate_bounds.plot_bounds"),
            mock_patch("scripts.calculate_bounds.Parallel") as mock_parallel,
        ):
            # Make Parallel actually invoke the delayed callables
            def run_delayed(iterable):
                return [func(*args, **kwargs) for func, args, kwargs in iterable]

            mock_parallel.return_value = run_delayed

            with mock_patch("scripts.calculate_bounds.datasets") as mock_ds_mod:
                mock_ds_mod.get_dataset_choices.return_value = ["MNIST"]
                mock_cls = MagicMock(return_value=mock_dataset_instance)
                setattr(mock_ds_mod, "MNIST", mock_cls)

                main(
                    [
                        "--transformations",
                        "ROTATE",
                        "--lower",
                        "0",
                        "--upper",
                        "0.05",
                        "--dataset",
                        "MNIST",
                        "--image-number",
                        "0",
                        "--plot",
                        "--num-samples",
                        "5",
                        "--num-init-splits",
                        "2",
                        "--num-splits",
                        "2",
                        "--num-subdomain-samples",
                        "3",
                        "--max-bab-iter",
                        "50",
                        "--lipschitz-error",
                        "0.1",
                    ]
                )

    def test_main_with_error_handling(self, tmp_path, monkeypatch):
        """Test the main function error handling when image processing fails."""
        from scripts.calculate_bounds import main
        from unittest.mock import patch as mock_patch

        mock_dataset_instance = MagicMock()
        # Make __getitem__ raise an exception to trigger the error handler
        mock_dataset_instance.__getitem__ = MagicMock(
            side_effect=RuntimeError("Bad image")
        )

        monkeypatch.chdir(tmp_path)
        mock_tfwb = TransformWithBounds(Rotation(x0=1, y0=1), 0.0, 0.05)

        with (
            mock_patch("scripts.calculate_bounds.DIR_RESULTS", tmp_path),
            mock_patch(
                "scripts.calculate_bounds.parse_transform", return_value=mock_tfwb
            ),
        ):
            with mock_patch("scripts.calculate_bounds.datasets") as mock_ds_mod:
                mock_ds_mod.get_dataset_choices.return_value = ["MNIST"]
                mock_cls = MagicMock(return_value=mock_dataset_instance)
                setattr(mock_ds_mod, "MNIST", mock_cls)

                # Should not raise - errors are caught and logged
                main(
                    [
                        "--transformations",
                        "ROTATE",
                        "--lower",
                        "0",
                        "--upper",
                        "0.05",
                        "--dataset",
                        "MNIST",
                        "--image-number",
                        "0",
                        "--num-samples",
                        "5",
                    ]
                )


# ============================================================================
# scripts/read_bounds.py
# ============================================================================
class TestReadBounds:
    def test_read_bounds(self, tmp_path):
        from scripts.read_bounds import read_bounds
        import matplotlib

        matplotlib.use("Agg")

        # Create mock bounds data
        lin_bound = np.zeros((5, 5, 2))
        pwl_bound = np.zeros((5, 5, 2, 4))  # 2 segments, 4 values each
        pwl_bound[:, :, :, 2] = 0.0  # segment start
        pwl_bound[:, :, :, 3] = 0.1  # segment end

        data = {
            "bounds": {
                "linear": {
                    "unsound": {"lower": lin_bound, "upper": lin_bound},
                    "sound": {"lower": lin_bound, "upper": lin_bound},
                },
                "pwl": {
                    "unsound": {"lower": pwl_bound, "upper": pwl_bound},
                    "sound": {"lower": pwl_bound, "upper": pwl_bound},
                },
            }
        }

        path_bounds = tmp_path / "bounds.pkl"
        with path_bounds.open("wb") as f:
            pickle.dump(data, f)

        import matplotlib.pyplot as plt

        with patch.object(plt, "show"):
            read_bounds(path_bounds, 2, 2)
        plt.close("all")

    def test_main(self, tmp_path):
        """Exercise main() with explicit argv."""
        from scripts.read_bounds import main
        import matplotlib
        import matplotlib.pyplot as plt

        matplotlib.use("Agg")

        # Create mock bounds data
        lin_bound = np.zeros((5, 5, 2))
        pwl_bound = np.zeros((5, 5, 2, 4))
        pwl_bound[:, :, :, 2] = 0.0
        pwl_bound[:, :, :, 3] = 0.1

        data = {
            "bounds": {
                "linear": {
                    "unsound": {"lower": lin_bound, "upper": lin_bound},
                    "sound": {"lower": lin_bound, "upper": lin_bound},
                },
                "pwl": {
                    "unsound": {"lower": pwl_bound, "upper": pwl_bound},
                    "sound": {"lower": pwl_bound, "upper": pwl_bound},
                },
            }
        }

        path_bounds = tmp_path / "bounds.pkl"
        with path_bounds.open("wb") as f:
            pickle.dump(data, f)

        with patch.object(plt, "show"):
            main([str(path_bounds), "2", "2"])
        plt.close("all")
