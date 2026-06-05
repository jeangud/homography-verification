"""Tests for the Transform abstract base class, using Rotation as concrete implementation."""

import itertools

import numpy as np
import pytest

from vnn.transforms.homography_roll import HomographyRoll
from vnn.transforms.rotation import Rotation
from vnn.transforms.transform import _values_in_interval


@pytest.fixture
def rotation():
    return Rotation(x0=5.0, y0=5.0)


@pytest.fixture
def small_img():
    """A small 1-channel 4x4 image."""
    np.random.seed(0)
    return np.random.rand(1, 4, 4).astype(np.float32)


class TestTransformStr:
    def test_rotation_str(self, rotation):
        assert str(rotation) == "Rotation(5.0,5.0)"

    def test_default_str(self):
        r = Rotation()
        assert "Rotation" in str(r)

    def test_base_str(self):
        tf = HomographyRoll(xc=0.0, yc=0.0)
        assert str(tf) == "HomographyRoll"


class TestApply:
    def test_apply_single_param(self, rotation, small_img):
        params = np.array([0.0])
        result = rotation.apply(small_img, params, padding=0.0)
        # Zero rotation should return ~original image
        assert result.shape == (1, 1, 4, 4)
        np.testing.assert_allclose(result[0], small_img, atol=1e-5)

    def test_apply_multiple_params(self, rotation, small_img):
        params = np.array([0.0, 0.01, 0.02])
        result = rotation.apply(small_img, params, padding=0.0)
        assert result.shape == (3, 1, 4, 4)

    def test_apply_with_border_replicate(self, rotation, small_img):
        params = np.array([0.05])
        result = rotation.apply(small_img, params, padding="BORDER_REPLICATE")
        assert result.shape == (1, 1, 4, 4)


class TestApplyOpenCV:
    def test_apply_opencv_identity(self, rotation):
        img = np.random.rand(1, 4, 4).astype(np.float32)  # (c, h, w)
        params = np.array([0.0])
        result = rotation.apply_opencv(img, params, padding_value=0.0)
        assert result.shape == (1, 1, 4, 4)  # (num_params, c, h, w)

    def test_apply_opencv_multiple(self, rotation):
        img = np.random.rand(1, 4, 4).astype(np.float32)  # (c, h, w)
        params = np.array([0.0, 0.05, 0.1])
        result = rotation.apply_opencv(img, params, padding_value=0.0)
        assert result.shape == (3, 1, 4, 4)  # (num_params, c, h, w)


class TestApplyPil:
    def test_apply_pil_identity(self):
        rotation = Rotation(x0=4.0, y0=4.0)
        img = np.random.rand(1, 8, 8).astype(np.float32)  # (c, h, w)
        params = np.array([0.0])
        result = rotation.apply_pil(img, params, padding_value=0)
        assert result.shape == (1, 1, 8, 8)  # (num_params, c, h, w)


class TestApplyTorch:
    def test_apply_torch_identity(self):
        rotation = Rotation(x0=4.0, y0=4.0)
        img = np.random.rand(1, 8, 8).astype(np.float32)  # (c, h, w)
        params = np.array([0.0])
        result = rotation.apply_torch(img, params, padding_value=0)
        assert result.shape == (1, 1, 8, 8)  # (num_params, c, h, w)


class TestGetOriginalPixelLocations:
    def test_identity_rotation(self, small_img):
        rotation = Rotation(x0=0.0, y0=0.0)
        params = np.array([0.0])
        i_coords, j_coords = rotation.get_original_pixel_locations(small_img, params)
        # For zero rotation: to_xy(+0.5) then identity H, then to_ij(-0.5) → net identity
        h, w = small_img.shape[-2:]
        expected_i_coords = np.repeat(np.arange(h), w).astype(float)
        expected_j_coords = np.tile(np.arange(w), h).astype(float)
        np.testing.assert_allclose(i_coords, expected_i_coords, atol=1e-10)
        np.testing.assert_allclose(j_coords, expected_j_coords, atol=1e-10)


class TestLipschitz:
    def test_get_lipschitz_analytical(self, small_img):
        rotation = Rotation(x0=2.0, y0=2.0)
        L = rotation.get_lipschitz_analytical(
            small_img, x=1.0, y=1.0, interval=(0.0, 0.1)
        )
        assert isinstance(L, (float, np.floating))
        assert L >= 0

    def test_get_lipschitz_analytical_cached(self, small_img):
        rotation = Rotation(x0=2.0, y0=2.0)
        L1 = rotation.get_lipschitz_analytical(
            small_img, x=1.0, y=1.0, interval=(0.0, 0.1)
        )
        L2 = rotation.get_lipschitz_analytical(
            small_img, x=1.0, y=1.0, interval=(0.0, 0.1)
        )
        assert L1 == L2

    def test_get_lipschitz_numerical(self, small_img):
        rotation = Rotation(x0=2.0, y0=2.0)
        L = rotation.get_lipschitz_numerical(
            small_img, x=1.0, y=1.0, interval=(0.0, 0.1)
        )
        assert isinstance(L, (float, np.floating))
        assert L >= 0
        # Call again to exercise the cached __max_grad_I branch
        L2 = rotation.get_lipschitz_numerical(
            small_img, x=1.0, y=1.0, interval=(0.0, 0.1)
        )
        assert L == L2


class TestValuesInInterval:
    def test_empty(self):
        assert _values_in_interval(0.0, (1.0, 2.0)) == []

    def test_single_value_inside(self):
        vals = _values_in_interval(0.5, (0.0, 1.0))
        assert len(vals) == 1
        np.testing.assert_allclose(vals[0], 0.5)

    def test_value_at_lower_bound(self):
        vals = _values_in_interval(0.0, (0.0, 1.0))
        assert len(vals) == 1
        np.testing.assert_allclose(vals[0], 0.0)

    def test_value_at_upper_bound(self):
        vals = _values_in_interval(1.0, (0.0, 1.0))
        assert len(vals) == 1
        np.testing.assert_allclose(vals[0], 1.0)

    def test_shifted_copy_inside(self):
        # base = 0.5, interval = (pi+0.3, pi+0.7) → base + pi ≈ 3.64 inside
        vals = _values_in_interval(0.5, (np.pi + 0.3, np.pi + 0.7))
        assert len(vals) == 1
        np.testing.assert_allclose(vals[0], 0.5 + np.pi)

    def test_negative_shift(self):
        # base = 0.5, interval = (-pi+0.3, -pi+0.7) → base - pi ≈ -2.64 inside
        vals = _values_in_interval(0.5, (-np.pi + 0.3, -np.pi + 0.7))
        assert len(vals) == 1
        np.testing.assert_allclose(vals[0], 0.5 - np.pi)

    def test_multiple_copies(self):
        # Wide interval spanning several periods
        vals = _values_in_interval(0.0, (-0.1, 3 * np.pi + 0.1))
        # Should find 0, pi, 2*pi, 3*pi
        assert len(vals) == 4
        for i, v in enumerate(vals):
            np.testing.assert_allclose(v, i * np.pi, atol=1e-10)

    def test_custom_period(self):
        vals = _values_in_interval(0.0, (0.0, 4.5), period=2.0)
        assert len(vals) == 3
        np.testing.assert_allclose(vals, [0.0, 2.0, 4.0])


class TestImplementationConsistency:
    """Cross-check all four apply* implementations against each other.

    Each implementation uses a different bilinear interpolation backend, so
    results are not bit-exact.  We compare only the interior of the output
    (to avoid border-handling disagreements) and allow a tolerance that
    reflects sub-pixel interpolation differences across backends.
    """

    @pytest.fixture
    def smooth_img_chw(self):
        """A smooth 1-channel 32x32 image whose low gradient keeps interpolation
        errors small."""
        h, w = 32, 32
        x = np.linspace(0, 2 * np.pi, w)
        y = np.linspace(0, 2 * np.pi, h)
        xx, yy = np.meshgrid(x, y)
        img_2d = (0.5 + 0.5 * np.sin(xx) * np.cos(yy)).astype(np.float32)
        return img_2d[np.newaxis]  # (1, h, w)

    @pytest.mark.parametrize("angle", [0.0, 0.15, -0.1])
    def test_pairwise_consistency(self, smooth_img_chw, angle):
        img = smooth_img_chw
        _, h, w = img.shape
        rotation = Rotation(x0=w / 2.0, y0=h / 2.0)
        params = np.array([angle])

        implementations = {
            "custom": rotation.apply(img, params, padding=0.0),
            "opencv": rotation.apply_opencv(img, params, padding_value=0.0),
            "pil": rotation.apply_pil(img, params, padding_value=0),
            "torch": rotation.apply_torch(img, params, padding_value=0),
        }
        # All results are (num_params, c, h, w) — no squeezing needed

        # Crop a 3-pixel border: border handling diverges across backends
        b = 3
        interior = np.s_[:, :, b:-b, b:-b]

        for (name_a, res_a), (name_b, res_b) in itertools.combinations(
            implementations.items(), 2
        ):
            np.testing.assert_allclose(
                res_a[interior],
                res_b[interior],
                atol=0.05,
                err_msg=f"{name_a} vs {name_b} disagree for angle={angle}",
            )
