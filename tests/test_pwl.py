import numpy as np
import pytest

from vnn import pwl
from vnn.linear_programming import Solver
from vnn.transforms.rotation import Rotation
from vnn.transforms.transform_with_bounds import TransformWithBounds


def test_split():
    subdomain = (0.0, 10.0)
    splits = pwl.split(subdomain, 2)

    assert isinstance(splits, list)
    assert len(splits) == 2

    # First split
    np.testing.assert_allclose(splits[0], [0.0, 5.0])
    # Second split
    np.testing.assert_allclose(splits[1], [5.0, 10.0])


def test_split_single():
    subdomain = (-1.0, 1.0)
    splits = pwl.split(subdomain, 1)

    assert len(splits) == 1
    np.testing.assert_allclose(splits[0], [-1.0, 1.0])


def test_split_many():
    subdomain = (0.0, 1.0)
    splits = pwl.split(subdomain, 5)
    assert len(splits) == 5
    np.testing.assert_allclose(splits[0][0], 0.0)
    np.testing.assert_allclose(splits[-1][1], 1.0)


def test_compute_split_segments():
    params = np.array([0, 1, 2, 3, 4])
    segments = pwl.compute_split_segments(params, 4)

    assert len(segments) == 4
    np.testing.assert_allclose(segments[0], [0.0, 1.0])
    np.testing.assert_allclose(segments[-1], [3.0, 4.0])


@pytest.fixture
def tiny_image():
    """A tiny single-channel 3x3 image for fast testing."""
    np.random.seed(42)
    return np.random.rand(1, 3, 3).astype(np.float32)


@pytest.fixture
def tiny_tfwb():
    """A small rotation transform with bounds."""
    rotation = Rotation(x0=1.0, y0=1.0)
    return TransformWithBounds(rotation, lower_bound=0.0, upper_bound=0.05)


def test_generate_samples(tiny_image, tiny_tfwb):
    samples, params = pwl.generate_samples(
        tiny_image,
        tiny_tfwb.transform,
        lower_bound=tiny_tfwb.lower_bound,
        upper_bound=tiny_tfwb.upper_bound,
        num_samples=10,
        padding=0.0,
    )
    assert samples.shape == (10, 1, 3, 3)
    assert params.shape == (10,)
    np.testing.assert_allclose(params[0], 0.0)
    np.testing.assert_allclose(params[-1], 0.05)


def test_generate_samples_identity(tiny_image, tiny_tfwb):
    """With zero rotation, first sample should be ~original image."""
    samples, params = pwl.generate_samples(
        tiny_image,
        tiny_tfwb.transform,
        lower_bound=0.0,
        upper_bound=0.0,
        num_samples=1,
        padding=0.0,
    )
    np.testing.assert_allclose(samples[0], tiny_image, atol=1e-5)


def test_fit_linear_bound(tiny_image, tiny_tfwb):
    """Test unsound linear bound fitting."""
    samples, params = pwl.generate_samples(
        tiny_image,
        tiny_tfwb.transform,
        lower_bound=tiny_tfwb.lower_bound,
        upper_bound=tiny_tfwb.upper_bound,
        num_samples=10,
        padding=0.0,
    )
    lb = pwl.fit_linear_bound(
        samples, params, is_upper_bound=False, solver=Solver.SCIPY
    )
    ub = pwl.fit_linear_bound(samples, params, is_upper_bound=True, solver=Solver.SCIPY)

    # Bounds shape: (channels, rows, cols, 2) where 2 = [slope, intercept]
    assert lb.shape == (1, 3, 3, 2)
    assert ub.shape == (1, 3, 3, 2)


def test_fit_pwl_bound(tiny_image, tiny_tfwb):
    """Test unsound PWL bound fitting."""
    samples, params = pwl.generate_samples(
        tiny_image,
        tiny_tfwb.transform,
        lower_bound=tiny_tfwb.lower_bound,
        upper_bound=tiny_tfwb.upper_bound,
        num_samples=10,
        padding=0.0,
    )
    lb_pwl = pwl.fit_pwl_bound(
        samples, params, is_upper_bound=False, num_segments=2, solver=Solver.SCIPY
    )
    ub_pwl = pwl.fit_pwl_bound(
        samples, params, is_upper_bound=True, num_segments=2, solver=Solver.SCIPY
    )

    # Shape: (channels, rows, cols, num_segments, 4)
    assert lb_pwl.shape == (1, 3, 3, 2, 4)
    assert ub_pwl.shape == (1, 3, 3, 2, 4)


def test_shift_linear_bound(tiny_image, tiny_tfwb):
    """Test sound linear bound shifting."""
    samples, params = pwl.generate_samples(
        tiny_image,
        tiny_tfwb.transform,
        lower_bound=tiny_tfwb.lower_bound,
        upper_bound=tiny_tfwb.upper_bound,
        num_samples=10,
        padding=0.0,
    )
    lb = pwl.fit_linear_bound(
        samples, params, is_upper_bound=False, solver=Solver.SCIPY
    )

    lb_sound, bab_iters = pwl.shift_linear_bound(
        lb,
        tiny_image,
        tiny_tfwb,
        lipschitz_error=0.1,
        num_init_splits=2,
        num_splits=2,
        num_samples_per_subdomain=5,
        padding_value=0.0,
        is_upper_bound=False,
        max_iterations=100,
    )
    assert lb_sound.shape == lb.shape
    assert isinstance(bab_iters, list)


def test_shift_linear_bound_upper(tiny_image, tiny_tfwb):
    """Test sound upper linear bound shifting."""
    samples, params = pwl.generate_samples(
        tiny_image,
        tiny_tfwb.transform,
        lower_bound=tiny_tfwb.lower_bound,
        upper_bound=tiny_tfwb.upper_bound,
        num_samples=10,
        padding=0.0,
    )
    ub = pwl.fit_linear_bound(samples, params, is_upper_bound=True, solver=Solver.SCIPY)

    ub_sound, bab_iters = pwl.shift_linear_bound(
        ub,
        tiny_image,
        tiny_tfwb,
        lipschitz_error=0.1,
        num_init_splits=2,
        num_splits=2,
        num_samples_per_subdomain=5,
        padding_value=0.0,
        is_upper_bound=True,
        max_iterations=100,
    )
    assert ub_sound.shape == ub.shape


def test_shift_pwl_bound(tiny_image, tiny_tfwb):
    """Test sound PWL bound shifting."""
    samples, params = pwl.generate_samples(
        tiny_image,
        tiny_tfwb.transform,
        lower_bound=tiny_tfwb.lower_bound,
        upper_bound=tiny_tfwb.upper_bound,
        num_samples=10,
        padding=0.0,
    )
    ub_pwl = pwl.fit_pwl_bound(
        samples, params, is_upper_bound=True, num_segments=2, solver=Solver.SCIPY
    )

    ub_sound, bab_iters = pwl.shift_pwl_bound(
        ub_pwl,
        tiny_image,
        tiny_tfwb,
        lipschitz_error=0.1,
        num_init_splits=2,
        num_splits=2,
        num_samples_per_subdomain=5,
        padding_value=0.0,
        is_upper_bound=True,
        max_iterations=100,
    )
    assert ub_sound.shape == ub_pwl.shape
    assert isinstance(bab_iters, list)


def test_shift_pwl_bound_lower(tiny_image, tiny_tfwb):
    """Test sound lower PWL bound shifting."""
    samples, params = pwl.generate_samples(
        tiny_image,
        tiny_tfwb.transform,
        lower_bound=tiny_tfwb.lower_bound,
        upper_bound=tiny_tfwb.upper_bound,
        num_samples=10,
        padding=0.0,
    )
    lb_pwl = pwl.fit_pwl_bound(
        samples, params, is_upper_bound=False, num_segments=2, solver=Solver.SCIPY
    )

    lb_sound, bab_iters = pwl.shift_pwl_bound(
        lb_pwl,
        tiny_image,
        tiny_tfwb,
        lipschitz_error=0.1,
        num_init_splits=2,
        num_splits=2,
        num_samples_per_subdomain=5,
        padding_value=0.0,
        is_upper_bound=False,
        max_iterations=100,
    )
    assert lb_sound.shape == lb_pwl.shape


def test_calculate_bounds_full(tiny_image, tiny_tfwb):
    """Integration test for the full bounds calculation pipeline."""
    result = pwl.calculate_bounds(
        img=tiny_image,
        transform_with_bounds=tiny_tfwb,
        padding=0.0,
        num_samples=10,
        lipschitz_error=0.1,
        num_init_splits=2,
        num_splits=2,
        num_subdomains=5,
        max_iterations=100,
        solver=Solver.SCIPY,
    )

    assert "params" in result
    assert "samples" in result
    assert "linear" in result
    assert "pwl" in result

    # Check linear bounds structure
    assert "unsound" in result["linear"]
    assert "sound" in result["linear"]
    assert result["linear"]["sound"]["lower"].shape == (1, 3, 3, 2)
    assert result["linear"]["sound"]["upper"].shape == (1, 3, 3, 2)

    # Check PWL bounds structure
    assert "unsound" in result["pwl"]
    assert "sound" in result["pwl"]


def test_calculate_bounds_invalid_image():
    """Test that invalid image shapes are rejected."""
    rotation = Rotation(x0=1.0, y0=1.0)
    tfwb = TransformWithBounds(rotation, 0.0, 0.05)

    # Invalid shape: (10, 10) - missing channel dimension
    img = np.random.rand(10, 10).astype(np.float32)
    with pytest.raises(ValueError, match="single-channel"):
        pwl.calculate_bounds(
            img=img,
            transform_with_bounds=tfwb,
            padding=0.0,
            num_samples=5,
            lipschitz_error=0.1,
            num_init_splits=2,
            num_splits=2,
            num_subdomains=5,
            max_iterations=100,
        )


def test_shift_linear_bound_max_iterations():
    """Test that exceeding max iterations raises RuntimeError."""
    np.random.seed(0)
    img = np.random.rand(1, 3, 3).astype(np.float32)
    rotation = Rotation(x0=1.0, y0=1.0)
    tfwb = TransformWithBounds(rotation, lower_bound=0.0, upper_bound=1.0)

    samples, params = pwl.generate_samples(
        img,
        tfwb.transform,
        0.0,
        1.0,
        num_samples=10,
        padding=0.0,
    )
    lb = pwl.fit_linear_bound(
        samples, params, is_upper_bound=False, solver=Solver.SCIPY
    )

    with pytest.raises(RuntimeError, match="Exceeded max"):
        pwl.shift_linear_bound(
            lb,
            img,
            tfwb,
            lipschitz_error=1e-10,
            num_init_splits=2,
            num_splits=2,
            num_samples_per_subdomain=5,
            padding_value=0.0,
            is_upper_bound=False,
            max_iterations=1,
        )


def test_shift_pwl_bound_max_iterations():
    """Test that exceeding max iterations raises RuntimeError for PWL."""
    np.random.seed(0)
    img = np.random.rand(1, 3, 3).astype(np.float32)
    rotation = Rotation(x0=1.0, y0=1.0)
    tfwb = TransformWithBounds(rotation, lower_bound=0.0, upper_bound=1.0)

    samples, params = pwl.generate_samples(
        img,
        tfwb.transform,
        0.0,
        1.0,
        num_samples=10,
        padding=0.0,
    )
    ub_pwl = pwl.fit_pwl_bound(
        samples, params, is_upper_bound=True, num_segments=2, solver=Solver.SCIPY
    )

    with pytest.raises(RuntimeError, match="Exceeded max"):
        pwl.shift_pwl_bound(
            ub_pwl,
            img,
            tfwb,
            lipschitz_error=1e-10,
            num_init_splits=2,
            num_splits=2,
            num_samples_per_subdomain=5,
            padding_value=0.0,
            is_upper_bound=True,
            max_iterations=1,
        )
