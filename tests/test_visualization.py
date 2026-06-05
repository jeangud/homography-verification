import pytest
import matplotlib.pyplot as plt
import numpy as np
from scipy import constants
from unittest.mock import Mock, patch
from vnn import visualization
from vnn.transforms import TransformWithBounds
from vnn.transforms.rotation import Rotation


def test_setup_latex():
    # Cache state
    original_usetex = plt.rcParams.get("text.usetex")

    try:
        visualization.setup_latex()

        # Verify changes
        assert plt.rcParams["text.usetex"] is True
        assert plt.rcParams["font.family"] == ["serif"]
        assert plt.rcParams["font.size"] == 9.0
    finally:
        # Restore state to prevent messing up other tests
        if original_usetex is not None:
            plt.rcParams["text.usetex"] = original_usetex


def test_setup_latex_fallback():
    """setup_latex() falls back gracefully when LaTeX is not installed."""
    original_usetex = plt.rcParams.get("text.usetex")

    try:
        plt.rcParams["text.usetex"] = False

        with patch("vnn.visualization.shutil.which", return_value=None):
            visualization.setup_latex()

        assert plt.rcParams["text.usetex"] is False
        assert plt.rcParams["font.family"] == ["serif"]
        assert plt.rcParams["font.size"] == 9.0
    finally:
        plt.rcParams["text.usetex"] = original_usetex


def test_get_figure_size():
    # width_pt = 100, fraction = 1, aspect_ratio = 1
    # inches_per_pt = 1 / 72
    width_in, height_in = visualization.get_figure_size(100.0)

    assert isinstance(width_in, float)
    assert isinstance(height_in, float)
    assert width_in > 0
    assert height_in > 0

    # 100 points to inches
    expected_inches = 100.0 * (constants.point / constants.inch)
    np.testing.assert_allclose(width_in, expected_inches)
    np.testing.assert_allclose(height_in, expected_inches)

    # Test aspect ratio modification
    width_in2, height_in2 = visualization.get_figure_size(100.0, aspect_ratio=2.0)
    np.testing.assert_allclose(width_in2, expected_inches)
    np.testing.assert_allclose(height_in2, expected_inches / 2.0)


def test_plot_samples():
    # Lightweight test for matplotlib figure generation
    # We verify that it doesn't crash and returns a Figure object

    # Generate mock samples: (num_samples, channels, height, width)
    mock_samples = np.random.rand(10, 1, 32, 32)

    fig = visualization.plot_samples(mock_samples, num_images=5)

    assert isinstance(fig, plt.Figure)
    assert len(fig.axes) == 5  # We asked for 5 images

    # Best practice: always close figures in tests to prevent accumulating memory
    plt.close(fig)


def test_plot_samples_with_point():
    # Verify the branch where a specific pixel is highlighted with i and j
    mock_samples = np.random.rand(10, 1, 32, 32)
    fig = visualization.plot_samples(mock_samples, num_images=5, i=16, j=16)

    assert isinstance(fig, plt.Figure)
    assert len(fig.axes) == 5
    plt.close(fig)


def test_plot_transform():
    # Verify the plot_transform method handles simple transformations without crashing
    img = np.random.rand(1, 5, 5)
    tf = Rotation()

    fig = visualization.plot_transform(
        img=img,
        transform=tf,
        p_min=-1.0,
        p_max=1.0,
        i=2,
        j=2,
        padding_value=0.0,
        num_samples=10,
    )

    assert isinstance(fig, plt.Figure)
    plt.close(fig)


@pytest.fixture
def mock_bounds_data():
    c, i, j = 0, 2, 2

    # 1 channel, 5x5 image, 2 values for linear (w_lo/hi, b_lo/hi)
    lin_bound = np.zeros((1, 5, 5, 2))
    # 1 channel, 5x5 image, 3 domains, 2 values per domain
    pwl_bound = np.zeros((1, 5, 5, 3, 2))

    bounds = {
        "linear": {
            "sound": {"lower": lin_bound, "upper": lin_bound},
            "unsound": {"lower": lin_bound, "upper": lin_bound},
        },
        "pwl": {
            "sound": {"lower": pwl_bound, "upper": pwl_bound},
            "unsound": {"lower": pwl_bound, "upper": pwl_bound},
        },
    }

    samples = np.zeros((10, 1, 5, 5))
    params = np.linspace(-1.0, 1.0, 10)

    return {
        "c": c,
        "i": i,
        "j": j,
        "bounds": bounds,
        "samples": samples,
        "params": params,
    }


def test_plot_bounds(mock_bounds_data):
    # Verify the plot_bounds method with a mock set of bounding curves
    tf = Rotation()
    tfwb = TransformWithBounds(tf, -1.0, 1.0)

    fig = visualization.plot_bounds(
        c=mock_bounds_data["c"],
        i=mock_bounds_data["i"],
        j=mock_bounds_data["j"],
        samples=mock_bounds_data["samples"],
        params=mock_bounds_data["params"],
        tfwb=tfwb,
        bounds=mock_bounds_data["bounds"],
        num_samples=10,
    )

    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_plot_bounds_non_rotation(mock_bounds_data):
    # Verify the plot_bounds fallback logic for standard non-rotation parameters
    tf = Mock()  # Not a Rotation/Homography, drops into standard parameters block
    tfwb = TransformWithBounds(tf, -1.0, 1.0)

    fig = visualization.plot_bounds(
        c=mock_bounds_data["c"],
        i=mock_bounds_data["i"],
        j=mock_bounds_data["j"],
        samples=mock_bounds_data["samples"],
        params=mock_bounds_data["params"],
        tfwb=tfwb,
        bounds=mock_bounds_data["bounds"],
        num_samples=10,
    )

    assert isinstance(fig, plt.Figure)
    plt.close(fig)
