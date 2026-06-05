import numpy as np

from vnn.transforms.homogeneous import from_homogeneous, to_homogeneous


def test_to_homogeneous():
    # Shape: (2, N) where N=3 points
    xy = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    xy_homo = to_homogeneous(xy)

    assert xy_homo.shape == (3, 3)
    np.testing.assert_allclose(xy_homo[2, :], [1.0, 1.0, 1.0])
    np.testing.assert_allclose(xy_homo[:2, :], xy)


def test_from_homogeneous():
    # Shape: (3, N) where N=2 points
    xyz = np.array(
        [
            [2.0, 4.0],
            [6.0, 8.0],
            [2.0, 4.0],  # The w-component (scale factor)
        ]
    )

    xy_expected = np.array([[1.0, 1.0], [3.0, 2.0]])

    xy = from_homogeneous(xyz)
    np.testing.assert_allclose(xy, xy_expected)
