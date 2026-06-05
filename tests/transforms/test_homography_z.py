"""Tests for HomographyZ transform class."""

import numpy as np
import pytest

from vnn.transforms.homography_z import HomographyZ


class TestHomographyZ:
    def test_init(self):
        h = HomographyZ(f=10.0, xc=5.0, yc=5.0, z=-10.0)
        assert h.f == 10.0
        assert h.z == -10.0

    def test_prepare_matrix_identity(self):
        h = HomographyZ(f=10.0, xc=0.0, yc=0.0, z=-10.0)
        params = np.array([0.0])
        H = h.prepare_matrix(params)
        assert H.shape == (1, 3, 3)
        np.testing.assert_allclose(H[0], np.eye(3), atol=1e-10)

    def test_prepare_matrix_shape(self):
        h = HomographyZ(f=10.0, xc=5.0, yc=5.0, z=-10.0)
        params = np.linspace(-1.0, 1.0, 4)
        H = h.prepare_matrix(params)
        assert H.shape == (4, 3, 3)

    def test_gradient_shape(self):
        h = HomographyZ(f=10.0, xc=0.0, yc=0.0, z=-10.0)
        grad = h.gradient(1.0, 2.0, [0.0, 0.5])
        assert grad.shape == (2, 2)
        # First component (dx) should be 0
        np.testing.assert_allclose(grad[0, :], 0.0)

    def test_get_max_grad_candidates(self):
        h = HomographyZ(f=10.0, xc=0.0, yc=0.0, z=-10.0)
        grad = h.get_max_grad_candidates(1.0, 1.0, interval=(0.0, 1.0))
        assert grad.shape[0] == 2

    def test_get_max_grad_candidates_critical_raises(self):
        h = HomographyZ(f=10.0, xc=0.0, yc=0.0, z=-10.0)
        # Critical: dz = -z = 10
        with pytest.raises(ValueError, match="Critical z"):
            h.get_max_grad_candidates(1.0, 1.0, interval=(5.0, 15.0))
