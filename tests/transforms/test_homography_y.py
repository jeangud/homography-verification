"""Tests for HomographyY transform class."""

import numpy as np

from vnn.transforms.homography_y import HomographyY


class TestHomographyY:
    def test_init(self):
        h = HomographyY(f=10.0, xc=5.0, yc=5.0, z=-10.0)
        assert h.f == 10.0
        assert h.xc == 5.0

    def test_prepare_matrix_identity(self):
        h = HomographyY(f=10.0, xc=0.0, yc=0.0, z=-10.0)
        params = np.array([0.0])
        H = h.prepare_matrix(params)
        assert H.shape == (1, 3, 3)
        np.testing.assert_allclose(H[0], np.eye(3), atol=1e-10)

    def test_prepare_matrix_shape(self):
        h = HomographyY(f=10.0, xc=5.0, yc=5.0, z=-10.0)
        params = np.linspace(-1.0, 1.0, 3)
        H = h.prepare_matrix(params)
        assert H.shape == (3, 3, 3)

    def test_gradient_constant(self):
        h = HomographyY(f=10.0, xc=5.0, yc=5.0, z=-10.0)
        # Gradient is constant w.r.t. params
        grad = h.gradient(1.0, 2.0, [0.0, 0.5, 1.0])
        assert grad.shape == (2, 3)
        # All gradient values in each row should be the same
        np.testing.assert_allclose(grad[0, 0], grad[0, 1])
        np.testing.assert_allclose(grad[0, 1], grad[0, 2])

    def test_get_max_grad_candidates(self):
        h = HomographyY(f=10.0, xc=0.0, yc=0.0, z=-10.0)
        grad = h.get_max_grad_candidates(1.0, 1.0, interval=(0.0, 1.0))
        # Only returns one candidate since gradient is constant
        assert grad.shape[0] == 2
