"""Tests for HomographyRoll transform class."""

import numpy as np

from vnn.transforms.homography_roll import HomographyRoll


class TestHomographyRoll:
    def test_init(self):
        h = HomographyRoll(xc=5.0, yc=5.0)
        assert h.xc == 5.0
        assert h.yc == 5.0

    def test_prepare_matrix_identity(self):
        h = HomographyRoll(xc=0.0, yc=0.0)
        params = np.array([0.0])
        H = h.prepare_matrix(params)
        assert H.shape == (1, 3, 3)
        np.testing.assert_allclose(H[0], np.eye(3), atol=1e-10)

    def test_prepare_matrix_shape(self):
        h = HomographyRoll(xc=5.0, yc=5.0)
        params = np.linspace(0, 0.1, 10)
        H = h.prepare_matrix(params)
        assert H.shape == (10, 3, 3)

    def test_gradient_shape(self):
        h = HomographyRoll(xc=5.0, yc=5.0)
        params = np.array([0.0, 0.05])
        grad = h.gradient(1.0, 2.0, params)
        assert grad.shape == (2, 2)

    def test_get_max_grad_candidates(self):
        h = HomographyRoll(xc=0.0, yc=0.0)
        grad = h.get_max_grad_candidates(1.0, 1.0, interval=(0.0, 0.1))
        assert grad.shape[0] == 2

    def test_get_max_grad_candidates_wide_interval(self):
        h = HomographyRoll(xc=0.0, yc=0.0)
        grad = h.get_max_grad_candidates(1.0, 1.0, interval=(-1.0, 1.0))
        assert grad.shape[0] == 2

    def test_get_max_grad_candidates_candidate_modulo(self):
        h = HomographyRoll(xc=0.0, yc=0.0)
        # phi1 = arctan(-1/1) = -pi/4, phi2 = arctan(1/1) = pi/4
        # Interval around phi1 + pi = 3*pi/4: shifted candidate should be included
        phi1 = np.arctan(-1.0)
        grad = h.get_max_grad_candidates(
            1.0, 1.0, interval=(phi1 + np.pi - 0.1, phi1 + np.pi + 0.1)
        )
        # 2 boundary candidates + 1 shifted candidate = 3 columns
        assert grad.shape == (2, 3)
