"""Tests for HomographyYaw transform class."""

import numpy as np
import pytest

from vnn.transforms.homography_yaw import HomographyYaw


class TestHomographyYaw:
    def test_init(self):
        h = HomographyYaw(f=10.0, xc=5.0, yc=5.0)
        assert h.f == 10.0
        assert h.xc == 5.0
        assert h.yc == 5.0

    def test_prepare_matrix_identity(self):
        h = HomographyYaw(f=10.0, xc=0.0, yc=0.0)
        params = np.array([0.0])
        H = h.prepare_matrix(params)
        assert H.shape == (1, 3, 3)
        np.testing.assert_allclose(H[0], np.eye(3), atol=1e-10)

    def test_prepare_matrix_shape(self):
        h = HomographyYaw(f=10.0, xc=5.0, yc=5.0)
        params = np.linspace(0, 0.05, 7)
        H = h.prepare_matrix(params)
        assert H.shape == (7, 3, 3)

    def test_gradient_shape(self):
        h = HomographyYaw(f=10.0, xc=5.0, yc=5.0)
        grad = h.gradient(1.0, 2.0, np.array([0.0, 0.01]))
        assert grad.shape == (2, 2)

    def test_get_max_grad_candidates(self):
        h = HomographyYaw(f=10.0, xc=0.0, yc=0.0)
        grad = h.get_max_grad_candidates(1.0, 1.0, interval=(0.0, 0.1))
        assert grad.shape[0] == 2

    def test_get_max_grad_candidates_critical_outside(self):
        h = HomographyYaw(f=10.0, xc=0.0, yc=0.0)
        # psi_0 = arctan((xc-x)/f) = arctan(-1/10) ≈ -0.0997
        # Use interval where psi_0 is outside
        grad = h.get_max_grad_candidates(1.0, 1.0, interval=(0.2, 0.3))
        assert grad.shape[0] == 2

    def test_get_max_grad_candidates_critical_raises(self):
        h = HomographyYaw(f=10.0, xc=0.0, yc=0.0)
        # Critical: arctan(f/(x-xc)) = arctan(10/1) ≈ 1.47
        with pytest.raises(ValueError, match="Critical angle"):
            h.get_max_grad_candidates(1.0, 1.0, interval=(1.0, 2.0))

    def test_get_max_grad_candidates_x_equals_xc(self):
        h = HomographyYaw(f=10.0, xc=5.0, yc=0.0)
        # When x == xc, psi_c = pi/2
        grad = h.get_max_grad_candidates(5.0, 1.0, interval=(0.0, 0.1))
        assert grad.shape[0] == 2

    def test_get_max_grad_candidates_critical_modulo_raises(self):
        h = HomographyYaw(f=10.0, xc=0.0, yc=0.0)
        # psi_c = arctan(10/1) ≈ 1.471, psi_c - pi ≈ -1.670
        # Interval around the shifted copy should raise
        psi_c = np.arctan(10.0 / 1.0)
        with pytest.raises(ValueError, match="Critical angle"):
            h.get_max_grad_candidates(
                1.0, 1.0, interval=(psi_c - np.pi - 0.1, psi_c - np.pi + 0.1)
            )

    def test_get_max_grad_candidates_candidate_modulo(self):
        h = HomographyYaw(f=10.0, xc=0.0, yc=0.0)
        # psi_0 = arctan(-1/10) ≈ -0.0997
        # Interval around psi_0 + pi: shifted candidate should be included
        psi_0 = np.arctan(-1.0 / 10.0)
        grad = h.get_max_grad_candidates(
            1.0, 1.0, interval=(psi_0 + np.pi - 0.1, psi_0 + np.pi + 0.1)
        )
        # 2 boundary candidates + 1 shifted candidate = 3 columns
        assert grad.shape == (2, 3)
