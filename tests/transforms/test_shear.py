"""Tests for ShearX transform class."""

import numpy as np

from vnn.transforms.shear import ShearX


class TestShearX:
    def test_init(self):
        s = ShearX(y0=5.0)
        assert s.y0 == 5.0

    def test_init_default(self):
        s = ShearX()
        assert s.y0 == 0

    def test_str(self):
        s = ShearX(y0=5.0)
        assert str(s) == "ShearX(5.0)"

    def test_prepare_matrix_identity(self):
        s = ShearX(y0=0.0)
        params = np.array([0.0])
        H = s.prepare_matrix(params)
        assert H.shape == (1, 3, 3)
        np.testing.assert_allclose(H[0], np.eye(3), atol=1e-10)

    def test_prepare_matrix_shape(self):
        s = ShearX(y0=5.0)
        params = np.linspace(0, 0.1, 5)
        H = s.prepare_matrix(params)
        assert H.shape == (5, 3, 3)

    def test_gradient_shape(self):
        s = ShearX(y0=1.0)
        params = np.array([0.0, 0.05, 0.1])
        grad = s.gradient(1.5, 2.0, params)
        assert grad.shape == (2, 3)

    def test_gradient_zero_angle(self):
        s = ShearX(y0=0.0)
        grad = s.gradient(5.0, 2.0, np.array([0.0]))
        # dx/dphi_x = -(y - y0) * sec^2(0) = -2.0
        # dy/dphi_x = 0
        np.testing.assert_allclose(grad[0, 0], -2.0, atol=1e-10)
        np.testing.assert_allclose(grad[1, 0], 0.0, atol=1e-10)

    def test_gradient_non_zero_angle(self):
        s = ShearX(y0=2.0)
        angle = np.pi / 4
        # sec^2(pi/4) = 2
        grad = s.gradient(5.0, 5.0, np.array([angle]))
        # dx/dphi_x = -(5 - 2) * 2 = -6.0
        np.testing.assert_allclose(grad[0, 0], -6.0, atol=1e-10)
        np.testing.assert_allclose(grad[1, 0], 0.0, atol=1e-10)

    def test_get_max_grad_candidates_crosses_zero(self):
        s = ShearX(y0=2.0)
        interval = [-np.pi / 4, np.pi / 4]
        # Should yield 2 candidates: -np.pi/4, np.pi/4 (0 is a local minimum, not max absolute grad)
        cands = s.get_max_grad_candidates(5.0, 5.0, interval)
        assert cands.shape == (2, 2)
        # Expected grad for -pi/4 and pi/4 is sec^2(+-pi/4)*(-(5-2)) = -6
        np.testing.assert_allclose(cands[0, 0], -6.0, atol=1e-10)  # -np.pi/4
        np.testing.assert_allclose(cands[0, 1], -6.0, atol=1e-10)  # np.pi/4

    def test_get_max_grad_candidates_positive_only(self):
        s = ShearX(y0=2.0)
        interval = [np.pi / 6, np.pi / 4]
        # Should yield 2 candidates since 0 is not in interval
        cands = s.get_max_grad_candidates(5.0, 5.0, interval)
        assert cands.shape == (2, 2)
