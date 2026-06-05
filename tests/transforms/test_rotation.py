"""Tests for Rotation transform class."""

import numpy as np

from vnn.transforms.rotation import Rotation


class TestRotation:
    def test_init(self):
        r = Rotation(x0=1.0, y0=2.0)
        assert r.x0 == 1.0
        assert r.y0 == 2.0

    def test_init_default(self):
        r = Rotation()
        assert r.x0 == 0
        assert r.y0 == 0

    def test_str(self):
        r = Rotation(x0=1.0, y0=2.0)
        assert str(r) == "Rotation(1.0,2.0)"

    def test_prepare_matrix_identity(self):
        r = Rotation(x0=0.0, y0=0.0)
        params = np.array([0.0])
        H = r.prepare_matrix(params)
        assert H.shape == (1, 3, 3)
        np.testing.assert_allclose(H[0], np.eye(3), atol=1e-10)

    def test_prepare_matrix_shape(self):
        r = Rotation(x0=5.0, y0=5.0)
        params = np.linspace(0, 0.1, 5)
        H = r.prepare_matrix(params)
        assert H.shape == (5, 3, 3)

    def test_gradient_shape(self):
        r = Rotation(x0=1.0, y0=1.0)
        params = np.array([0.0, 0.05, 0.1])
        grad = r.gradient(1.5, 2.0, params)
        assert grad.shape == (2, 3)

    def test_gradient_zero_angle(self):
        r = Rotation(x0=0.0, y0=0.0)
        grad = r.gradient(1.0, 0.0, np.array([0.0]))
        # dx/dtheta = -x*sin(0) + y*cos(0) = 0
        # dy/dtheta = -x*cos(0) - y*cos(0) = -1
        np.testing.assert_allclose(grad[0, 0], 0.0, atol=1e-10)
        np.testing.assert_allclose(grad[1, 0], -1.0, atol=1e-10)

    def test_get_max_grad_candidates_inside(self):
        r = Rotation(x0=0.0, y0=0.0)
        grad = r.get_max_grad_candidates(1.0, 1.0, interval=(0.0, 2.0))
        assert grad.shape[0] == 2

    def test_get_max_grad_candidates_outside(self):
        r = Rotation(x0=0.0, y0=0.0)
        # Use a narrow interval where the critical angles are outside
        grad = r.get_max_grad_candidates(1.0, 1.0, interval=(0.0, 0.01))
        assert grad.shape[0] == 2

    def test_get_max_grad_candidates_candidate_modulo(self):
        r = Rotation(x0=0.0, y0=0.0)
        # angle_1 = arctan(-1/1) = -pi/4, angle_2 = arctan(1/1) = pi/4
        # Interval around angle_1 + pi = 3*pi/4: shifted candidate should be included
        angle_1 = np.arctan(-1.0)
        grad = r.get_max_grad_candidates(
            1.0, 1.0, interval=(angle_1 + np.pi - 0.1, angle_1 + np.pi + 0.1)
        )
        # 2 boundary candidates + 1 shifted candidate = 3 columns
        assert grad.shape == (2, 3)
