"""Tests for the Scale transform class."""

import numpy as np
import pytest

from vnn.transforms.scale import Scale


class TestScale:
    def test_init(self):
        s = Scale(x0=1.0, y0=2.0)
        assert s.x0 == 1.0
        assert s.y0 == 2.0

    def test_init_default(self):
        s = Scale()
        assert s.x0 == 0.0
        assert s.y0 == 0.0

    def test_str(self):
        s = Scale(x0=1.0, y0=2.0)
        assert str(s) == "Scale(1.0,2.0)"

    def test_prepare_matrix_identity(self):
        s = Scale(x0=0.0, y0=0.0)
        params = np.array([1.0])
        H = s.prepare_matrix(params)
        assert H.shape == (1, 3, 3)
        np.testing.assert_allclose(H[0], np.eye(3), atol=1e-10)

    def test_prepare_matrix_translation(self):
        s = Scale(x0=5.0, y0=5.0)
        params = np.array([2.0])
        H = s.prepare_matrix(params)
        # H[0, 0, 0] = 1/2
        # H[0, 0, 2] = 5 - 5/2 = 2.5
        expected = np.array([[0.5, 0.0, 2.5], [0.0, 0.5, 2.5], [0.0, 0.0, 1.0]])
        np.testing.assert_allclose(H[0], expected, atol=1e-10)

    def test_prepare_matrix_shape(self):
        s = Scale(x0=5.0, y0=5.0)
        params = np.array([1.0, 1.5, 2.0, 2.5])
        H = s.prepare_matrix(params)
        assert H.shape == (4, 3, 3)

    def test_prepare_matrix_negative_params(self):
        s = Scale()
        with pytest.raises(AssertionError, match="strictly positive"):
            s.prepare_matrix(np.array([0.0]))

    def test_gradient(self):
        s = Scale(x0=1.0, y0=2.0)
        # grad is np.array([ -(x - self.x0) / (s**2), -(y - self.y0) / (s**2) ])
        grad = s.gradient(3.0, 6.0, np.array([2.0]))
        # dx/ds = -(3-1) / 4 = -0.5
        # dy/ds = -(6-2) / 4 = -1.0
        expected = np.array([[-0.5], [-1.0]])
        np.testing.assert_allclose(grad, expected, atol=1e-10)

    def test_get_max_grad_candidates(self):
        s = Scale(x0=1.0, y0=2.0)
        interval = (2.0, 4.0)
        cands = s.get_max_grad_candidates(3.0, 6.0, interval)
        assert cands.shape == (2, 1)
        # Should be evaluated at s = 2.0
        expected = np.array([[-0.5], [-1.0]])
        np.testing.assert_allclose(cands, expected, atol=1e-10)

    def test_get_max_grad_candidates_negative_interval(self):
        s = Scale()
        with pytest.raises(AssertionError, match="strictly positive"):
            s.get_max_grad_candidates(3.0, 6.0, (-1.0, 2.0))
