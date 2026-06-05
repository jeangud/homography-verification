"""Tests for the TranslateX transform class."""

import numpy as np

from vnn.transforms.translate_x import TranslateX


class TestTranslateX:
    def test_init(self):
        t = TranslateX()
        assert t is not None

    def test_str(self):
        t = TranslateX()
        assert str(t) == "TranslateX()"

    def test_prepare_matrix_identity(self):
        t = TranslateX()
        params = np.array([0.0])
        H = t.prepare_matrix(params)
        assert H.shape == (1, 3, 3)
        np.testing.assert_allclose(H[0], np.eye(3), atol=1e-10)

    def test_prepare_matrix_translation(self):
        t = TranslateX()
        params = np.array([2.0])
        H = t.prepare_matrix(params)
        expected = np.array([[1.0, 0.0, -2.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        np.testing.assert_allclose(H[0], expected, atol=1e-10)

    def test_prepare_matrix_shape(self):
        t = TranslateX()
        params = np.array([1.0, 1.5, 2.0, 2.5])
        H = t.prepare_matrix(params)
        assert H.shape == (4, 3, 3)

    def test_gradient(self):
        t = TranslateX()
        grad = t.gradient(3.0, 6.0, np.array([2.0]))
        expected = np.array([[-1.0], [0.0]])
        np.testing.assert_allclose(grad, expected, atol=1e-10)

    def test_get_max_grad_candidates(self):
        t = TranslateX()
        interval = (2.0, 4.0)
        cands = t.get_max_grad_candidates(3.0, 6.0, interval)
        assert cands.shape == (2, 1)
        expected = np.array([[-1.0], [0.0]])
        np.testing.assert_allclose(cands, expected, atol=1e-10)
