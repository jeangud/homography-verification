"""Tests for the TranslateY transform class."""

import numpy as np

from vnn.transforms.translate_y import TranslateY


class TestTranslateY:
    def test_init(self):
        t = TranslateY()
        assert t is not None

    def test_str(self):
        t = TranslateY()
        assert str(t) == "TranslateY()"

    def test_prepare_matrix_identity(self):
        t = TranslateY()
        params = np.array([0.0])
        H = t.prepare_matrix(params)
        assert H.shape == (1, 3, 3)
        np.testing.assert_allclose(H[0], np.eye(3), atol=1e-10)

    def test_prepare_matrix_translation(self):
        t = TranslateY()
        params = np.array([2.0])
        H = t.prepare_matrix(params)
        expected = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, -2.0], [0.0, 0.0, 1.0]])
        np.testing.assert_allclose(H[0], expected, atol=1e-10)

    def test_prepare_matrix_shape(self):
        t = TranslateY()
        params = np.array([1.0, 1.5, 2.0, 2.5])
        H = t.prepare_matrix(params)
        assert H.shape == (4, 3, 3)

    def test_gradient(self):
        t = TranslateY()
        grad = t.gradient(3.0, 6.0, np.array([2.0]))
        expected = np.array([[0.0], [-1.0]])
        np.testing.assert_allclose(grad, expected, atol=1e-10)

    def test_get_max_grad_candidates(self):
        t = TranslateY()
        interval = (2.0, 4.0)
        cands = t.get_max_grad_candidates(3.0, 6.0, interval)
        assert cands.shape == (2, 1)
        expected = np.array([[0.0], [-1.0]])
        np.testing.assert_allclose(cands, expected, atol=1e-10)
