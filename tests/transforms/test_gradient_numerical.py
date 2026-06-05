"""Numerical gradient validation for all transforms.

For every transform, the analytical gradient d(T^{-1})/d(kappa) is compared
against a central finite-difference estimate.  A mismatch here means the closed-
form gradient formula is wrong, which would propagate directly into wrong Lipschitz
constants and therefore unsound bounds.
"""

import numpy as np
import pytest

from vnn.transforms import (
    HomographyPitch,
    HomographyRoll,
    HomographyX,
    HomographyY,
    HomographyYaw,
    HomographyZ,
    Rotation,
    Scale,
    ShearX,
    TranslateX,
    TranslateY,
)

from .helpers import _preimage

_GRAD_RTOL = 1e-4
_GRAD_ATOL = 1e-7

# Tolerance for exact structural checks (gradient is zero/constant),
# only floating-point noise is expected.
_EXACT_ATOL = 1e-12


def _numerical_gradient(transform, x, y, kappa, delta=1e-6):
    """Central finite-difference estimate of d(T^{-1})/d(kappa) at (x, y, kappa).

    Returns shape (2,): [dx0/dkappa, dy0/dkappa].
    """
    x0p, y0p = _preimage(transform, x, y, kappa + delta)
    x0m, y0m = _preimage(transform, x, y, kappa - delta)
    return np.array([(x0p - x0m) / (2 * delta), (y0p - y0m) / (2 * delta)])


# ---------------------------------------------------------------------------
# Rotation
# ---------------------------------------------------------------------------
class TestRotationGradientNumerical:
    """The rotation gradient has a subtle closed-form: the second component is
    -(x-x0)*cos - (y-y0)*sin.  With `sin=cos` (at pi/4) or `y=y0` the wrong
    formula -(y-y0)*cos also gives the right answer — those edge cases hide the
    bug. These tests use generic (x, y, kappa) to avoid such coincidences."""

    @pytest.mark.parametrize(
        "x, y, kappa",
        [
            (2.0, 3.0, 0.0),
            (2.0, 3.0, 0.3),
            (2.0, 3.0, -0.5),
            (1.0, 4.0, np.pi / 6),  # sin != cos, exposes wrong-coefficient bug
            (5.0, 1.0, 1.1),
        ],
    )
    def test_rotation_gradient(self, x, y, kappa):
        r = Rotation(x0=1.0, y0=2.0)
        analytical = r.gradient(x, y, np.array([kappa]))[:, 0]
        numerical = _numerical_gradient(r, x, y, kappa)
        np.testing.assert_allclose(
            analytical,
            numerical,
            rtol=_GRAD_RTOL,
            atol=_GRAD_ATOL,
        )


# ---------------------------------------------------------------------------
# HomographyRoll
# ---------------------------------------------------------------------------
class TestHomographyRollGradientNumerical:
    @pytest.mark.parametrize(
        "x, y, kappa",
        [
            (3.0, 7.0, 0.0),
            (3.0, 7.0, 0.1),
            (3.0, 7.0, -0.2),
            (8.0, 2.0, 0.5),
            (1.0, 9.0, np.pi / 6),
        ],
    )
    def test_homography_roll_gradient(self, x, y, kappa):
        h = HomographyRoll(xc=5.0, yc=5.0)
        analytical = h.gradient(x, y, np.array([kappa]))[:, 0]
        numerical = _numerical_gradient(h, x, y, kappa)
        np.testing.assert_allclose(
            analytical,
            numerical,
            rtol=_GRAD_RTOL,
            atol=_GRAD_ATOL,
        )


# ---------------------------------------------------------------------------
# HomographyPitch
# ---------------------------------------------------------------------------
class TestHomographyPitchGradientNumerical:
    """Avoid the critical angle phi_c = arctan(-f/(y-yc)).
    For f=10, yc=5: with y=8, phi_c = arctan(-10/3) ≈ -1.28 rad.
    Test angles are in (-0.5, 0.5) which is safely far from -1.28."""

    @pytest.mark.parametrize(
        "x, y, kappa",
        [
            (3.0, 8.0, 0.0),
            (3.0, 8.0, 0.1),
            (3.0, 8.0, -0.15),
            (7.0, 3.0, 0.2),
            (2.0, 9.0, -0.3),
        ],
    )
    def test_homography_pitch_gradient(self, x, y, kappa):
        h = HomographyPitch(f=10.0, xc=5.0, yc=5.0)
        analytical = h.gradient(x, y, np.array([kappa]))[:, 0]
        numerical = _numerical_gradient(h, x, y, kappa)
        np.testing.assert_allclose(
            analytical,
            numerical,
            rtol=_GRAD_RTOL,
            atol=_GRAD_ATOL,
        )


# ---------------------------------------------------------------------------
# HomographyYaw
# ---------------------------------------------------------------------------
class TestHomographyYawGradientNumerical:
    """Avoid the critical angle psi_c = arctan(f/(x-xc)).
    For f=10, xc=5: with x=7, psi_c = arctan(10/2) ≈ 1.37 rad.
    Test angles are in (-0.3, 0.3) which is well within the safe range."""

    @pytest.mark.parametrize(
        "x, y, kappa",
        [
            (7.0, 8.0, 0.0),
            (7.0, 8.0, 0.1),
            (7.0, 8.0, -0.15),
            (3.0, 6.0, 0.2),
            (9.0, 2.0, -0.05),
        ],
    )
    def test_homography_yaw_gradient(self, x, y, kappa):
        h = HomographyYaw(f=10.0, xc=5.0, yc=5.0)
        analytical = h.gradient(x, y, np.array([kappa]))[:, 0]
        numerical = _numerical_gradient(h, x, y, kappa)
        np.testing.assert_allclose(
            analytical,
            numerical,
            rtol=_GRAD_RTOL,
            atol=_GRAD_ATOL,
        )


# ---------------------------------------------------------------------------
# HomographyX
# ---------------------------------------------------------------------------
class TestHomographyXGradientNumerical:
    """Critical value: dx_c = f*z/(y-yc) = 10*(-10)/(4-0) = -25.
    Test translations dx in (0, 2) are safely far from -25."""

    @pytest.mark.parametrize(
        "x, y, kappa",
        [
            (3.0, 4.0, 0.0),
            (3.0, 4.0, 0.5),
            (3.0, 4.0, 1.0),
            (6.0, 7.0, 0.3),
            (1.0, 2.0, 0.8),
        ],
    )
    def test_homography_x_gradient(self, x, y, kappa):
        h = HomographyX(f=10.0, xc=0.0, yc=0.0, z=-10.0)
        analytical = h.gradient(x, y, np.array([kappa]))[:, 0]
        numerical = _numerical_gradient(h, x, y, kappa)
        np.testing.assert_allclose(
            analytical,
            numerical,
            rtol=_GRAD_RTOL,
            atol=_GRAD_ATOL,
        )


# ---------------------------------------------------------------------------
# HomographyY
# ---------------------------------------------------------------------------
class TestHomographyYGradientNumerical:
    """Y-translation has constant gradient w.r.t. kappa (the dy parameter)."""

    @pytest.mark.parametrize(
        "x, y, kappa",
        [
            (3.0, 4.0, 0.0),
            (3.0, 4.0, 0.5),
            (3.0, 4.0, 1.0),
            (7.0, 6.0, -0.5),
        ],
    )
    def test_homography_y_gradient(self, x, y, kappa):
        h = HomographyY(f=10.0, xc=0.0, yc=0.0, z=-10.0)
        analytical = h.gradient(x, y, np.array([kappa]))[:, 0]
        numerical = _numerical_gradient(h, x, y, kappa)
        np.testing.assert_allclose(
            analytical,
            numerical,
            rtol=_GRAD_RTOL,
            atol=_GRAD_ATOL,
        )

    def test_gradient_is_constant_wrt_kappa(self):
        """The Y-translation gradient must be identical for all kappa values."""
        h = HomographyY(f=10.0, xc=0.0, yc=0.0, z=-10.0)
        x, y = 4.0, 7.0
        kappas = np.array([0.0, 0.3, 0.6, 1.0])
        grad = h.gradient(x, y, kappas)  # (2, 4)
        # All columns should be identical
        np.testing.assert_allclose(grad[:, 0], grad[:, 1], atol=_EXACT_ATOL)
        np.testing.assert_allclose(grad[:, 0], grad[:, 2], atol=_EXACT_ATOL)
        np.testing.assert_allclose(grad[:, 0], grad[:, 3], atol=_EXACT_ATOL)


# ---------------------------------------------------------------------------
# HomographyZ
# ---------------------------------------------------------------------------
class TestHomographyZGradientNumerical:
    """Critical value: dz_c = -z = 10.  Test range (0, 5) is safely below."""

    @pytest.mark.parametrize(
        "x, y, kappa",
        [
            (3.0, 4.0, 0.0),
            (3.0, 4.0, 1.0),
            (3.0, 4.0, 3.0),
            (7.0, 6.0, 0.5),
            (1.0, 2.0, 4.0),
        ],
    )
    def test_homography_z_gradient(self, x, y, kappa):
        h = HomographyZ(f=10.0, xc=0.0, yc=0.0, z=-10.0)
        analytical = h.gradient(x, y, np.array([kappa]))[:, 0]
        numerical = _numerical_gradient(h, x, y, kappa)
        np.testing.assert_allclose(
            analytical,
            numerical,
            rtol=_GRAD_RTOL,
            atol=_GRAD_ATOL,
        )

    def test_x_component_is_zero(self):
        """Z-translation does not move pixels horizontally."""
        h = HomographyZ(f=10.0, xc=0.0, yc=0.0, z=-10.0)
        grad = h.gradient(3.0, 7.0, np.array([0.0, 0.5, 1.0]))
        np.testing.assert_allclose(
            grad[0, :],
            0.0,
            atol=_EXACT_ATOL,
            err_msg="HomographyZ x-gradient should be zero",
        )


# ---------------------------------------------------------------------------
# TranslateX
# ---------------------------------------------------------------------------
class TestTranslateXGradientNumerical:
    @pytest.mark.parametrize("x, y, kappa", [(2.0, 3.0, 0.0), (5.0, 8.0, 1.5)])
    def test_translate_x_gradient(self, x, y, kappa):
        t = TranslateX()
        analytical = t.gradient(x, y, np.array([kappa]))[:, 0]
        numerical = _numerical_gradient(t, x, y, kappa)
        np.testing.assert_allclose(
            analytical, numerical, rtol=_GRAD_RTOL, atol=_GRAD_ATOL
        )

    def test_translate_x_gradient_is_minus_one_zero(self):
        """d(T^{-1})/d(dx) = [-1, 0] everywhere (pure x shift)."""
        t = TranslateX()
        grad = t.gradient(3.0, 5.0, np.array([0.0, 0.5, 2.0]))  # (2, 3)
        np.testing.assert_allclose(grad[0, :], -1.0, atol=_EXACT_ATOL)
        np.testing.assert_allclose(grad[1, :], 0.0, atol=_EXACT_ATOL)


# ---------------------------------------------------------------------------
# TranslateY
# ---------------------------------------------------------------------------
class TestTranslateYGradientNumerical:
    @pytest.mark.parametrize("x, y, kappa", [(2.0, 3.0, 0.0), (5.0, 8.0, 1.5)])
    def test_translate_y_gradient(self, x, y, kappa):
        t = TranslateY()
        analytical = t.gradient(x, y, np.array([kappa]))[:, 0]
        numerical = _numerical_gradient(t, x, y, kappa)
        np.testing.assert_allclose(
            analytical, numerical, rtol=_GRAD_RTOL, atol=_GRAD_ATOL
        )

    def test_translate_y_gradient_is_zero_minus_one(self):
        """d(T^{-1})/d(dy) = [0, -1] everywhere (pure y shift)."""
        t = TranslateY()
        grad = t.gradient(3.0, 5.0, np.array([0.0, 0.5, 2.0]))  # (2, 3)
        np.testing.assert_allclose(grad[0, :], 0.0, atol=_EXACT_ATOL)
        np.testing.assert_allclose(grad[1, :], -1.0, atol=_EXACT_ATOL)


# ---------------------------------------------------------------------------
# Scale
# ---------------------------------------------------------------------------
class TestScaleGradientNumerical:
    @pytest.mark.parametrize(
        "x, y, kappa",
        [
            (3.0, 4.0, 0.5),
            (3.0, 4.0, 1.0),
            (3.0, 4.0, 2.0),
            (7.0, 6.0, 1.5),
        ],
    )
    def test_scale_gradient(self, x, y, kappa):
        s = Scale(x0=2.0, y0=3.0)
        analytical = s.gradient(x, y, np.array([kappa]))[:, 0]
        numerical = _numerical_gradient(s, x, y, kappa)
        np.testing.assert_allclose(
            analytical,
            numerical,
            rtol=_GRAD_RTOL,
            atol=_GRAD_ATOL,
        )


# ---------------------------------------------------------------------------
# ShearX
# ---------------------------------------------------------------------------
class TestShearXGradientNumerical:
    @pytest.mark.parametrize(
        "x, y, kappa",
        [
            (3.0, 4.0, 0.0),
            (3.0, 4.0, 0.2),
            (3.0, 4.0, -0.3),
            (7.0, 6.0, 0.1),
        ],
    )
    def test_shear_x_gradient(self, x, y, kappa):
        sh = ShearX(y0=2.0)
        analytical = sh.gradient(x, y, np.array([kappa]))[:, 0]
        numerical = _numerical_gradient(sh, x, y, kappa)
        np.testing.assert_allclose(
            analytical,
            numerical,
            rtol=_GRAD_RTOL,
            atol=_GRAD_ATOL,
        )

    def test_shear_x_y_component_is_zero(self):
        """ShearX only moves pixels horizontally: dy/dkappa = 0."""
        sh = ShearX(y0=2.0)
        grad = sh.gradient(3.0, 5.0, np.array([0.0, 0.1, 0.3]))
        np.testing.assert_allclose(
            grad[1, :],
            0.0,
            atol=_EXACT_ATOL,
            err_msg="ShearX y-gradient should be zero",
        )
