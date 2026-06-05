"""Verify that prepare_matrix() implements the closed-form preimage equations
from the paper.

For each non-affine homography the paper gives an explicit formula for
(u0, v0) = T^{-1}(kappa) applied to pixel (u, v).  These tests check that the
H^{-1} matrix produces exactly those values, confirming that the matrix
coefficients are algebraically correct.

Equation references are to the supplementary material of the paper:
  Sec. 10.1 roll (Eq. 16), 10.2 pitch (Eq. 21), 10.3 yaw (Eq. 26),
  10.4 x-translation (Eq. 31), 10.5 y-translation (Eq. 36),
  10.6 z-translation (Eq. 40).
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
)

from .helpers import _preimage

# Tolerance for matrix-vs-formula comparisons: relative error is the right
# measure since pixel coordinates can be large (|rtol| ~ 1e-10 means ~10
# decimal digits of agreement).
_FORMULA_RTOL = 1e-10

# Tolerance for identity and invariant checks (e.g. kappa=0 → no change, or
# a coordinate that must be exactly preserved).  Absolute is appropriate here
# since the expected value can be zero.
_EXACT_ATOL = 1e-10


# ---------------------------------------------------------------------------
# HomographyYaw  (paper Eq. 26)
# ---------------------------------------------------------------------------
class TestHomographyYawPreimage:
    """
    u0(Δψ) = xc + f * [f sin(Δψ) + (u-xc) cos(Δψ)] / [f cos(Δψ) - (u-xc) sin(Δψ)]
    v0(Δψ) = yc + f * (v-yc)          / [f cos(Δψ) - (u-xc) sin(Δψ)]
    """

    F, XC, YC = 10.0, 5.0, 5.0

    @pytest.mark.parametrize(
        "u, v, psi",
        [
            (7.0, 8.0, 0.0),
            (7.0, 8.0, 0.1),
            (7.0, 8.0, -0.15),
            (3.0, 6.0, 0.2),
            (9.0, 2.0, -0.05),
        ],
    )
    def test_preimage_matches_paper_formula(self, u, v, psi):
        h = HomographyYaw(f=self.F, xc=self.XC, yc=self.YC)
        u0_mat, v0_mat = _preimage(h, u, v, psi)

        c, s = np.cos(psi), np.sin(psi)
        f, xc, yc = self.F, self.XC, self.YC
        denom = f * c - (u - xc) * s

        u0_formula = xc + f * (f * s + (u - xc) * c) / denom
        v0_formula = yc + f * (v - yc) / denom

        np.testing.assert_allclose(
            u0_mat,
            u0_formula,
            rtol=_FORMULA_RTOL,
            err_msg=f"Yaw u0 mismatch at u={u},v={v},psi={psi}",
        )
        np.testing.assert_allclose(
            v0_mat,
            v0_formula,
            rtol=_FORMULA_RTOL,
            err_msg=f"Yaw v0 mismatch at u={u},v={v},psi={psi}",
        )

    def test_identity_at_zero_yaw(self):
        """At kappa=0 the preimage equals the input pixel."""
        h = HomographyYaw(f=self.F, xc=self.XC, yc=self.YC)
        for u, v in [(7.0, 8.0), (3.0, 2.0), (self.XC, self.YC)]:
            u0, v0 = _preimage(h, u, v, 0.0)
            np.testing.assert_allclose(u0, u, atol=_EXACT_ATOL)
            np.testing.assert_allclose(v0, v, atol=_EXACT_ATOL)


# ---------------------------------------------------------------------------
# HomographyPitch  (paper Eq. 21)
# ---------------------------------------------------------------------------
class TestHomographyPitchPreimage:
    """
    u0(Δθ) = xc + f * (u-xc) / [f cos(Δθ) + (v-yc) sin(Δθ)]
    v0(Δθ) = yc - f * [f sin(Δθ) - (v-yc) cos(Δθ)] / [f cos(Δθ) + (v-yc) sin(Δθ)]
    """

    F, XC, YC = 10.0, 5.0, 5.0

    @pytest.mark.parametrize(
        "u, v, theta",
        [
            (3.0, 8.0, 0.0),
            (3.0, 8.0, 0.1),
            (3.0, 8.0, -0.15),
            (7.0, 3.0, 0.2),
        ],
    )
    def test_preimage_matches_paper_formula(self, u, v, theta):
        h = HomographyPitch(f=self.F, xc=self.XC, yc=self.YC)
        u0_mat, v0_mat = _preimage(h, u, v, theta)

        c, s = np.cos(theta), np.sin(theta)
        f, xc, yc = self.F, self.XC, self.YC
        denom = f * c + (v - yc) * s

        u0_formula = xc + f * (u - xc) / denom
        v0_formula = yc - f * (f * s - (v - yc) * c) / denom

        np.testing.assert_allclose(
            u0_mat,
            u0_formula,
            rtol=_FORMULA_RTOL,
            err_msg=f"Pitch u0 mismatch at u={u},v={v},theta={theta}",
        )
        np.testing.assert_allclose(
            v0_mat,
            v0_formula,
            rtol=_FORMULA_RTOL,
            err_msg=f"Pitch v0 mismatch at u={u},v={v},theta={theta}",
        )

    def test_identity_at_zero_pitch(self):
        h = HomographyPitch(f=self.F, xc=self.XC, yc=self.YC)
        for u, v in [(3.0, 8.0), (7.0, 3.0)]:
            u0, v0 = _preimage(h, u, v, 0.0)
            np.testing.assert_allclose(u0, u, atol=_EXACT_ATOL)
            np.testing.assert_allclose(v0, v, atol=_EXACT_ATOL)


# ---------------------------------------------------------------------------
# HomographyRoll  (paper Eq. 15/16)
# Roll is an affine rotation.  The matrix H^{-1} is verified against the
# explicit rotation formula:
#   u0 = (u-xc)*cos(phi) - (v-yc)*sin(phi) + xc
#   v0 = (u-xc)*sin(phi) + (v-yc)*cos(phi) + yc
# ---------------------------------------------------------------------------
class TestHomographyRollPreimage:
    XC, YC = 5.0, 5.0

    @pytest.mark.parametrize(
        "u, v, phi",
        [
            (3.0, 7.0, 0.0),
            (3.0, 7.0, 0.1),
            (3.0, 7.0, -0.2),
            (8.0, 2.0, np.pi / 4),
        ],
    )
    def test_preimage_matches_rotation_formula(self, u, v, phi):
        h = HomographyRoll(xc=self.XC, yc=self.YC)
        u0_mat, v0_mat = _preimage(h, u, v, phi)

        c, s = np.cos(phi), np.sin(phi)
        xc, yc = self.XC, self.YC
        u0_formula = (u - xc) * c - (v - yc) * s + xc
        v0_formula = (u - xc) * s + (v - yc) * c + yc

        np.testing.assert_allclose(
            u0_mat,
            u0_formula,
            rtol=_FORMULA_RTOL,
            err_msg=f"Roll u0 mismatch at u={u},v={v},phi={phi}",
        )
        np.testing.assert_allclose(
            v0_mat,
            v0_formula,
            rtol=_FORMULA_RTOL,
            err_msg=f"Roll v0 mismatch at u={u},v={v},phi={phi}",
        )

    def test_identity_at_zero_roll(self):
        h = HomographyRoll(xc=self.XC, yc=self.YC)
        for u, v in [(3.0, 7.0), (8.0, 2.0)]:
            u0, v0 = _preimage(h, u, v, 0.0)
            np.testing.assert_allclose(u0, u, atol=_EXACT_ATOL)
            np.testing.assert_allclose(v0, v, atol=_EXACT_ATOL)


# ---------------------------------------------------------------------------
# HomographyX  (paper Eq. 31)
# ---------------------------------------------------------------------------
class TestHomographyXPreimage:
    """
    u0(Δx) = [Δx*(v-yc)*xc - f*z*u] / [Δx*(v-yc) - f*z]
    v0(Δx) = [Δx*(v-yc)*yc - f*z*v] / [Δx*(v-yc) - f*z]
    """

    F, XC, YC, Z = 10.0, 0.0, 0.0, -10.0

    @pytest.mark.parametrize(
        "u, v, dx",
        [
            (3.0, 4.0, 0.0),
            (3.0, 4.0, 0.5),
            (3.0, 4.0, 1.0),
            (6.0, 7.0, 0.3),
        ],
    )
    def test_preimage_matches_paper_formula(self, u, v, dx):
        h = HomographyX(f=self.F, xc=self.XC, yc=self.YC, z=self.Z)
        u0_mat, v0_mat = _preimage(h, u, v, dx)

        f, xc, yc, z = self.F, self.XC, self.YC, self.Z
        denom = dx * (v - yc) - f * z

        u0_formula = (dx * (v - yc) * xc - f * z * u) / denom
        v0_formula = (dx * (v - yc) * yc - f * z * v) / denom

        np.testing.assert_allclose(
            u0_mat,
            u0_formula,
            rtol=_FORMULA_RTOL,
            err_msg=f"HomographyX u0 mismatch at u={u},v={v},dx={dx}",
        )
        np.testing.assert_allclose(
            v0_mat,
            v0_formula,
            rtol=_FORMULA_RTOL,
            err_msg=f"HomographyX v0 mismatch at u={u},v={v},dx={dx}",
        )

    def test_identity_at_zero_x_translation(self):
        h = HomographyX(f=self.F, xc=self.XC, yc=self.YC, z=self.Z)
        for u, v in [(3.0, 4.0), (7.0, 6.0)]:
            u0, v0 = _preimage(h, u, v, 0.0)
            np.testing.assert_allclose(u0, u, atol=_EXACT_ATOL)
            np.testing.assert_allclose(v0, v, atol=_EXACT_ATOL)


# ---------------------------------------------------------------------------
# HomographyY  (paper Eq. 36)
# ---------------------------------------------------------------------------
class TestHomographyYPreimage:
    """
    u0(Δy) = u + Δy*(yc/f - v/z)
    v0(Δy) = v
    """

    F, XC, YC, Z = 10.0, 0.0, 0.0, -10.0

    @pytest.mark.parametrize(
        "u, v, dy",
        [
            (3.0, 4.0, 0.0),
            (3.0, 4.0, 0.5),
            (3.0, 4.0, -0.7),
            (7.0, 6.0, 1.0),
        ],
    )
    def test_preimage_matches_paper_formula(self, u, v, dy):
        h = HomographyY(f=self.F, xc=self.XC, yc=self.YC, z=self.Z)
        u0_mat, v0_mat = _preimage(h, u, v, dy)

        f, yc, z = self.F, self.YC, self.Z
        u0_formula = u + dy * (yc / f - v / z)
        v0_formula = v

        np.testing.assert_allclose(
            u0_mat,
            u0_formula,
            rtol=_FORMULA_RTOL,
            err_msg=f"HomographyY u0 mismatch at u={u},v={v},dy={dy}",
        )
        np.testing.assert_allclose(
            v0_mat,
            v0_formula,
            rtol=_FORMULA_RTOL,
            err_msg=f"HomographyY v0 mismatch at u={u},v={v},dy={dy}",
        )

    def test_v_coordinate_unchanged(self):
        """Y-translation does not affect the v coordinate."""
        h = HomographyY(f=self.F, xc=self.XC, yc=self.YC, z=self.Z)
        for u, v, dy in [(3.0, 4.0, 0.5), (7.0, 6.0, -1.0)]:
            _, v0 = _preimage(h, u, v, dy)
            np.testing.assert_allclose(
                v0, v, atol=_EXACT_ATOL, err_msg="HomographyY must leave v unchanged"
            )


# ---------------------------------------------------------------------------
# HomographyZ  (paper Eq. 40)
# ---------------------------------------------------------------------------
class TestHomographyZPreimage:
    """
    u0(Δz) = u
    v0(Δz) = (z*v + Δz*yc) / (z + Δz)
    """

    F, XC, YC, Z = 10.0, 0.0, 0.0, -10.0

    @pytest.mark.parametrize(
        "u, v, dz",
        [
            (3.0, 4.0, 0.0),
            (3.0, 4.0, 1.0),
            (3.0, 4.0, 3.0),
            (7.0, 6.0, 0.5),
        ],
    )
    def test_preimage_matches_paper_formula(self, u, v, dz):
        h = HomographyZ(f=self.F, xc=self.XC, yc=self.YC, z=self.Z)
        u0_mat, v0_mat = _preimage(h, u, v, dz)

        yc, z = self.YC, self.Z
        u0_formula = u
        v0_formula = (z * v + dz * yc) / (z + dz)

        np.testing.assert_allclose(
            u0_mat,
            u0_formula,
            rtol=_FORMULA_RTOL,
            err_msg=f"HomographyZ u0 mismatch at u={u},v={v},dz={dz}",
        )
        np.testing.assert_allclose(
            v0_mat,
            v0_formula,
            rtol=_FORMULA_RTOL,
            err_msg=f"HomographyZ v0 mismatch at u={u},v={v},dz={dz}",
        )

    def test_u_coordinate_unchanged(self):
        """Z-translation (zoom) must not move pixels horizontally."""
        h = HomographyZ(f=self.F, xc=self.XC, yc=self.YC, z=self.Z)
        for u, v, dz in [(3.0, 4.0, 1.0), (7.0, 6.0, 3.0)]:
            u0, _ = _preimage(h, u, v, dz)
            np.testing.assert_allclose(
                u0, u, atol=_EXACT_ATOL, err_msg="HomographyZ must leave u unchanged"
            )


# ---------------------------------------------------------------------------
# Rotation  (affine, about a center)
# H^{-1} = [[c, s, -x0*c+x0-y0*s],
#            [-s, c, x0*s-y0*c+y0],
#            [0,  0, 1]]
# This gives:
#   u0 = c*(u-x0) + s*(v-y0) + x0
#   v0 = -s*(u-x0) + c*(v-y0) + y0
# ---------------------------------------------------------------------------
class TestRotationPreimage:
    X0, Y0 = 3.0, 4.0

    @pytest.mark.parametrize(
        "u, v, theta",
        [
            (5.0, 7.0, 0.0),
            (5.0, 7.0, np.pi / 6),  # 30 deg — sin != cos, shows any formula error
            (5.0, 7.0, -0.4),
            (1.0, 2.0, 0.8),
        ],
    )
    def test_preimage_matches_affine_rotation_formula(self, u, v, theta):
        r = Rotation(x0=self.X0, y0=self.Y0)
        u0_mat, v0_mat = _preimage(r, u, v, theta)

        c, s = np.cos(theta), np.sin(theta)
        x0, y0 = self.X0, self.Y0
        u0_formula = c * (u - x0) + s * (v - y0) + x0
        v0_formula = -s * (u - x0) + c * (v - y0) + y0

        np.testing.assert_allclose(
            u0_mat,
            u0_formula,
            rtol=_FORMULA_RTOL,
            err_msg=f"Rotation u0 mismatch at u={u},v={v},theta={theta}",
        )
        np.testing.assert_allclose(
            v0_mat,
            v0_formula,
            rtol=_FORMULA_RTOL,
            err_msg=f"Rotation v0 mismatch at u={u},v={v},theta={theta}",
        )

    def test_rotation_center_fixed(self):
        """The rotation center must map to itself under any rotation."""
        r = Rotation(x0=self.X0, y0=self.Y0)
        for theta in [0.0, 0.3, np.pi / 4, 1.2]:
            u0, v0 = _preimage(r, self.X0, self.Y0, theta)
            np.testing.assert_allclose(
                u0,
                self.X0,
                atol=_EXACT_ATOL,
                err_msg=f"Rotation center not fixed at theta={theta}",
            )
            np.testing.assert_allclose(
                v0,
                self.Y0,
                atol=_EXACT_ATOL,
                err_msg=f"Rotation center not fixed at theta={theta}",
            )
