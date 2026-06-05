"""Verify that analytical Lipschitz constants are valid upper bounds.

get_lipschitz_analytical() must return a value >= get_lipschitz_numerical().
If analytical < numerical the Lipschitz bound used in branch-and-bound is too
small, meaning the algorithm may stop early and produce an unsound bound.

The numerical estimate is computed by sampling the gradient at 100 evenly-spaced
points inside the interval — it is a lower bound on the true supremum, not the
true value.  The test therefore checks:

    analytical >= numerical - tolerance

where tolerance accommodates the fact that the numerical estimate may
occasionally overshoot the true gradient by floating-point errors.
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

# Fixed 8×8 single-channel image with values in [0, 1]
RNG = np.random.default_rng(42)
_IMG_8x8 = RNG.random((1, 8, 8)).astype(np.float32)

# Tolerance: the analytical bound should be within 1% of the numerical one
_TOL = 0.01


def _check_lipschitz(transform, img, x, y, interval):
    """Assert analytical Lipschitz >= numerical Lipschitz (up to _TOL)."""
    L_an = transform.get_lipschitz_analytical(img, x, y, interval)
    L_nu = transform.get_lipschitz_numerical(img, x, y, interval)
    assert L_an >= L_nu - _TOL, (
        f"{transform}: analytical Lipschitz {L_an:.6f} < numerical {L_nu:.6f} "
        f"at ({x},{y}) over {interval}"
    )


# ---------------------------------------------------------------------------
# HomographyYaw
# ---------------------------------------------------------------------------
class TestLipschitzHomographyYaw:
    H = HomographyYaw(f=10.0, xc=4.0, yc=4.0)

    @pytest.mark.parametrize(
        "x, y, interval",
        [
            (7.0, 6.0, (0.0, 0.05)),
            (7.0, 6.0, (0.0, 0.15)),
            (3.0, 5.0, (-0.1, 0.1)),
        ],
    )
    def test_lipschitz_is_upper_bound(self, x, y, interval):
        _check_lipschitz(self.H, _IMG_8x8, x, y, interval)

    def test_lipschitz_positive(self):
        """Lipschitz constant must be non-negative for any image."""
        L = self.H.get_lipschitz_analytical(_IMG_8x8, 6.0, 5.0, (0.0, 0.1))
        assert L >= 0.0


# ---------------------------------------------------------------------------
# HomographyPitch
# ---------------------------------------------------------------------------
class TestLipschitzHomographyPitch:
    H = HomographyPitch(f=10.0, xc=4.0, yc=4.0)

    @pytest.mark.parametrize(
        "x, y, interval",
        [
            (3.0, 7.0, (0.0, 0.05)),
            (3.0, 7.0, (0.0, 0.15)),
            (6.0, 2.0, (-0.1, 0.1)),
        ],
    )
    def test_lipschitz_is_upper_bound(self, x, y, interval):
        _check_lipschitz(self.H, _IMG_8x8, x, y, interval)


# ---------------------------------------------------------------------------
# HomographyRoll
# ---------------------------------------------------------------------------
class TestLipschitzHomographyRoll:
    H = HomographyRoll(xc=4.0, yc=4.0)

    @pytest.mark.parametrize(
        "x, y, interval",
        [
            (3.0, 7.0, (0.0, 0.1)),
            (6.0, 2.0, (-0.05, 0.15)),
        ],
    )
    def test_lipschitz_is_upper_bound(self, x, y, interval):
        _check_lipschitz(self.H, _IMG_8x8, x, y, interval)


# ---------------------------------------------------------------------------
# HomographyX
# ---------------------------------------------------------------------------
class TestLipschitzHomographyX:
    H = HomographyX(f=10.0, xc=0.0, yc=0.0, z=-10.0)

    @pytest.mark.parametrize(
        "x, y, interval",
        [
            (3.0, 5.0, (0.0, 0.5)),
            (6.0, 3.0, (0.0, 1.0)),
        ],
    )
    def test_lipschitz_is_upper_bound(self, x, y, interval):
        _check_lipschitz(self.H, _IMG_8x8, x, y, interval)


# ---------------------------------------------------------------------------
# HomographyY
# ---------------------------------------------------------------------------
class TestLipschitzHomographyY:
    H = HomographyY(f=10.0, xc=0.0, yc=0.0, z=-10.0)

    @pytest.mark.parametrize(
        "x, y, interval",
        [
            (3.0, 5.0, (0.0, 0.5)),
            (6.0, 3.0, (0.0, 1.0)),
        ],
    )
    def test_lipschitz_is_upper_bound(self, x, y, interval):
        _check_lipschitz(self.H, _IMG_8x8, x, y, interval)


# ---------------------------------------------------------------------------
# HomographyZ
# ---------------------------------------------------------------------------
class TestLipschitzHomographyZ:
    H = HomographyZ(f=10.0, xc=0.0, yc=0.0, z=-10.0)

    @pytest.mark.parametrize(
        "x, y, interval",
        [
            (3.0, 5.0, (0.0, 1.0)),
            (6.0, 3.0, (0.0, 3.0)),
        ],
    )
    def test_lipschitz_is_upper_bound(self, x, y, interval):
        _check_lipschitz(self.H, _IMG_8x8, x, y, interval)


# ---------------------------------------------------------------------------
# Rotation
# ---------------------------------------------------------------------------
class TestLipschitzRotation:
    H = Rotation(x0=4.0, y0=4.0)

    @pytest.mark.parametrize(
        "x, y, interval",
        [
            (6.0, 7.0, (0.0, 0.1)),
            (2.0, 5.0, (-0.05, 0.1)),
        ],
    )
    def test_lipschitz_is_upper_bound(self, x, y, interval):
        _check_lipschitz(self.H, _IMG_8x8, x, y, interval)


# ---------------------------------------------------------------------------
# Scale
# ---------------------------------------------------------------------------
class TestLipschitzScale:
    H = Scale(x0=4.0, y0=4.0)

    @pytest.mark.parametrize(
        "x, y, interval",
        [
            (6.0, 7.0, (0.5, 1.5)),
            (2.0, 3.0, (0.8, 2.0)),
        ],
    )
    def test_lipschitz_is_upper_bound(self, x, y, interval):
        _check_lipschitz(self.H, _IMG_8x8, x, y, interval)


# ---------------------------------------------------------------------------
# ShearX
# ---------------------------------------------------------------------------
class TestLipschitzShearX:
    H = ShearX(y0=4.0)

    @pytest.mark.parametrize(
        "x, y, interval",
        [
            (6.0, 7.0, (0.0, 0.2)),
            (2.0, 3.0, (-0.1, 0.3)),
        ],
    )
    def test_lipschitz_is_upper_bound(self, x, y, interval):
        _check_lipschitz(self.H, _IMG_8x8, x, y, interval)


# ---------------------------------------------------------------------------
# TranslateX / TranslateY
# ---------------------------------------------------------------------------
class TestLipschitzTranslations:
    def test_translate_x(self):
        _check_lipschitz(TranslateX(), _IMG_8x8, 3.0, 5.0, (0.0, 1.0))

    def test_translate_y(self):
        _check_lipschitz(TranslateY(), _IMG_8x8, 3.0, 5.0, (0.0, 1.0))
