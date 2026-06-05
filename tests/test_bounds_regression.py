"""Regression tests: re-run the bounds pipeline and compare against saved fixtures.

Fixture: tests/fixtures/bounds_yaw_regression.npz
  Generated from HomographyYaw(f=10, xc=2.5, yc=2.5) on a 5×5 image with
  yaw range [0, 5°], lipschitz_error=0.01, seed=7.

Purpose: catch any numerical regression when algorithmic changes are made,
or dependencies are updated. When the fixture values change intentionally,
update the fixture by re-running:

    python - <<'EOF'
    # paste the fixture generation script from tests/test_bounds_regression.py
    EOF

"""

from pathlib import Path

import numpy as np
import pytest

from vnn import pwl
from vnn.linear_programming import Solver
from vnn.transforms import HomographyYaw, TransformWithBounds

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "bounds_yaw_regression.npz"

_ATOL = 1e-5


@pytest.fixture(scope="module")
def fixture_data():
    """Load the pre-computed reference bounds from disk."""
    assert FIXTURE_PATH.exists(), (
        f"Fixture not found at {FIXTURE_PATH}.  "
        "Re-generate with the script at the top of this file."
    )
    return np.load(FIXTURE_PATH)


@pytest.fixture(scope="module")
def fresh_result(fixture_data):
    """Re-run calculate_bounds with the same inputs as the fixture."""
    img = fixture_data["img"]
    lower = float(fixture_data["lower_bound"][0])
    upper = float(fixture_data["upper_bound"][0])

    transform = HomographyYaw(f=10.0, xc=2.5, yc=2.5)
    tfwb = TransformWithBounds(transform, lower_bound=lower, upper_bound=upper)

    return pwl.calculate_bounds(
        img=img,
        transform_with_bounds=tfwb,
        padding=0.0,
        num_samples=30,
        lipschitz_error=0.01,
        num_init_splits=4,
        num_splits=2,
        num_subdomains=15,
        max_iterations=2000,
        solver=Solver.SCIPY,
    )


# ---------------------------------------------------------------------------
# Structure checks
# ---------------------------------------------------------------------------
class TestFixtureStructure:
    def test_fixture_contains_expected_keys(self, fixture_data):
        required_keys = [
            "img",
            "params",
            "samples",
            "linear_unsound_lower",
            "linear_unsound_upper",
            "linear_sound_lower",
            "linear_sound_upper",
            "pwl_unsound_lower",
            "pwl_unsound_upper",
            "pwl_sound_lower",
            "pwl_sound_upper",
            "lower_bound",
            "upper_bound",
        ]
        for key in required_keys:
            assert key in fixture_data, f"Missing key in fixture: {key}"

    def test_fixture_image_shape(self, fixture_data):
        assert fixture_data["img"].shape == (1, 5, 5)

    def test_fixture_linear_bound_shape(self, fixture_data):
        assert fixture_data["linear_sound_lower"].shape == (1, 5, 5, 2)
        assert fixture_data["linear_sound_upper"].shape == (1, 5, 5, 2)

    def test_fixture_pwl_bound_shape(self, fixture_data):
        assert fixture_data["pwl_sound_lower"].shape == (1, 5, 5, 2, 4)
        assert fixture_data["pwl_sound_upper"].shape == (1, 5, 5, 2, 4)


# ---------------------------------------------------------------------------
# Sample reproducibility
# ---------------------------------------------------------------------------
class TestSamplesRegression:
    def test_params_match(self, fresh_result, fixture_data):
        """Parameter grid must be identical (fully deterministic)."""
        np.testing.assert_array_equal(
            fresh_result["params"],
            fixture_data["params"],
            err_msg="Parameter array changed — linspace inputs must be identical",
        )

    def test_samples_match(self, fresh_result, fixture_data):
        """Pixel samples must be bit-identical (deterministic interpolation)."""
        np.testing.assert_allclose(
            fresh_result["samples"],
            fixture_data["samples"],
            rtol=0,
            atol=1e-6,
            err_msg="Sample pixel values changed — interpolation or matrix may have changed",
        )


# ---------------------------------------------------------------------------
# Unsound LP bounds regression (deterministic given scipy)
# ---------------------------------------------------------------------------
class TestUnsoundLinearBoundsRegression:
    def test_lower_slope_matches(self, fresh_result, fixture_data):
        np.testing.assert_allclose(
            fresh_result["linear"]["unsound"]["lower"][:, :, :, 0],
            fixture_data["linear_unsound_lower"][:, :, :, 0],
            atol=_ATOL,
            err_msg="Unsound lower bound SLOPE changed",
        )

    def test_lower_intercept_matches(self, fresh_result, fixture_data):
        np.testing.assert_allclose(
            fresh_result["linear"]["unsound"]["lower"][:, :, :, 1],
            fixture_data["linear_unsound_lower"][:, :, :, 1],
            atol=_ATOL,
            err_msg="Unsound lower bound INTERCEPT changed",
        )

    def test_upper_slope_matches(self, fresh_result, fixture_data):
        np.testing.assert_allclose(
            fresh_result["linear"]["unsound"]["upper"][:, :, :, 0],
            fixture_data["linear_unsound_upper"][:, :, :, 0],
            atol=_ATOL,
            err_msg="Unsound upper bound SLOPE changed",
        )

    def test_upper_intercept_matches(self, fresh_result, fixture_data):
        np.testing.assert_allclose(
            fresh_result["linear"]["unsound"]["upper"][:, :, :, 1],
            fixture_data["linear_unsound_upper"][:, :, :, 1],
            atol=_ATOL,
            err_msg="Unsound upper bound INTERCEPT changed",
        )


# ---------------------------------------------------------------------------
# Sound linear bounds regression
# ---------------------------------------------------------------------------
class TestSoundLinearBoundsRegression:
    """Sound bounds depend on branch-and-bound so we use a looser tolerance,
    but they must still be close to the saved values."""

    def test_sound_lower_matches(self, fresh_result, fixture_data):
        np.testing.assert_allclose(
            fresh_result["linear"]["sound"]["lower"],
            fixture_data["linear_sound_lower"],
            atol=_ATOL,
            err_msg="Sound lower linear bound changed — check Lipschitz or BaB logic",
        )

    def test_sound_upper_matches(self, fresh_result, fixture_data):
        np.testing.assert_allclose(
            fresh_result["linear"]["sound"]["upper"],
            fixture_data["linear_sound_upper"],
            atol=_ATOL,
            err_msg="Sound upper linear bound changed — check Lipschitz or BaB logic",
        )

    def test_sound_lower_leq_unsound_lower(self, fresh_result, fixture_data):
        """The sound lower bound must be <= the unsound lower bound at every point.
        Sound bounds are shifted DOWN to guarantee they are below all pixel values."""
        lb_sound = fresh_result["linear"]["sound"]["lower"]  # (c, h, w, 2)
        lb_unsound = fresh_result["linear"]["unsound"]["lower"]

        # Check at a few kappa values
        for kappa in np.linspace(0.0, float(fixture_data["upper_bound"][0]), 5):
            sound_vals = lb_sound[:, :, :, 0] * kappa + lb_sound[:, :, :, 1]
            unsound_vals = lb_unsound[:, :, :, 0] * kappa + lb_unsound[:, :, :, 1]
            assert np.all(sound_vals <= unsound_vals + _ATOL), (
                f"Sound LB > unsound LB at kappa={kappa:.4f} — soundness shift broken"
            )

    def test_sound_upper_geq_unsound_upper(self, fresh_result, fixture_data):
        """The sound upper bound must be >= the unsound upper bound."""
        ub_sound = fresh_result["linear"]["sound"]["upper"]
        ub_unsound = fresh_result["linear"]["unsound"]["upper"]

        for kappa in np.linspace(0.0, float(fixture_data["upper_bound"][0]), 5):
            sound_vals = ub_sound[:, :, :, 0] * kappa + ub_sound[:, :, :, 1]
            unsound_vals = ub_unsound[:, :, :, 0] * kappa + ub_unsound[:, :, :, 1]
            assert np.all(sound_vals >= unsound_vals - _ATOL), (
                f"Sound UB < unsound UB at kappa={kappa:.4f} — soundness shift broken"
            )


# ---------------------------------------------------------------------------
# Sound PWL bounds regression
# ---------------------------------------------------------------------------
class TestSoundPWLBoundsRegression:
    def test_pwl_sound_lower_matches(self, fresh_result, fixture_data):
        np.testing.assert_allclose(
            fresh_result["pwl"]["sound"]["lower"],
            fixture_data["pwl_sound_lower"],
            atol=_ATOL,
            err_msg="Sound PWL lower bound changed",
        )

    def test_pwl_sound_upper_matches(self, fresh_result, fixture_data):
        np.testing.assert_allclose(
            fresh_result["pwl"]["sound"]["upper"],
            fixture_data["pwl_sound_upper"],
            atol=_ATOL,
            err_msg="Sound PWL upper bound changed",
        )
