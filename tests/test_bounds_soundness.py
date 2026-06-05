"""Verify that sound bounds actually contain all pixel values.

A "sound" lower bound LB(kappa) must satisfy LB(kappa) <= G(kappa) for every
kappa in [lower, upper], and similarly UB(kappa) >= G(kappa).

This is tested by:
1. Computing sound linear and PWL bounds with calculate_bounds().
2. Densely sampling the pixel curve G(kappa) at 1,000 points.
3. Asserting the inequality at every sample point.

Passing these tests proves end-to-end correctness: generate_samples →
fit_linear_bound → shift_linear_bound (and the PWL counterparts) together
produce an over-approximation of the true pixel trajectory.

All eleven transforms are covered. Parameter intervals are chosen to stay
well away from singularities and to exercise non-trivial pixel movement on
the shared 5×5 test image.
"""

import numpy as np
import pytest

from vnn import pwl
from vnn.linear_programming import Solver
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
    TransformWithBounds,
)

# Shared test image: 5×5 single-channel, fixed seed for reproducibility
_RNG = np.random.default_rng(0)
_IMG = _RNG.random((1, 5, 5)).astype(np.float32)

# Number of dense verification samples
_N_VERIFY = 1_000

# Tolerance: the bound may be slightly violated by floating-point rounding
# during interpolation, but must hold up to this margin
_ATOL = 1e-4

# (label, transform, lower, upper)
# Intervals are chosen to stay away from singularities.  For projective
# transforms the critical parameter is where the denominator of T^{-1} → 0;
# all chosen intervals are safely far from those values on a 5×5 image.
_CASES = [
    # Non-affine (projective) homographies — small angular perturbations
    ("HomographyYaw", HomographyYaw(f=10.0, xc=2.5, yc=2.5), 0.00, 0.10),
    ("HomographyPitch", HomographyPitch(f=10.0, xc=2.5, yc=2.5), 0.00, 0.10),
    ("HomographyRoll", HomographyRoll(xc=2.5, yc=2.5), 0.00, 0.10),
    # Translational homographies — pixel-scale shifts
    ("HomographyX", HomographyX(f=10.0, xc=0.0, yc=0.0, z=-10.0), 0.00, 0.50),
    ("HomographyY", HomographyY(f=10.0, xc=0.0, yc=0.0, z=-10.0), 0.00, 0.50),
    ("HomographyZ", HomographyZ(f=10.0, xc=0.0, yc=0.0, z=-10.0), 0.00, 1.00),
    # Affine transforms
    ("Rotation", Rotation(x0=2.5, y0=2.5), 0.00, 0.05),
    ("Scale", Scale(x0=2.5, y0=2.5), 0.80, 1.20),
    ("ShearX", ShearX(y0=2.5), 0.00, 0.10),
    ("TranslateX", TranslateX(), 0.00, 1.00),
    ("TranslateY", TranslateY(), 0.00, 1.00),
]


# ---------------------------------------------------------------------------
# Module-scoped fixture — computed once per transform, shared across all tests
# ---------------------------------------------------------------------------
@pytest.fixture(
    params=_CASES,
    ids=[c[0] for c in _CASES],
    scope="module",
)
def case(request):
    label, transform, lower, upper = request.param
    tfwb = TransformWithBounds(transform, lower_bound=lower, upper_bound=upper)
    result = pwl.calculate_bounds(
        img=_IMG,
        transform_with_bounds=tfwb,
        padding=0.0,
        num_samples=30,
        lipschitz_error=0.01,
        num_init_splits=4,
        num_splits=2,
        num_subdomains=15,
        max_iterations=2_000,
        solver=Solver.SCIPY,
    )
    samples, params = pwl.generate_samples(
        _IMG, transform, lower, upper, num_samples=_N_VERIFY, padding=0.0
    )
    return dict(
        label=label,
        lower=lower,
        upper=upper,
        result=result,
        samples=samples,
        params=params,
    )


# ---------------------------------------------------------------------------
# Tests — run for every transform in _CASES
# ---------------------------------------------------------------------------
def test_linear_bounds_sound(case):
    """LB(kappa) <= G(kappa) <= UB(kappa) for all sampled kappa (linear bounds)."""
    lb_sound = case["result"]["linear"]["sound"]["lower"]  # (c, h, w, 2)
    ub_sound = case["result"]["linear"]["sound"]["upper"]
    label = f"{case['label']} linear"

    for idx, kappa in enumerate(case["params"]):
        pixel_vals = case["samples"][idx]  # (c, h, w)
        lb_vals = lb_sound[:, :, :, 0] * kappa + lb_sound[:, :, :, 1]
        ub_vals = ub_sound[:, :, :, 0] * kappa + ub_sound[:, :, :, 1]

        assert np.all(pixel_vals - lb_vals >= -_ATOL), (
            f"{label} lower bound VIOLATION at kappa={kappa:.4f}: "
            f"min(pixel - LB) = {(pixel_vals - lb_vals).min():.6f}"
        )
        assert np.all(ub_vals - pixel_vals >= -_ATOL), (
            f"{label} upper bound VIOLATION at kappa={kappa:.4f}: "
            f"min(UB - pixel) = {(ub_vals - pixel_vals).min():.6f}"
        )


def test_pwl_bounds_sound(case):
    """PWL sound bounds enclose the pixel curve.

    PWL bound shape: (c, h, w, num_segments, 4) where axis -1 is
    [slope, intercept, seg_start, seg_end].
    Lower bound = max over segments; upper bound = min over segments.
    """
    lb_pwl = case["result"]["pwl"]["sound"]["lower"]  # (c, h, w, q, 4)
    ub_pwl = case["result"]["pwl"]["sound"]["upper"]
    c_dim, h, w, _, _ = lb_pwl.shape
    label = f"{case['label']} PWL"

    for idx, kappa in enumerate(case["params"]):
        pixel_vals = case["samples"][idx]  # (c, h, w)

        for ci in range(c_dim):
            for i in range(h):
                for j in range(w):
                    lb_val = np.max(
                        lb_pwl[ci, i, j, :, 0] * kappa + lb_pwl[ci, i, j, :, 1]
                    )
                    ub_val = np.min(
                        ub_pwl[ci, i, j, :, 0] * kappa + ub_pwl[ci, i, j, :, 1]
                    )
                    pv = pixel_vals[ci, i, j]
                    assert pv >= lb_val - _ATOL, (
                        f"{label} PWL lower VIOLATION at ({ci},{i},{j}), "
                        f"kappa={kappa:.4f}: pixel={pv:.6f} lb={lb_val:.6f}"
                    )
                    assert pv <= ub_val + _ATOL, (
                        f"{label} PWL upper VIOLATION at ({ci},{i},{j}), "
                        f"kappa={kappa:.4f}: pixel={pv:.6f} ub={ub_val:.6f}"
                    )


def test_sound_bounds_tighter_than_range(case):
    """Sound bounds must be informative — not trivially [0, 1]."""
    lb_sound = case["result"]["linear"]["sound"]["lower"]
    ub_sound = case["result"]["linear"]["sound"]["upper"]

    kappa = case["lower"]
    lb_vals = lb_sound[:, :, :, 0] * kappa + lb_sound[:, :, :, 1]
    ub_vals = ub_sound[:, :, :, 0] * kappa + ub_sound[:, :, :, 1]
    assert np.any(ub_vals - lb_vals < 1.0), (
        f"{case['label']}: bounds should be informative (narrower than [0,1])"
    )


def test_pwl_average_width_not_worse_than_linear(case):
    """Mean (UB - LB) for PWL must not exceed 1.20× the linear mean width.

    PWL bounds are expected to be tighter on average — or at worst equal.
    The 1.20 margin accommodates BaB randomness; tightness holds on average,
    not necessarily pointwise.
    """
    kappas = np.linspace(case["lower"], case["upper"], 50)
    result = case["result"]

    lb_linear = result["linear"]["sound"]["lower"]  # (c, h, w, 2)
    ub_linear = result["linear"]["sound"]["upper"]
    lb_pwl = result["pwl"]["sound"]["lower"]  # (c, h, w, q, 4)
    ub_pwl = result["pwl"]["sound"]["upper"]

    widths_linear = []
    widths_pwl = []
    for kappa in kappas:
        lb_l = lb_linear[:, :, :, 0] * kappa + lb_linear[:, :, :, 1]
        ub_l = ub_linear[:, :, :, 0] * kappa + ub_linear[:, :, :, 1]
        widths_linear.append(np.mean(ub_l - lb_l))

        lb_p = np.max(lb_pwl[:, :, :, :, 0] * kappa + lb_pwl[:, :, :, :, 1], axis=-1)
        ub_p = np.min(ub_pwl[:, :, :, :, 0] * kappa + ub_pwl[:, :, :, :, 1], axis=-1)
        widths_pwl.append(np.mean(ub_p - lb_p))

    mean_width_linear = np.mean(widths_linear)
    mean_width_pwl = np.mean(widths_pwl)
    assert mean_width_pwl <= mean_width_linear * 1.20, (
        f"{case['label']}: PWL mean width {mean_width_pwl:.5f} "
        f"> 1.20 × linear {mean_width_linear:.5f}"
    )
