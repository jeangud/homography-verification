"""Unit tests for vnn.abcrown_adapter."""

import numpy as np
import torch

from vnn.abcrown_adapter import (
    _build_per_bound,
    bounds_dict_to_perturbation,
)


def _make_pieces(slopes_intercepts, t_min, t_max):
    """Build a `(D, 2, 4)` pickle-format array.

    slopes_intercepts: list of (w0, b0, w1, b1) tuples, one per dim.
    The two pieces split [t_min, t_max] at the midpoint (matches the
    `compute_split_segments` convention in `src/vnn/pwl.py`).
    """
    t_mid = 0.5 * (t_min + t_max)
    out = np.zeros((len(slopes_intercepts), 2, 4), dtype=np.float64)
    for i, (w0, b0, w1, b1) in enumerate(slopes_intercepts):
        out[i, 0] = [w0, b0, t_min, t_mid]
        out[i, 1] = [w1, b1, t_mid, t_max]
    return out


# ──────────────────────────────────────────────────────────────────────
# _build_per_bound
# ──────────────────────────────────────────────────────────────────────
def test_lower_envelope_intersection_inside():
    """Two pieces that intersect inside [0,1] -> bp = intersection."""
    # piece0: y = t  ; piece1: y = -t + 1  ; intersect at t=0.5, y=0.5
    pieces = _make_pieces([(1.0, 0.0, -1.0, 1.0)], 0.0, 1.0)
    bp, A, b = _build_per_bound(pieces, 0.0, 1.0, is_upper=False)
    assert np.isclose(bp[0], 0.5)
    # For a max-envelope: on the left, the dominant piece is the LARGER one at t_min=0.
    # At t=0: piece0=0, piece1=1 -> left piece = piece1.
    # On the right (t > 0.5): piece0 wins.
    # Check continuity at bp.
    L_left = A[0, 0] * bp[0] + b[0, 0]
    L_right = A[1, 0] * bp[0] + b[1, 0]
    assert np.isclose(L_left, L_right)
    # Envelope is max — at bp it equals 0.5
    assert np.isclose(L_left, 0.5)


def test_upper_envelope_intersection_inside():
    """min-envelope of two upper pieces."""
    pieces = _make_pieces([(1.0, 0.0, -1.0, 1.0)], 0.0, 1.0)
    bp, A, b = _build_per_bound(pieces, 0.0, 1.0, is_upper=True)
    assert np.isclose(bp[0], 0.5)
    U_left = A[0, 0] * bp[0] + b[0, 0]
    U_right = A[1, 0] * bp[0] + b[1, 0]
    assert np.isclose(U_left, U_right)
    # min-envelope: at t=0 picks min(0, 1) = piece0 ; at t=1 picks min(1, 0) = piece1.
    # Left piece (t < 0.5) should be piece0 (slope +1, intercept 0).
    assert np.isclose(A[0, 0], 1.0) and np.isclose(b[0, 0], 0.0)


def test_intersection_outside_collapses():
    """If pieces don't intersect inside [t_min, t_max], collapse to dominant."""
    # piece0: y = t + 5  ; piece1: y = t      (parallel-ish; same slope -> truly parallel)
    # Use slightly different slopes so they intersect outside [0,1].
    # piece0: y = t + 10 ; piece1: y = 2*t  ; intersect at t = 10 (outside)
    pieces = _make_pieces([(1.0, 10.0, 2.0, 0.0)], 0.0, 1.0)
    bp, A, b = _build_per_bound(pieces, 0.0, 1.0, is_upper=False)
    # Lower-envelope = max. At t=0: piece0=10, piece1=0 -> piece0 dominates.
    # At t=1: piece0=11, piece1=2 -> piece0 still dominates. Collapse to piece0.
    assert np.allclose(A[:, 0], 1.0)
    assert np.allclose(b[:, 0], 10.0)


def test_parallel_lines_collapse():
    """Truly parallel lines: collapse to the dominant one."""
    # piece0: y = t + 1 ; piece1: y = t  (slope identical)
    pieces = _make_pieces([(1.0, 1.0, 1.0, 0.0)], 0.0, 1.0)
    bp, A, b = _build_per_bound(pieces, 0.0, 1.0, is_upper=False)
    # max-envelope -> piece0 (intercept 1)
    assert np.allclose(A, 1.0)
    assert np.allclose(b, 1.0)


# ──────────────────────────────────────────────────────────────────────
# bounds_dict_to_perturbation (end-to-end on a tiny synthetic image)
# ──────────────────────────────────────────────────────────────────────
def test_bounds_dict_to_perturbation_shapes():
    """1x2x2 synthetic bounds; check returned PerturbationPWL has right shapes."""
    C, H, W, K = 1, 2, 2, 2
    t_min, t_max = 0.0, 1.0

    lower = np.zeros((C, H, W, K, 4), dtype=np.float64)
    upper = np.zeros((C, H, W, K, 4), dtype=np.float64)
    # Constant bounds: pixel value in [0.2, 0.8] independent of t.
    lower[..., 0, :] = [0.0, 0.2, t_min, 0.5 * (t_min + t_max)]
    lower[..., 1, :] = [0.0, 0.2, 0.5 * (t_min + t_max), t_max]
    upper[..., 0, :] = [0.0, 0.8, t_min, 0.5 * (t_min + t_max)]
    upper[..., 1, :] = [0.0, 0.8, 0.5 * (t_min + t_max), t_max]

    ptb, x_mid, chw = bounds_dict_to_perturbation({"lower": lower, "upper": upper})
    assert chw == (C, H, W)
    assert x_mid.shape == (1, C, H, W)
    D = C * H * W
    assert ptb.bp_lower.shape == (1, D)
    assert ptb.bp_upper.shape == (1, D)
    assert ptb.pwl_lower_A.shape == (1, 2, D)
    # Constant bounds -> concrete x_L = 0.2, x_U = 0.8 everywhere.
    assert torch.allclose(ptb.x_L, torch.full_like(ptb.x_L, 0.2), atol=1e-5)
    assert torch.allclose(ptb.x_U, torch.full_like(ptb.x_U, 0.8), atol=1e-5)
    assert torch.allclose(x_mid, torch.full_like(x_mid, 0.5), atol=1e-5)


def test_bounds_dict_continuity():
    """After conversion, PWL pieces must agree at the chosen breakpoints."""
    C, H, W, K = 1, 1, 2, 2
    t_min, t_max = -1.0, 1.0
    rng = np.random.default_rng(0)

    lower = np.zeros((C, H, W, K, 4))
    upper = np.zeros((C, H, W, K, 4))
    for i in range(H):
        for j in range(W):
            # Random non-parallel lines that intersect inside [-1, 1].
            w0, w1 = rng.uniform(-1, 1), rng.uniform(-1, 1)
            while abs(w0 - w1) < 0.2:
                w1 = rng.uniform(-1, 1)
            b0 = rng.uniform(-0.5, 0.5)
            # Pick b1 such that intersection is in [-0.5, 0.5]
            t_int = rng.uniform(-0.5, 0.5)
            b1 = (w0 - w1) * t_int + b0
            lower[0, i, j, 0] = [w0, b0, t_min, 0.0]
            lower[0, i, j, 1] = [w1, b1, 0.0, t_max]
            # Upper: ensure strictly above lower by adding a big offset.
            upper[0, i, j, 0] = [w0, b0 + 5.0, t_min, 0.0]
            upper[0, i, j, 1] = [w1, b1 + 5.0, 0.0, t_max]

    ptb, _, _ = bounds_dict_to_perturbation({"lower": lower, "upper": upper})
    # Continuity is checked inside the constructor; if we got here, it passed.
    # Spot-check: evaluate L at bp_lower from both pieces.
    bp = ptb.bp_lower
    L0 = ptb.pwl_lower_A[:, 0] * bp + ptb.pwl_lower_b[:, 0]
    L1 = ptb.pwl_lower_A[:, 1] * bp + ptb.pwl_lower_b[:, 1]
    assert torch.allclose(L0, L1, atol=1e-4)
