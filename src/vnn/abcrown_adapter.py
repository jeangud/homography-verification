"""Adapter: precomputed PWL pixel bounds (pickle) -> auto_LiRPA PerturbationPWL.

The bounds pickles produced by `src/vnn/pwl.py::fit_pwl_bound` store, for every
pixel and every segment, an array of shape `(C, H, W, K, 4)` whose last axis is
`[slope, intercept, segment_start, segment_end]`. Each of the K pieces is a
*globally-valid* linear bound over `[t_min, t_max]` (the LP enforces validity
over the whole domain and minimizes error only over its sub-segment). The tight
2-piece envelope is therefore:

    L_envelope(t) = max(piece0(t), piece1(t))     # lower bound
    U_envelope(t) = min(piece0(t), piece1(t))     # upper bound

This module converts that representation into a `PerturbationPWL` instance,
which requires the two pieces to be **continuous at a breakpoint**. We choose
each per-pixel, per-bound breakpoint as the *intersection* of the two pieces:

    t* = (b1 - b0) / (w0 - w1)

clamped to `[t_min, t_max]`. Degenerate cases (parallel lines, intersection
outside the domain) are handled by collapsing both pieces to the single
dominant line.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import torch

from auto_LiRPA.perturbations import PerturbationPWL


def _build_per_bound(
    pieces: np.ndarray,
    t_min: float,
    t_max: float,
    is_upper: bool,
    parallel_eps: float = 1e-12,
):
    """Compute per-dim breakpoint and continuous pieces for one bound.

    Args:
        pieces: (D, 2, 4) array — last axis is [slope, intercept, seg_start, seg_end].
        is_upper: True if these are upper-bound pieces (envelope is `min`),
            False if lower-bound pieces (envelope is `max`).

    Returns:
        bp: (D,) numpy array of per-dim breakpoints in [t_min, t_max].
        A:  (2, D) slopes for the two continuous pieces.
        b:  (2, D) intercepts for the two continuous pieces.
    """
    w0 = pieces[:, 0, 0]
    b0 = pieces[:, 0, 1]
    w1 = pieces[:, 1, 0]
    b1 = pieces[:, 1, 1]

    dw = w0 - w1
    near_parallel = np.abs(dw) < parallel_eps

    # Intersection (with safe denominator for the near-parallel mask)
    safe_dw = np.where(near_parallel, 1.0, dw)
    t_int = (b1 - b0) / safe_dw

    inside = (t_int >= t_min) & (t_int <= t_max) & ~near_parallel

    # For dims where the intersection is outside the domain (or lines are
    # parallel), one piece dominates over the whole interval. For a lower-bound
    # envelope (max), pick the piece with the larger value at t_min; for an
    # upper-bound envelope (min), pick the smaller. Either endpoint works since
    # they don't cross inside.
    v0_min = w0 * t_min + b0
    v1_min = w1 * t_min + b1
    if is_upper:
        # Envelope is min(piece0, piece1) — dominant = the lower piece
        dominant_is_0 = v0_min <= v1_min
    else:
        dominant_is_0 = v0_min >= v1_min

    dom_w = np.where(dominant_is_0, w0, w1)
    dom_b = np.where(dominant_is_0, b0, b1)

    # For each dim, decide which of the two pickle pieces is "piece 0" vs
    # "piece 1" of the continuous PerturbationPWL (piece 0 applies for
    # t <= bp). When inside, the envelope follows the dominant piece on each
    # side of the intersection; outside, both PWL pieces are the dominant line.

    # Piece-0 (t <= bp) and piece-1 (t >= bp) of the envelope:
    # On the left of t_int, pick whichever pickle piece is dominant just left
    # of t_int (i.e. at a t slightly < t_int). We use t_min as a witness — both
    # are linear so the sign of (piece0-piece1) is constant on each side.
    if is_upper:
        # left side: dominant is the smaller one at t_min
        left_is_0 = v0_min <= v1_min
    else:
        left_is_0 = v0_min >= v1_min

    A_left = np.where(left_is_0, w0, w1)
    b_left = np.where(left_is_0, b0, b1)
    A_right = np.where(left_is_0, w1, w0)
    b_right = np.where(left_is_0, b1, b0)

    # Where intersection is outside / parallel: collapse to dominant line.
    A_left = np.where(inside, A_left, dom_w)
    b_left = np.where(inside, b_left, dom_b)
    A_right = np.where(inside, A_right, dom_w)
    b_right = np.where(inside, b_right, dom_b)

    # Breakpoint: clamp intersection into [t_min, t_max]; for collapsed dims
    # any bp in range works since pieces are identical — pick (t_min+t_max)/2.
    bp = np.where(inside, np.clip(t_int, t_min, t_max), 0.5 * (t_min + t_max))

    A = np.stack([A_left, A_right], axis=0)  # (2, D)
    b = np.stack([b_left, b_right], axis=0)  # (2, D)
    return bp.astype(np.float64), A.astype(np.float64), b.astype(np.float64)


def bounds_dict_to_perturbation(
    bounds_dict: dict,
    *,
    bound_kind: str = "sound",
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.float32,
) -> tuple[PerturbationPWL, torch.Tensor, tuple[int, int, int]]:
    """Build a `PerturbationPWL` from a `bounds["pwl"][bound_kind]` sub-dict.

    Args:
        bounds_dict: the inner `bounds["pwl"][bound_kind]` dict with "lower"/"upper".
        bound_kind: "sound" or "unsound" (just for error messages).
        device: target torch device.
        dtype: target torch dtype.

    Returns:
        ptb: `PerturbationPWL` with batch size 1, D = C*H*W.
        x_mid: (1, C, H, W) torch tensor at the midpoint of (x_L, x_U).
        chw: (C, H, W) original shape.
    """
    lower = bounds_dict["lower"]
    upper = bounds_dict["upper"]

    if lower.ndim != 5 or lower.shape != upper.shape:
        raise ValueError(
            f"Expected lower/upper of shape (C,H,W,K,4); got {lower.shape} and {upper.shape}"
        )
    C, H, W, K, last = lower.shape
    if K != 2 or last != 4:
        raise ValueError(
            f"Adapter currently only supports K=2 segments and last-dim=4, got K={K}, last={last}"
        )

    # Global parameter interval (consistent across bounds and pixels).
    t_min = float(lower[..., 0, 2].min())
    t_max = float(lower[..., -1, 3].max())
    if (
        not np.isclose(lower[..., 0, 2], t_min).all()
        or not np.isclose(upper[..., 0, 2], t_min).all()
    ):
        raise ValueError("Inconsistent t_min across pixels/bounds.")
    if (
        not np.isclose(lower[..., -1, 3], t_max).all()
        or not np.isclose(upper[..., -1, 3], t_max).all()
    ):
        raise ValueError("Inconsistent t_max across pixels/bounds.")
    if not (t_min < t_max):
        raise ValueError(f"Degenerate interval [t_min={t_min}, t_max={t_max}]")

    D = C * H * W
    lower_flat = lower.reshape(D, K, 4)
    upper_flat = upper.reshape(D, K, 4)

    bp_lower_np, lA_np, lb_np = _build_per_bound(
        lower_flat, t_min, t_max, is_upper=False
    )
    bp_upper_np, uA_np, ub_np = _build_per_bound(
        upper_flat, t_min, t_max, is_upper=True
    )

    # Add batch dim (B=1).
    def _t(arr):
        return torch.as_tensor(arr, dtype=dtype, device=device).unsqueeze(0)

    bp_lower = _t(bp_lower_np)  # (1, D)
    bp_upper = _t(bp_upper_np)  # (1, D)
    lower_A = torch.as_tensor(lA_np, dtype=dtype, device=device).unsqueeze(
        0
    )  # (1, 2, D)
    lower_b = torch.as_tensor(lb_np, dtype=dtype, device=device).unsqueeze(0)
    upper_A = torch.as_tensor(uA_np, dtype=dtype, device=device).unsqueeze(0)
    upper_b = torch.as_tensor(ub_np, dtype=dtype, device=device).unsqueeze(0)

    ptb = PerturbationPWL(
        t_min=t_min,
        t_max=t_max,
        lower_A=lower_A,
        upper_A=upper_A,
        lower_b=lower_b,
        upper_b=upper_b,
        lower_breakpoints=bp_lower,
        upper_breakpoints=bp_upper,
    )

    # The model expects 4-D image inputs; reshape the per-pixel concrete bounds
    # accordingly. (Internally PerturbationPWL stores A/b flat over D dims.)
    ptb.x_L = ptb.x_L.view(1, C, H, W)
    ptb.x_U = ptb.x_U.view(1, C, H, W)

    x_mid = (ptb.x_L + ptb.x_U) / 2
    return ptb, x_mid, (C, H, W)


def pwl_pickle_to_perturbation(
    pickle_path: str | Path,
    *,
    bound_kind: str = "sound",
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.float32,
) -> tuple[PerturbationPWL, torch.Tensor, tuple[int, int, int]]:
    """Load a pickle from `scripts/calculate_bounds.py` and return a PerturbationPWL.

    Args:
        pickle_path: path to a `.pkl` file produced by the bounds-calculation script.
        bound_kind: "sound" (default) or "unsound".
        device, dtype: target torch device/dtype for the returned tensors.

    Returns:
        (ptb, x_mid, chw) — see `bounds_dict_to_perturbation`.
    """
    with Path(pickle_path).open("rb") as f:
        data = pickle.load(f)
    return bounds_dict_to_perturbation(
        data["bounds"]["pwl"][bound_kind],
        bound_kind=bound_kind,
        device=device,
        dtype=dtype,
    )
