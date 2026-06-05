"""Shared test helpers for transform tests."""

import numpy as np


def _preimage(transform, x, y, kappa):
    """Return (x0, y0) = T^{-1}(kappa) applied to spatial point (x, y)."""
    H = transform.prepare_matrix(np.array([float(kappa)]))  # (1, 3, 3)
    xyz = H[0] @ np.array([x, y, 1.0])
    return xyz[0] / xyz[2], xyz[1] / xyz[2]
