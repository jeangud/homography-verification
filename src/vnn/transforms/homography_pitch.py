"""Operations related to a pitch perturbation of the camera viewpoint."""

import numpy as np

from .transform import Transform, _values_in_interval


class HomographyPitch(Transform):
    """Pitch viewpoint perturbation."""

    def __init__(self, f: float, xc: float = 0, yc: float = 0):
        """Initialize the homography transformation based on its center."""
        super().__init__()
        self.xc = xc
        self.yc = yc
        self.f = f

    def prepare_matrix(self, params):
        c = np.cos(params)
        s = np.sin(params)
        f = self.f
        f2 = f * f
        xc = self.xc
        yc = self.yc

        # This is H^{-1} from the paper
        H = np.zeros((params.shape[0], 3, 3))  # (num_params, 3, 3)
        H[:, 0, 0] = 1
        H[:, 0, 1] = xc * s / f
        H[:, 0, 2] = xc * (f * c - f - yc * s) / f
        H[:, 1, 0] = 0
        H[:, 1, 1] = (f * c + yc * s) / f
        H[:, 1, 2] = -(f2 + yc**2) * s / f
        H[:, 2, 0] = 0
        H[:, 2, 1] = s / f
        H[:, 2, 2] = c - yc * s / f

        return H

    def gradient(self, x, y, params):
        c = np.cos(params)
        s = np.sin(params)
        f = self.f
        xc = self.xc
        yc = self.yc

        # Shape (2, num_params)
        denominator = (f * c + (y - yc) * s) ** 2
        return np.array(
            [
                -f * (x - xc) * (-f * s + (y - yc) * c) / denominator,
                -f * (f**2 + (y - yc) ** 2) / denominator,
            ]
        )

    def get_max_grad_candidates(self, x, y, interval):
        f = self.f
        yc = self.yc

        # Critical value (loss of continuity)
        phi_c = np.pi / 2 if y == yc else np.arctan(-f / (y - yc))
        if _values_in_interval(phi_c, interval):
            raise ValueError("Critical pitch angle inside interval")

        phi_0 = np.arctan((y - yc) / f)
        phi_candidates = [interval[0], interval[1]]
        phi_candidates.extend(_values_in_interval(phi_0, interval))

        return self.gradient(x, y, phi_candidates)
