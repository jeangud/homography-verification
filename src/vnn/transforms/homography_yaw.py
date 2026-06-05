"""Operations related to a yaw perturbation of the camera viewpoint."""

import numpy as np

from .transform import Transform, _values_in_interval


class HomographyYaw(Transform):
    """Yaw viewpoint perturbation"""

    def __init__(self, f: float, xc: float = 0, yc: float = 0):
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
        H[:, 0, 0] = c - xc * s / f
        H[:, 0, 1] = 0
        H[:, 0, 2] = (f2 + xc**2) * s / f
        H[:, 1, 0] = -yc * s / f
        H[:, 1, 1] = 1
        H[:, 1, 2] = yc * (f * (c - 1) + xc * s) / f
        H[:, 2, 0] = -s / f
        H[:, 2, 1] = 0
        H[:, 2, 2] = c + xc * s / f

        return H

    def gradient(self, x, y, params):
        c = np.cos(params)
        s = np.sin(params)
        f = self.f
        xc = self.xc
        yc = self.yc

        # Shape (2, num_params)
        denominator = (f * c - (x - xc) * s) ** 2
        return np.array(
            [
                f * (f**2 + (x - xc) ** 2) / denominator,
                f * (y - yc) * (f * s + (x - xc) * c) / denominator,
            ]
        )

    def get_max_grad_candidates(self, x, y, interval):
        f = self.f
        xc = self.xc

        # Critical value (loss of continuity)
        psi_c = np.pi / 2 if x == xc else np.arctan(f / (x - xc))
        if _values_in_interval(psi_c, interval):
            raise ValueError("Critical angle inside interval")

        psi_0 = np.arctan((xc - x) / f)
        psi_candidates = [interval[0], interval[1]]
        psi_candidates.extend(_values_in_interval(psi_0, interval))

        return self.gradient(x, y, psi_candidates)
