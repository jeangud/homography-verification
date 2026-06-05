"""Operations related to a roll perturbation of the camera viewpoint."""

import numpy as np

from .transform import Transform, _values_in_interval


class HomographyRoll(Transform):
    """Roll viewpoint perturbation."""

    def __init__(self, xc: float, yc: float):
        super().__init__()
        self.xc = xc
        self.yc = yc

    def prepare_matrix(self, params):
        c = np.cos(params)
        s = np.sin(params)
        xc = self.xc
        yc = self.yc

        # This is H^{-1} from the paper
        H = np.zeros((params.shape[0], 3, 3))  # (num_params, 3, 3)
        H[:, 0, 0] = c
        H[:, 0, 1] = -s
        H[:, 0, 2] = -xc * c + xc + yc * s
        H[:, 1, 0] = s
        H[:, 1, 1] = c
        H[:, 1, 2] = -xc * s - yc * c + yc
        H[:, 2, 0] = 0
        H[:, 2, 1] = 0
        H[:, 2, 2] = 1

        return H

    def gradient(self, x, y, params):
        c = np.cos(params)
        s = np.sin(params)
        xc = self.xc
        yc = self.yc

        return np.array([-(x - xc) * s - (y - yc) * c, (x - xc) * c - (y - yc) * s])

    def get_max_grad_candidates(self, x, y, interval):
        xc = self.xc
        yc = self.yc

        phi1 = np.arctan2(-(y - yc), (x - xc))
        phi2 = np.arctan2((x - xc), (y - yc))
        phi_candidates = [interval[0], interval[1]]
        phi_candidates.extend(_values_in_interval(phi1, interval))
        phi_candidates.extend(_values_in_interval(phi2, interval))

        return self.gradient(x, y, phi_candidates)
