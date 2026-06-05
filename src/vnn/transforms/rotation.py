"""Operations related to a rotation transformation."""

import numpy as np

from .transform import Transform, _values_in_interval


class Rotation(Transform):
    """A clockwise rotation around a given center."""

    def __init__(self, x0: float = 0, y0: float = 0):
        super().__init__()
        self.x0 = x0
        self.y0 = y0

    def __str__(self):
        return f"{self.__class__.__name__}({self.x0},{self.y0})"

    def prepare_matrix(self, params):
        c = np.cos(params)
        s = np.sin(params)
        x0 = self.x0
        y0 = self.y0

        # This is H^{-1} from the paper
        # We directly use the closed-form solution
        # This is the active transformation (moves the points)
        R = np.zeros((params.shape[0], 3, 3))  # (num_params, 3, 3)
        R[:, 0, 0] = c
        R[:, 0, 1] = s
        R[:, 1, 0] = -s
        R[:, 1, 1] = c
        R[:, 0, 2] = -x0 * c + x0 - y0 * s
        R[:, 1, 2] = x0 * s - y0 * c + y0
        R[:, 2, 2] = 1

        return R

    def gradient(self, x, y, params):
        c = np.cos(params)
        s = np.sin(params)

        return np.array(
            [
                -(x - self.x0) * s + (y - self.y0) * c,
                -(x - self.x0) * c - (y - self.y0) * s,
            ]
        )

    def get_max_grad_candidates(self, x, y, interval):
        # Compute rotation static points
        angle_1 = np.arctan2(-(x - self.x0), (y - self.y0))
        angle_2 = np.arctan2((y - self.y0), (x - self.x0))

        # Candidate parameters for the max. gradient location
        theta_candidates = [interval[0], interval[1]]
        theta_candidates.extend(_values_in_interval(angle_1, interval))
        theta_candidates.extend(_values_in_interval(angle_2, interval))

        return self.gradient(x, y, theta_candidates)
