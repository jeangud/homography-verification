"""Operations related to a y-axis (to the right) translation perturbation of the camera viewpoint."""

import numpy as np

from .transform import Transform


class HomographyY(Transform):
    """Y-axis (to the right) translation viewpoint perturbation."""

    def __init__(self, f: float, xc: float, yc: float, z: float):
        super().__init__()
        self.f = f
        self.xc = xc
        self.yc = yc
        self.z = z

    def prepare_matrix(self, params):
        dy = params
        yc = self.yc
        z = self.z

        # This is H^{-1} from the paper
        H = np.zeros((params.shape[0], 3, 3))  # (num_params, 3, 3)
        H[:, 0, 0] = 1
        H[:, 0, 1] = -dy / z
        H[:, 0, 2] = dy * yc / z
        H[:, 1, 0] = 0
        H[:, 1, 1] = 1
        H[:, 1, 2] = 0
        H[:, 2, 0] = 0
        H[:, 2, 1] = 0
        H[:, 2, 2] = 1

        return H

    def gradient(self, x, y, params):
        f = self.f
        yc = self.yc
        z = self.z

        grad = np.zeros((2, len(params)))
        grad[0, :] = yc / f - y / z  # Same term for all params (constant gradient)
        # grad[1, :] remains zero

        return grad

    def get_max_grad_candidates(self, x, y, interval):
        # Gradient is independent of interval parameter here,
        # we can just compute at the interval start point for example
        return self.gradient(x, y, [interval[0]])
