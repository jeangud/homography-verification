"""Operations related to a shear transformation."""

import numpy as np

from .transform import Transform


class ShearX(Transform):
    """A horizontal shear transform"""

    def __init__(self, y0: float = 0):
        super().__init__()
        self.y0 = y0

    def __str__(self):
        return f"{self.__class__.__name__}({self.y0})"

    def prepare_matrix(self, params):
        # params represents shear angle phi
        hx = np.tan(params)
        # This is H^{-1} from the paper
        H = np.zeros((params.shape[0], 3, 3))  # (num_params, 3, 3)
        H[:, 0, 0] = 1
        H[:, 0, 1] = -hx
        H[:, 0, 2] = hx * self.y0
        H[:, 1, 0] = 0
        H[:, 1, 1] = 1
        H[:, 1, 2] = 0
        H[:, 2, 0] = 0
        H[:, 2, 1] = 0
        H[:, 2, 2] = 1

        return H

    def gradient(self, x, y, params):
        # gradient of x_old with respect to phi is -(y - y0) * sec^2(phi)
        hx = np.tan(params)
        sec2 = 1.0 + hx**2
        grad_x = -(y - self.y0) * sec2
        grad_y = np.zeros_like(params)
        return np.array([grad_x, grad_y])

    def get_max_grad_candidates(self, x, y, interval):
        # sec^2(phi) is monotonic on [0, pi/2] and [-pi/2, 0] and has a minimum at 0.
        # Therefore, the maximum absolute gradient ALWAYS occurs at the interval boundaries.
        candidates = [interval[0], interval[1]]
        return self.gradient(x, y, np.array(candidates))
