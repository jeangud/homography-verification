"""Operations related to a z-axis (down) translation perturbation of the camera viewpoint."""

import numpy as np

from .transform import Transform


class HomographyZ(Transform):
    """Z-axis (down) translation viewpoint perturbation."""

    def __init__(self, f: float, xc: float, yc: float, z: float):
        super().__init__()
        self.f = f
        self.xc = xc
        self.yc = yc
        self.z = z

    def prepare_matrix(self, params):
        dz = params
        yc = self.yc
        z = self.z

        # This is H^{-1} from the paper
        H = np.zeros((params.shape[0], 3, 3))  # (num_params, 3, 3)
        H[:, 0, 0] = 1
        H[:, 0, 1] = 0
        H[:, 0, 2] = 0
        H[:, 1, 0] = 0
        H[:, 1, 1] = z / (dz + z)
        H[:, 1, 2] = dz * yc / (dz + z)
        H[:, 2, 0] = 0
        H[:, 2, 1] = 0
        H[:, 2, 2] = 1

        return H

    def gradient(self, x, y, params):
        dz = np.array(params)
        yc = self.yc
        z = self.z

        # Shape (2, num_params)
        grad = np.zeros((2, len(params)))
        # grad[0, :] remains zero
        grad[1, :] = -z * (y - yc) / (z + dz) ** 2

        return grad

    def get_max_grad_candidates(self, x, y, interval):
        if interval[0] <= -self.z <= interval[1]:
            raise ValueError("Critical z inside interval")

        # Shape (2, num_params)
        dz_candidates = [interval[0], interval[1]]
        return self.gradient(x, y, dz_candidates)
