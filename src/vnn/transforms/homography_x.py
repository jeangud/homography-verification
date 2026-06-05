"""Operations related to a x-axis (forward) translation perturbation of the camera viewpoint."""

import numpy as np

from .transform import Transform


class HomographyX(Transform):
    """X-axis (forward) translation viewpoint perturbation."""

    def __init__(self, f: float, xc: float, yc: float, z: float):
        super().__init__()
        self.f = f
        self.xc = xc
        self.yc = yc
        self.z = z

    def prepare_matrix(self, params):
        dx = params
        xc = self.xc
        yc = self.yc
        f = self.f
        z = self.z

        # This is H^{-1} from the paper
        H = np.zeros((params.shape[0], 3, 3))  # (num_params, 3, 3)
        H[:, 0, 0] = 1
        H[:, 0, 1] = -dx * xc / (f * z)
        H[:, 0, 2] = dx * xc * yc / (f * z)
        H[:, 1, 0] = 0
        H[:, 1, 1] = -(dx * yc - f * z) / (f * z)
        H[:, 1, 2] = dx * yc**2 / (f * z)
        H[:, 2, 0] = 0
        H[:, 2, 1] = -dx / (f * z)
        H[:, 2, 2] = (dx * yc + f * z) / (f * z)

        return H

    def gradient(self, x, y, params):
        dx = np.array(params)
        xc = self.xc
        yc = self.yc
        f = self.f
        z = self.z

        # Shape (2, num_params)
        common = f * z * (y - yc) / (dx * (y - yc) - f * z) ** 2
        return np.array([(x - xc) * common, (y - yc) * common])

    def get_max_grad_candidates(self, x, y, interval):
        yc = self.yc
        f = self.f
        z = self.z

        # Critical value (loss of continuity)
        if y != yc:
            dxc = f * z / (y - yc)
            if interval[0] <= dxc <= interval[1]:
                raise ValueError("Critical dx inside interval")

        dx_candidates = [interval[0], interval[1]]
        return self.gradient(x, y, dx_candidates)  # (2, num_candidates)
