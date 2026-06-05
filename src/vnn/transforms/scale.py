"""Operations related to a scaling transformation."""

import numpy as np

from .transform import Transform


class Scale(Transform):
    """An isotropic scaling transform around a given center."""

    def __init__(self, x0: float = 0, y0: float = 0):
        """Initialize the scaling transformation based on its center."""
        super().__init__()
        self.x0 = x0
        self.y0 = y0

    def __str__(self):
        return f"{self.__class__.__name__}({self.x0},{self.y0})"

    def prepare_matrix(self, params):
        """Prepare the scale matrix for a set of scaling factors."""
        assert np.all(params > 0), "Scale parameters must be strictly positive (s > 0)"
        s = params
        x0 = self.x0
        y0 = self.y0

        # This is H^{-1} from the paper
        # We explicitly map a point in the scaled image to the original image
        H = np.zeros((params.shape[0], 3, 3))  # (num_params, 3, 3)
        H[:, 0, 0] = 1.0 / s
        H[:, 0, 1] = 0
        H[:, 0, 2] = x0 - x0 / s
        H[:, 1, 0] = 0
        H[:, 1, 1] = 1.0 / s
        H[:, 1, 2] = y0 - y0 / s
        H[:, 2, 0] = 0
        H[:, 2, 1] = 0
        H[:, 2, 2] = 1

        return H

    def gradient(self, x, y, params):
        """Compute the scale gradient for the given parameters."""
        s = params

        return np.array(
            [
                -(x - self.x0) / (s**2),
                -(y - self.y0) / (s**2),
            ]
        )

    def get_max_grad_candidates(self, x, y, interval):
        """Computes candidate locations for the gradient maximum.
        For scaling, the gradient magnitude is monotonic with respect to s (proportional to 1/s^2).
        Thus, for s > 0, the maximum will always occur at the lower interval boundary (smallest s).
        """
        assert interval[0] > 0, "Lower scale bound must be strictly positive (s > 0)"

        s_candidates = [interval[0]]

        return self.gradient(x, y, np.array(s_candidates))  # (2, num_candidates)
