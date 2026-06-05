"""Operations related to a x-translation transformation."""

import numpy as np

from .transform import Transform


class TranslateX(Transform):
    """A x-translation transform"""

    def __init__(self):
        super().__init__()

    def __str__(self):
        return f"{self.__class__.__name__}()"

    def prepare_matrix(self, params):
        # This is H^{-1} from the paper
        H = np.zeros((params.shape[0], 3, 3))  # (num_params, 3, 3)
        H[:, 0, 0] = 1
        H[:, 0, 1] = 0
        H[:, 0, 2] = -params
        H[:, 1, 0] = 0
        H[:, 1, 1] = 1
        H[:, 1, 2] = 0
        H[:, 2, 0] = 0
        H[:, 2, 1] = 0
        H[:, 2, 2] = 1

        return H

    def gradient(self, x, y, params):
        grad = np.zeros((2, len(params)))
        grad[0, :] = -1
        return grad

    def get_max_grad_candidates(self, x, y, interval):
        # Gradient is constant, so any values within interval are valid candidates
        candidates = [interval[0]]
        return self.gradient(x, y, candidates)
