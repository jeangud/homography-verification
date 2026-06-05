"""Abstract class representing a generic geometric transform."""

from abc import ABC, abstractmethod

import cv2
import numpy as np
import torch
import torchvision.transforms.functional as F
from PIL import Image

from .homogeneous import to_homogeneous
from .image import to_ij, to_xy
from .interpolation import interpolate, get_max_interpolation_grad


def _values_in_interval(base, interval, period=np.pi):
    """Return all base + k*period values within [interval[0], interval[1]]."""
    lo, hi = interval
    k = int(np.ceil((lo - base) / period))
    values = []
    v = base + k * period
    while v <= hi:
        values.append(v)
        v += period
    return values


class Transform(ABC):
    """Abstract class representing a generic geometric transform."""

    def __init__(self):
        self.__max_grad_I = None  # Cache for the maximum gradient of the interpolation

    def __str__(self):
        """Return a string representation of the transformation."""
        return f"{self.__class__.__name__}"

    @abstractmethod
    def prepare_matrix(self, params):
        """Prepare the transformation matrix for a set of parameters, based on image"""

    @abstractmethod
    def gradient(self, x, y, params):
        """Compute the gradient of the transformation at a given point (x, y)"""

    @abstractmethod
    def get_max_grad_candidates(self, x, y, interval):
        """Compute candidate gradient maxima over the given interval"""

    def apply(self, img, params, padding):
        """Apply the transform to the given image, for a range of parameters.
        This is our custom implementation, which is not as efficient as OpenCV or PIL.
        It also handles border effects differently."""
        i_coords, j_coords = self.get_original_pixel_locations(img, params)
        # (num_params x h x w) Independent of channel!

        num_params = len(params)
        c, h, w = img.shape
        pixel_values = interpolate(
            img, i_coords, j_coords, padding
        )  # (c, num_params x h x w)
        pixel_values = pixel_values.reshape(c, num_params, h, w)
        return pixel_values.transpose(1, 0, 2, 3)  # (num_params, c, h, w)

    def apply_opencv(self, img, params, padding_value):
        """Apply the transform to the given image, for a range of parameters.
        This method uses OpenCV's warpPerspective for efficiency. However, the resulting
        image is not identical to our custom implemetation, PIL, nor PyTorch."""
        Hs = self.prepare_matrix(params)  # (num_params, 3, 3)
        _, h, w = img.shape
        img_hwc = img.transpose(1, 2, 0)  # (h, w, c) — OpenCV convention

        transformed_images = []
        for i in range(len(params)):
            warped = cv2.warpPerspective(
                img_hwc,
                Hs[i],
                (w, h),
                flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=padding_value,
            )
            # OpenCV drops the channel dim for single-channel inputs
            if warped.ndim == 2:
                warped = warped[:, :, np.newaxis]
            transformed_images.append(warped)

        return np.stack(transformed_images).transpose(
            0, 3, 1, 2
        )  # (num_params, c, h, w)

    def apply_pil(self, img, params, padding_value):
        """Apply the transform to the given image, for a range of parameters.
        This method uses PIL's implementation for efficiency.
        Note: PIL's float32 mode ('F') is single-channel only, so multi-channel
        images are processed one channel at a time."""
        Hs = self.prepare_matrix(params)  # (num_params, 3, 3)
        Hs_flat = Hs.reshape(len(params), 9)  # (num_params, 9)
        c, h, w = img.shape

        # Build PIL images once per channel; reuse across all params
        pil_channels = [Image.fromarray(img[ch]) for ch in range(c)]

        results = []
        for i in range(len(params)):
            channels = [
                np.array(
                    pil_ch.transform(
                        (w, h),
                        Image.PERSPECTIVE,
                        Hs_flat[i],
                        resample=Image.Resampling.BILINEAR,
                        fillcolor=padding_value,
                    )
                )
                for pil_ch in pil_channels
            ]
            results.append(np.stack(channels))  # (c, h, w)

        return np.stack(results)  # (num_params, c, h, w)

    def apply_torch(self, img, params, padding_value):
        """Apply the transform to the given image, for a range of parameters.
        This method uses PyTorch's functional.perspective."""
        Hs = self.prepare_matrix(params)  # (num_params, 3, 3)
        _, h, w = img.shape
        img_tensor = torch.tensor(img).unsqueeze(0)  # (1, c, h, w)

        # Apply the transformations using PyTorch
        transformed_images = [
            F.perspective(
                img_tensor,
                endpoints=[[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]],
                startpoints=[
                    [Hs[i, 0, 2], Hs[i, 1, 2]],
                    [
                        Hs[i, 0, 2] + Hs[i, 0, 0] * (w - 1),
                        Hs[i, 1, 2] + Hs[i, 1, 0] * (w - 1),
                    ],
                    [
                        Hs[i, 0, 2] + Hs[i, 0, 0] * (w - 1) + Hs[i, 0, 1] * (h - 1),
                        Hs[i, 1, 2] + Hs[i, 1, 0] * (w - 1) + Hs[i, 1, 1] * (h - 1),
                    ],
                    [
                        Hs[i, 0, 2] + Hs[i, 0, 1] * (h - 1),
                        Hs[i, 1, 2] + Hs[i, 1, 1] * (h - 1),
                    ],
                ],
                interpolation=F.InterpolationMode.BILINEAR,
                fill=padding_value,
            )
            for i in range(len(params))
        ]

        return (
            torch.stack(transformed_images).squeeze(1).numpy()
        )  # (num_params, c, h, w)

    def get_original_pixel_locations(self, img, params):
        """Computes the original pixel locations for each of the transformed pixels in
        img."""
        # Pixel locations in result image
        h, w = img.shape[-2:]
        I_new = np.repeat(np.arange(h), w)  # (h x w) = (num_pixels)
        J_new = np.tile(np.arange(w), h)  # (h x w) = (num_pixels)
        # NOTE: num_pixels is for a single channel, same mapping for each channel

        # Find pixel locations in the original image
        X_new, Y_new = to_xy(i=I_new, j=J_new)  # (num_pixels)
        XY_new_h = to_homogeneous(np.vstack((X_new, Y_new)))  # (3, num_pixels)
        # Transform from new coordinates (x_new, y_new) to original coordinates (x, y)
        T_orig_from_new = self.prepare_matrix(params)  # (num_params, 3, 3)
        XY_h = T_orig_from_new @ XY_new_h  # (num_params, 3, num_pixels)

        # Reshape to (3, num_params x num_pixels) to help with further processing
        XY_h = XY_h.transpose(1, 0, 2)  # (3, num_params, num_pixels)
        XY_h = XY_h.reshape(3, -1)  # (3, num_params x num_pixels)
        # Convert to non-homogeneous coords with normalizing (non-affine transform)
        i_coords, j_coords = to_ij(x=XY_h[0] / XY_h[2], y=XY_h[1] / XY_h[2])
        # (num_params x num_pixels)

        return i_coords, j_coords  # (num_params x num_pixels)

    def get_lipschitz_analytical(self, img, x, y, interval):
        """Compute a possible Lipschitz constant for the given interval"""
        # Compute the maximum of gradient(x,y) within the subdomain
        # Interpolation gradient [dI/dx, dI/dy]
        if self.__max_grad_I is None:
            # Cache the result since it depends only on the image, not transform params
            self.__max_grad_I = get_max_interpolation_grad(img)  # (1 x 2)

        # Gradient of the inverse transform [dx(K)/dK, dy(K)/dK] (2 x num_candidates)
        max_grad_T_candidates = self.get_max_grad_candidates(x, y, interval)

        # By the chain rule: dG/dK = dI/dx * dx0/dK + dI/dy * dy0/dK
        # By the triangle inequality (Eq. 63 in the paper):
        #   sup|dG/dK| <= sup|dI/dx| * sup|dx0/dK| + sup|dI/dy| * sup|dy0/dK|
        # We maximise each component independently so that we never underestimate
        # when du0/dK and dv0/dK peak at different parameter values.
        dI_dx_max = self.__max_grad_I[0, 0]
        dI_dy_max = self.__max_grad_I[0, 1]
        sup_du0 = np.max(np.abs(max_grad_T_candidates[0]))
        sup_dv0 = np.max(np.abs(max_grad_T_candidates[1]))

        # Majorant of Lipschitz constant for G(κ) only
        # Per Eq. 63, the caller must add |w*| to get the Lipschitz constant of J(κ) = LB(κ) − G(κ)
        return dI_dx_max * sup_du0 + dI_dy_max * sup_dv0  # (1x1)

    def get_lipschitz_numerical(self, img, x, y, interval):
        """Sample gradients over the interval to estimate the Lipschitz"""
        # Compute the maximum of gradient(x,y) within the subdomain
        # Interpolation gradient [dI/dx, dI/dy]
        if self.__max_grad_I is None:
            # Cache the result since it depends only on the image, not transform params
            self.__max_grad_I = get_max_interpolation_grad(img)  # (1 x 2)

        # Evaluate the gradient at many samples in the interval
        theta_candidates = np.linspace(interval[0], interval[1], 100)
        max_grad_T_candidates = self.gradient(x, y, theta_candidates)
        max_grad_G_candidates = self.__max_grad_I @ max_grad_T_candidates
        # (1 x num_candidates)

        return np.max(np.abs(max_grad_G_candidates))  # Lipschitz constant (1x1)
