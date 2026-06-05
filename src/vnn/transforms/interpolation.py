"""Operations related to bilinear interpolation."""

import cv2
import numpy as np


def interpolate(img, i_coords, j_coords, padding):
    """Bilinear interpolation of pixel values in the image."""
    # Corner locations
    I1 = np.floor(i_coords).astype(int)  # (1, num_locations)
    J1 = np.floor(j_coords).astype(int)
    I2 = I1 + 1
    J2 = J1 + 1

    # Compute required padding size
    c, h, w = img.shape
    pad = int(np.maximum(np.max(I2) - (h - 1), np.max(J2) - (w - 1)))

    # Prepare padding
    if isinstance(padding, (int, float)):
        padding_kwargs = {
            "borderType": cv2.BORDER_CONSTANT,
            "value": [float(padding)] * c,  # Add the value for each channel
        }
    else:
        padding_kwargs = {"borderType": getattr(cv2, padding)}

    # Pad image (use OpenCV here to leverage their complex padding modes)
    img_padded = cv2.copyMakeBorder(
        img.transpose((1, 2, 0)),  # (h, w, c) for OpenCV
        top=pad,
        bottom=pad,
        left=pad,
        right=pad,
        **padding_kwargs,
    )

    if len(img_padded.shape) == 2:  # OpenCV flattened our image
        img_padded = img_padded[:, :, np.newaxis]
    img_padded = img_padded.transpose((2, 0, 1))  # Back to (c, h, w)

    # Obtain pixel values for the interpolation corners
    Q11 = img_padded[:, pad + I1, pad + J1]  # (c, num_locations)
    Q21 = img_padded[:, pad + I1, pad + J2]
    Q12 = img_padded[:, pad + I2, pad + J1]
    Q22 = img_padded[:, pad + I2, pad + J2]

    # Weights
    w11 = (J2 - j_coords) * (I2 - i_coords)  # (num_locations,)
    w12 = (J2 - j_coords) * (i_coords - I1)
    w21 = (j_coords - J1) * (I2 - i_coords)
    w22 = (j_coords - J1) * (i_coords - I1)

    # Compute weighted sum
    return w11 * Q11 + w21 * Q21 + w12 * Q12 + w22 * Q22  # (c, num_locations)


def get_max_interpolation_grad(img):
    """Compute the maximum gradient of the interpolation transformation."""
    # From our equation 63, we need dI/du but dI/du = dI/dx since u = x + 0.5.
    dI_dx = img[:, :-1, 1:] - img[:, :-1, :-1]
    dI_dy = img[:, 1:, :-1] - img[:, :-1, :-1]

    common = img[:, :-1, :-1] + img[:, 1:, 1:] - img[:, :-1, 1:] - img[:, 1:, :-1]

    # We take the absolute value by triangle inequality because
    # we only need a majorant on the gradient (eq. 63 in our paper)
    dI_dx_maxes = np.maximum(np.abs(dI_dx), np.abs(dI_dx + common))
    dI_dy_maxes = np.maximum(np.abs(dI_dy), np.abs(dI_dy + common))

    dI_dx_max = dI_dx_maxes.max()
    dI_dy_max = dI_dy_maxes.max()

    # At each pixel location (i,j), we store the max. gradient [dI/dx, dI/dy]
    return np.array([[dI_dx_max, dI_dy_max]])  # (1 x 2)
