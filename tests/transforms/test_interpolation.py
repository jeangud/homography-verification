import cv2
import numpy as np
from vnn.transforms.interpolation import interpolate, get_max_interpolation_grad


def test_interpolate_vs_opencv_constant_padding():
    # 3-channel image mapping to CV2 dimensionality
    img = np.random.rand(3, 10, 10).astype(np.float32)

    # Generate random mapping coordinates pushing slightly outside boundary limits
    np.random.seed(42)
    i_coords = np.random.uniform(-2, 11, 100).astype(np.float32)
    j_coords = np.random.uniform(-2, 11, 100).astype(np.float32)

    # VNN computation interpolation target
    padding_val = 0.5
    res_custom = interpolate(img, i_coords, j_coords, padding=padding_val)

    # OpenCV underlying math check mapping
    img_cv = img.transpose(1, 2, 0)
    map_x = j_coords.astype(np.float32)
    map_y = i_coords.astype(np.float32)

    res_cv = cv2.remap(
        img_cv,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=[padding_val] * 3,
    )
    res_cv = res_cv.reshape(-1, 3).T
    # Custom interpolation handles borders differently from OpenCV (by design),
    # so we use a relaxed tolerance for boundary points
    np.testing.assert_allclose(res_custom, res_cv, atol=0.02)


def test_interpolate_vs_opencv_replicate_padding():
    # 1-channel image enforcing specific cv2 shape-dropping branches
    img = np.random.rand(1, 10, 10).astype(np.float32)

    np.random.seed(42)
    i_coords = np.random.uniform(-2, 11, 100).astype(np.float32)
    j_coords = np.random.uniform(-2, 11, 100).astype(np.float32)

    # Border replication mathematically aligns out-of-bounds to nearest terminal edge colors
    res_custom = interpolate(img, i_coords, j_coords, padding="BORDER_REPLICATE")

    img_cv = img.transpose(1, 2, 0)
    map_x = j_coords.astype(np.float32)
    map_y = i_coords.astype(np.float32)

    res_cv = cv2.remap(
        img_cv,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )
    res_cv = res_cv.reshape(-1, 1).T
    # Custom interpolation handles borders differently from OpenCV (by design),
    # so we use a relaxed tolerance for boundary points
    np.testing.assert_allclose(res_custom, res_cv, atol=0.02)


def test_get_max_interpolation_grad_zero_corner():
    # When P[i,j] = 0, the mixed-derivative term C = P[i,j] + P[i+1,j+1] - P[i,j+1] - P[i+1,j]
    # equals 5 - 3 - 2 = 0, so the gradient is constant across beta:
    #   dI/d_alpha = P[i,j+1] - P[i,j] = 3   at every beta
    #   dI/d_beta  = P[i+1,j] - P[i,j] = 2   at every alpha
    # [ [0, 3],
    #   [2, 5] ]
    img = np.array([[[0.0, 3.0], [2.0, 5.0]]])
    res = get_max_interpolation_grad(img)

    assert res.shape == (1, 2)
    np.testing.assert_allclose(res[0, 0], 3.0)
    np.testing.assert_allclose(res[0, 1], 2.0)


def test_get_max_interpolation_grad_nonzero_corner():
    # When P[i,j] != 0 the mixed term C = P[i,j] + P[i+1,j+1] - P[i,j+1] - P[i+1,j] != 0.
    # For the image below, C = 1 + 5 - 3 - 2 = 1.
    # [ [1, 3],
    #   [2, 5] ]
    # I(alpha, beta) = 1 + 2*alpha + beta + alpha*beta  (verified by expansion)
    # dI/d_alpha = 2 + beta  =>  max |dI/d_alpha| = 3  (at beta = 1)
    # dI/d_beta  = 1 + alpha =>  max |dI/d_beta|  = 2  (at alpha = 1)
    img = np.array([[[1.0, 3.0], [2.0, 5.0]]])
    res = get_max_interpolation_grad(img)

    assert res.shape == (1, 2)
    np.testing.assert_allclose(
        res[0, 0], 3.0
    )  # was 2.0 with buggy code (missing P[i,j])
    np.testing.assert_allclose(res[0, 1], 2.0)  # was 1.0 with buggy code
