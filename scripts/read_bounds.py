# Example usage:
#   $ python read_bounds.py bounds.pkl 11 18
#   Lower bound
#   ---
#   [slope, intercept, segment_start, segment_end]
#   [[1.86599274 0.27549021 0.         0.08726646]
#    [1.86599274 0.27549021 0.08726646 0.17453293]]
#
#   Upper bound
#   ---
#   [slope, intercept, segment_start, segment_end]
#   [[ 8.24485186  0.37549021  0.          0.08726646]
#    [-1.72469362  1.17299929  0.08726646  0.17453293]]

import argparse
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def read_bounds(path_bounds, i: int, j: int):
    with path_bounds.open("rb") as f:
        data = pickle.load(f)
        print(data.keys())
        bounds = data["bounds"]

    lb_lin_unsound = bounds["linear"]["unsound"]["lower"][i][j]
    ub_lin_unsound = bounds["linear"]["unsound"]["upper"][i][j]
    lb_lin_sound = bounds["linear"]["sound"]["lower"][i][j]
    ub_lin_sound = bounds["linear"]["sound"]["upper"][i][j]
    lb_pwl_unsound = bounds["pwl"]["unsound"]["lower"][i][j]
    ub_pwl_unsound = bounds["pwl"]["unsound"]["upper"][i][j]
    lb_pwl_sound = bounds["pwl"]["sound"]["lower"][i][j]
    ub_pwl_sound = bounds["pwl"]["sound"]["upper"][i][j]

    print("Sound bounds")
    print("---")
    print("Linear: [slope, intercept]")
    print(f"Lower: {lb_lin_sound}")
    print(f"Upper: {ub_lin_sound}")
    print("Piecewise linear (PWL) [slope, intercept, segment_start, segment_end]:")
    print(f"Lower: {lb_pwl_sound}")
    print(f"Upper: {ub_pwl_sound}")
    print()

    print("Unsound bounds")
    print("---")
    print("Linear: [slope, intercept]")
    print(f"Lower: {lb_lin_unsound}")
    print(f"Upper: {ub_lin_unsound}")
    print("Piecewise linear (PWL) [slope, intercept, segment_start, segment_end]:")
    print(f"Lower: {lb_pwl_unsound}")
    print(f"Upper: {ub_pwl_unsound}")

    # Get domain for the transform parameter
    kappa_0 = lb_pwl_sound[0][
        2
    ]  # Start of the first subdomain = overall interval start
    kappa_1 = lb_pwl_sound[-1][3]  # End of the last subdomain = overall interval end
    kappas = np.linspace(kappa_0, kappa_1, 100)

    plt.figure()

    # Lower linear
    w = lb_lin_sound[0]
    b = lb_lin_sound[1]
    lb = w * kappas + b
    plt.plot(kappas, lb, "--", label="Lower (linear)")

    # Upper linear
    w = ub_lin_sound[0]
    b = ub_lin_sound[1]
    ub = w * kappas + b
    plt.plot(kappas, ub, "--", label="Upper (linear)")

    # Lower PWL (equation y = w*kappa + b)
    # One line equation per subdomain of the overall interval
    w = lb_pwl_sound[:, 0]  # (num_subdomains, 1)
    b = lb_pwl_sound[:, 1]  # (num_subdomains, 1)
    lb = w[:, np.newaxis] * kappas + b[:, np.newaxis]
    lb = np.max(lb, axis=0)  # Proxy to computing the intersection by hand
    plt.plot(kappas, lb, label="Lower (PWL)")

    # Upper PWL (equation y = w*kappa + b)
    # One line equation per subdomain of the overall interval
    w_up = ub_pwl_sound[:, 0]  # (num_subdomains, 1)
    b_up = ub_pwl_sound[:, 1]  # (num_subdomains, 1)
    ub = w_up[:, np.newaxis] * kappas + b_up[:, np.newaxis]
    ub = np.min(ub, axis=0)
    plt.plot(kappas, ub, label="Upper (PWL)")

    plt.xlabel("Transform parameter $\kappa$")
    plt.ylabel("Pixel values")
    plt.title("Bounds Visualization")
    plt.grid()
    plt.legend()
    plt.show()


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Display bound coefficients for a given pixel."
    )
    parser.add_argument("file_path", type=Path, help="Path to the bounds.pkl file")
    parser.add_argument("pixel_x", type=int, help="X coordinate of the pixel")
    parser.add_argument("pixel_y", type=int, help="Y coordinate of the pixel")

    args = parser.parse_args(argv)

    read_bounds(args.file_path, args.pixel_x, args.pixel_y)


if __name__ == "__main__":
    main()
