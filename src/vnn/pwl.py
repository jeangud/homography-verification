"""Piecewise-linear (PWL) bounds computation."""

import logging
import time

from joblib import Parallel, delayed
import numpy as np

from . import linear_programming as lp
from .linear_programming import Solver
from .transforms.image import to_xy
from .transforms.transform_with_bounds import Transform, TransformWithBounds

LOGGER = logging.getLogger(__name__)


def calculate_bounds(
    img,
    transform_with_bounds: TransformWithBounds,
    padding,
    num_samples: int,
    lipschitz_error: float,
    num_init_splits: int,
    num_splits: int,
    num_subdomains: int,
    max_iterations: int,
    check_numerical: bool = True,
    num_jobs: int = 1,
    solver: Solver = Solver.GUROBI,
):
    """Takes a single image and returns a verification result."""
    if len(img.shape) > 3 or (img.shape[0] != 1 and img.shape[0] != 3):
        raise ValueError(
            "Image must be a single-channel (grayscale) or 3-channel (RGB) image."
        )

    LOGGER.info("Calculating unsound linear bounds")
    samples, params = generate_samples(
        img,
        transform_with_bounds.transform,
        lower_bound=transform_with_bounds.lower_bound,
        upper_bound=transform_with_bounds.upper_bound,
        num_samples=num_samples,
        padding=padding,
    )
    lb_unsound = fit_linear_bound(samples, params, is_upper_bound=False, solver=solver)
    ub_unsound = fit_linear_bound(samples, params, is_upper_bound=True, solver=solver)

    LOGGER.info("Adjusting to sound linear bounds")
    lb_sound, num_bab_lb = shift_linear_bound(
        lb_unsound,
        img,
        transform_with_bounds,
        lipschitz_error,
        num_init_splits,
        num_splits,
        num_subdomains,
        padding,
        is_upper_bound=False,
        max_iterations=max_iterations,
        check_numerical=check_numerical,
        num_jobs=num_jobs,
    )
    ub_sound, num_bab_ub = shift_linear_bound(
        ub_unsound,
        img,
        transform_with_bounds,
        lipschitz_error,
        num_init_splits,
        num_splits,
        num_subdomains,
        padding,
        is_upper_bound=True,
        max_iterations=max_iterations,
        check_numerical=check_numerical,
        num_jobs=num_jobs,
    )

    LOGGER.info("Calculating unsound piecewise-linear (PWL) bounds")
    lb_pwl_unsound = fit_pwl_bound(samples, params, is_upper_bound=False, solver=solver)
    ub_pwl_unsound = fit_pwl_bound(samples, params, is_upper_bound=True, solver=solver)

    LOGGER.info("Adjusting to sound piecewise-linear (PWL) bounds")
    lb_pwl_sound, num_bab_pwl_lb = shift_pwl_bound(
        lb_pwl_unsound,
        img,
        transform_with_bounds,
        lipschitz_error,
        num_init_splits,
        num_splits,
        num_subdomains,
        padding,
        is_upper_bound=False,
        max_iterations=max_iterations,
        check_numerical=check_numerical,
        num_jobs=num_jobs,
    )
    ub_pwl_sound, num_bab_pwl_ub = shift_pwl_bound(
        ub_pwl_unsound,
        img,
        transform_with_bounds,
        lipschitz_error,
        num_init_splits,
        num_splits,
        num_subdomains,
        padding,
        is_upper_bound=True,
        max_iterations=max_iterations,
        check_numerical=check_numerical,
        num_jobs=num_jobs,
    )

    return {
        "params": params,
        "samples": samples,
        "linear": {
            "unsound": {"lower": lb_unsound, "upper": ub_unsound},
            "sound": {
                "lower": lb_sound,
                "upper": ub_sound,
                "num_bab_lb": num_bab_lb,
                "num_bab_ub": num_bab_ub,
            },
        },
        "pwl": {
            "unsound": {"lower": lb_pwl_unsound, "upper": ub_pwl_unsound},
            "sound": {
                "lower": lb_pwl_sound,
                "upper": ub_pwl_sound,
                "num_bab_lb": num_bab_pwl_lb,
                "num_bab_ub": num_bab_pwl_ub,
            },
        },
    }


def fit_linear_bound(
    samples: list, params: list, is_upper_bound: bool, solver: Solver = Solver.GUROBI
):
    """Uses the given sample images to compODOe an unsound linear bound for the pixel values.

    See equation (9) in batten2024verification."""
    # Assemble linear program (LP) c @ x s.t. A @ x <= b
    # A (num_samples,2) We add 1 for the intercept
    A = np.column_stack((params, np.ones_like(params)))

    # c (2, 1) See equation (9) in batten2024verification
    c = -A.sum(axis=0)

    # For each pixel
    num_channels, num_rows, num_cols = samples[0].shape
    bounds = np.zeros((num_channels, num_rows, num_cols, 2))
    for h in range(num_channels):
        for i in range(num_rows):
            for j in range(num_cols):
                # b (num_samples, 1) These are the pixel values
                b = samples[:, h, i, j]

                # Solve LP
                if is_upper_bound:
                    x_star = lp.solve(-c, -A, -b, solver=solver)
                else:
                    x_star = lp.solve(c, A, b, solver=solver)

                bounds[h, i, j] = x_star

    return bounds


def shift_linear_bound(
    linear_bounds: np.ndarray,
    img,
    transform_with_bounds: TransformWithBounds,
    # Parameters
    lipschitz_error: float,
    num_init_splits: int,
    num_splits: int,
    num_samples_per_subdomain: int,
    padding_value,
    is_upper_bound: bool,
    max_iterations: int,
    check_numerical: bool = False,
    num_jobs: int = 1,
):
    """Branch-and-bound Lipschitz optimization procedure.

    Shifts the given unsound linear bounds to make them sound,
    through Lipschitz optimization."""
    start = time.time()
    transform = transform_with_bounds.transform
    lower_bound = transform_with_bounds.lower_bound
    upper_bound = transform_with_bounds.upper_bound

    # For each pixel
    num_channels, num_rows, num_cols = img.shape
    max_violations = np.zeros((num_channels, num_rows, num_cols))

    def process_pixel(c, i, j):
        LOGGER.debug("Channel %d Pixel (%d, %d)", c, i, j)
        # Get pixel information
        x, y = to_xy(i, j)

        # Prepare violation function for this pixel
        w, b = linear_bounds[c, i, j]

        def f_uv(img_samples, params, is_upper_bound):
            """Vectorized violation function."""
            values = w * params + b - img_samples[:, c, i, j]
            return -values if is_upper_bound else values

        # TODO(jgd) Change subdomains to priority queue? Vectorize for speed?
        # From [PWL, p.9]We can ignore any sub-domain, κ^n_1 , of κ_1 where
        # the function bound f_bound in κ^n_1 is smaller than a maximum value candidate
        #  f_max in any other sub-domain.
        subdomains = split((lower_bound, upper_bound), num_init_splits)

        f_max = 0  # NOTE: f is always >= 0, by construction of the bound
        it = 0
        num_bab = num_init_splits
        while len(subdomains) > 0 and it < max_iterations:
            subdomain = subdomains.pop(0)

            # Maximum observed violation from sampling inside the subdomain
            samples, params = generate_samples(
                img,
                transform,
                lower_bound=subdomain[0],
                upper_bound=subdomain[1],
                num_samples=num_samples_per_subdomain,
                padding=padding_value,
            )
            f_samples = f_uv(samples, params, is_upper_bound)
            f_max_over_subdomain = np.max(f_samples)
            f_max = np.max((f_max, f_max_over_subdomain))

            # Bound the Lipschitz constant of the bound error (paper eq. 63)
            #   sup|∇f| ≤ |w*| + sup|∇G|  where  sup|∇G| = L_G
            L_G = transform.get_lipschitz_analytical(img, x, y, subdomain)
            L_m = np.abs(w) + L_G
            if check_numerical:
                L_G_numerical = transform.get_lipschitz_numerical(img, x, y, subdomain)
                # We use a small margin here to account for floating-point errors and numerical estimation noise
                assert L_G >= L_G_numerical * 0.99, (
                    f"Analytical Lipschitz {L_G} < numerical {L_G_numerical}"
                )

            # Compute a majorant of the violation function
            f_bound = 0.5 * (
                L_m * np.abs(subdomain[1] - subdomain[0]) + f_samples[0] + f_samples[-1]
            )

            # If the bound is not sound, split the subdomain further
            is_converged = (
                # Found valid upper bound
                (f_bound <= f_max_over_subdomain + lipschitz_error)
                # Some other subdomains already have larger violation
                or (f_bound < f_max)
                # Boundary is always respected, no violations
                or (f_bound <= 0)
            )
            if not is_converged:
                subdomains.extend(split(subdomain, num_splits))
                num_bab += num_splits - 1

            it += 1

        if it >= max_iterations:
            raise RuntimeError(
                "Exceeded max. number of Branch-and-Bound iterations."
                f" ({max_iterations} iterations). "
                "Perturbation domain might be too large?"
            )

        return c, i, j, f_max + lipschitz_error, num_bab

    results = Parallel(n_jobs=num_jobs)(
        delayed(process_pixel)(c, i, j)
        for c in range(num_channels)
        for i in range(num_rows)
        for j in range(num_cols)
    )

    bab_iterations = []
    for c, i, j, max_violation, num_bab in results:
        max_violations[c, i, j] = max_violation
        bab_iterations.append((i, j, c, num_bab))

    # Add sound offset (only on offset, not to the slope!)
    sound_bounds = linear_bounds.copy()
    if is_upper_bound:
        sound_bounds[:, :, :, -1] += max_violations
    else:
        sound_bounds[:, :, :, -1] -= max_violations

    LOGGER.info("Time elapsed: %.2f seconds", time.time() - start)

    return sound_bounds, bab_iterations


def fit_pwl_bound(
    samples,
    params,
    is_upper_bound: bool,
    num_segments: int = 2,
    solver: Solver = Solver.GUROBI,
):
    """Takes the upper and lower linear bounds and generates a second either upper or
    lower linear bound to combine with the bounds to make one PWL and a non-PWL
    bound."""
    segments = compute_split_segments(params, num_segments)

    # From batten2024verification: we minimize approximation error (c) over subdomain
    # only, but enforce the constraints (A and b) over the whole domain.
    A = np.column_stack((params, np.ones_like(params)))  # (num_samples, 2) Whole domain

    # For each pixel, fit a new linear bound on each segment
    num_chans, num_rows, num_cols = samples[0].shape
    # For each segment we store: [slope, intercept, segment_start, segment_end]
    pwl_bounds = np.zeros(
        (num_chans, num_rows, num_cols, num_segments, 4)
    )  # (c, h, w, num_segments, 4)
    for h in range(num_chans):
        for i in range(num_rows):
            for j in range(num_cols):
                for k, segment in enumerate(segments):
                    inside = (segment[0] <= params) & (params <= segment[1])
                    c = -A[inside].sum(axis=0)  # (num_inside, 2) Segment only

                    b = samples[:, h, i, j]  # (num_samples,) Whole domain

                    if is_upper_bound:
                        x_star = lp.solve(-c, -A, -b, solver=solver)
                    else:
                        x_star = lp.solve(c, A, b, solver=solver)

                    # Store subdomain bounds as well
                    pwl_bounds[h, i, j, k] = *x_star, *segment

    return pwl_bounds


def shift_pwl_bound(
    pwl_bounds,
    img,
    transform_with_bounds: TransformWithBounds,
    # Parameters
    lipschitz_error: float,
    num_init_splits: int,
    num_splits: int,
    num_samples_per_subdomain: int,
    padding_value,
    is_upper_bound: bool,
    max_iterations: int,
    check_numerical: bool = False,
    num_jobs: int = 1,
):
    """Shifts the given unsound PWL bounds to make them sound."""
    start = time.time()
    transform = transform_with_bounds.transform
    lower_bound = transform_with_bounds.lower_bound
    upper_bound = transform_with_bounds.upper_bound

    # For each pixel
    num_channels, num_rows, num_cols = img.shape
    max_violations = np.zeros((num_channels, num_rows, num_cols))

    def process_pixel(c, i, j):
        LOGGER.debug("Channel %d Pixel (%d, %d)", c, i, j)
        # Get pixel information
        x, y = to_xy(i, j)

        # Prepare violation function for this pixel
        w = pwl_bounds[c, i, j, :, 0]
        b = pwl_bounds[c, i, j, :, 1]

        def f_uv(imgs, params, is_upper_bound):
            """Vectorized violation function."""
            values = w[:, np.newaxis] * params + b[:, np.newaxis] - imgs[:, c, i, j]
            values = -values if is_upper_bound else values

            # Take the best piece-wise part
            # Not equivalent to splitting at the designed segment boundary, but since
            # the cost function also requires satisfying constraints *over the whole domain*,
            # we know there is no bound violation.
            return values.min(axis=0) if is_upper_bound else values.max(axis=0)

        subdomains = split((lower_bound, upper_bound), num_init_splits)

        f_max = 0  # NOTE: f is always >= 0, by construction of the bound
        it = 0
        num_bab = num_init_splits
        while len(subdomains) > 0 and it < max_iterations:
            subdomain = subdomains.pop(0)

            # Maximum observed violation from sampling inside the subdomain
            samples, params = generate_samples(
                img,
                transform,
                lower_bound=subdomain[0],
                upper_bound=subdomain[1],
                num_samples=num_samples_per_subdomain,
                padding=padding_value,
            )
            f_samples = f_uv(samples, params, is_upper_bound)
            f_max_over_subdomain = np.max(f_samples)
            f_max = np.max((f_max, f_max_over_subdomain))

            # Bound the Lipschitz constant of the bound error (paper eq. 63)
            #   sup|∇J| ≤ |w*| + sup|∇G|  where  w* = max_k|w_k|,  sup|∇G| = L_G
            L_G = transform.get_lipschitz_analytical(img, x, y, subdomain)
            L_m = np.max(np.abs(w)) + L_G
            if check_numerical:
                L_G_numerical = transform.get_lipschitz_numerical(img, x, y, subdomain)
                # We use a small margin here to account for floating-point errors and numerical estimation noise
                assert L_G >= L_G_numerical * 0.99, (
                    f"Analytical Lipschitz {L_G} < numerical {L_G_numerical}"
                )

            # Compute a majorant of the violation function
            f_bound = 0.5 * (
                L_m * np.abs(subdomain[1] - subdomain[0]) + f_samples[0] + f_samples[-1]
            )

            # If the bound is not sound, split the subdomain further
            is_converged = (
                # Found valid upper bound
                (f_bound <= f_max_over_subdomain + lipschitz_error)
                # Some other subdomains already have larger violation
                or (f_bound < f_max)
                # Boundary is always respected, no violations
                or (f_bound <= 0)
            )
            if not is_converged:
                subdomains.extend(split(subdomain, num_splits))
                num_bab += num_splits - 1

            it += 1

        if it >= max_iterations:
            raise RuntimeError(
                "Exceeded max. number of Branch-and-Bound iterations."
                f" ({max_iterations} iterations). "
                "Perturbation domain might be too large?"
            )

        return c, i, j, f_max + lipschitz_error, num_bab

    results = Parallel(n_jobs=num_jobs)(
        delayed(process_pixel)(c, i, j)
        for c in range(num_channels)
        for i in range(num_rows)
        for j in range(num_cols)
    )

    bab_iterations = []
    for c, i, j, max_violation, num_bab in results:
        max_violations[c, i, j] = max_violation
        bab_iterations.append((i, j, c, num_bab))

    # Add sound offset (only on offset, not to the slope!)
    sound_bounds = pwl_bounds.copy()
    num_pwl_bounds = pwl_bounds.shape[3]
    for k in range(num_pwl_bounds):
        if is_upper_bound:
            # Reminder: PWL format is [slope, intercept, segment_start, segment_end]
            # Contrary to the PWL paper, we return 2 piecewise linear bounds,
            # as opposed to 1 linear and 1 PWL.
            sound_bounds[:, :, :, k, 1] += max_violations
        else:
            sound_bounds[:, :, :, k, 1] -= max_violations

    LOGGER.info("Time elapsed: %.2f seconds", time.time() - start)

    return sound_bounds, bab_iterations


def compute_split_segments(params, num_segments):
    """Compute the best split point for the segments of the piece-wise linear bound."""
    return split((params[0], params[-1]), num_segments)


def split(subdomain, num_splits):
    """Splits the given subdomain into `num_splits` subdomains."""
    subdomains_bounds = np.linspace(subdomain[0], subdomain[1], num_splits + 1)
    # NOTE: we return a `list` because we want to use pop() later
    return list(np.column_stack((subdomains_bounds[:-1], subdomains_bounds[1:])))


def generate_samples(
    img,
    transform: Transform,
    lower_bound: float,
    upper_bound: float,
    num_samples: int,
    padding,
):
    """Generate image samples based on the given parameter space."""
    parameters = np.linspace(lower_bound, upper_bound, num_samples)
    samples = transform.apply(img, parameters, padding)
    return samples, parameters
