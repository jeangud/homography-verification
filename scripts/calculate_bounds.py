import argparse
import logging
import pickle
import time
from datetime import datetime
from pathlib import Path

import cv2
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from vnn import datasets
from vnn import utils
from vnn.linear_programming import Solver
from vnn.pwl import calculate_bounds, generate_samples
from vnn.transforms import (
    TransformType,
    TransformWithBounds,
    # Affine
    Rotation,
    Scale,
    ShearX,
    TranslateX,
    TranslateY,
    # Non-affine
    HomographyRoll,
    HomographyPitch,
    HomographyYaw,
    HomographyX,
    HomographyY,
    HomographyZ,
)
from vnn.visualization import plot_samples, plot_bounds

LOGGER = logging.getLogger(__name__)
DIR_RESULTS = Path("./bounds").resolve()

# Use Agg backend to avoid GUI
matplotlib.use("Agg")


def get_experiment_name(timestamp, dataset, transform_types, lower, upper, padding):
    """Generate a flat experiment directory name from CLI args.

    Examples:
        2026-04-23-14-30-00_MNIST_ROTATE_0.00_0.35_pad0.5
        2026-04-23-14-30-00_LARD_H_ROLL_0.00_0.00_pad0.5
    """
    tf_name = transform_types[0].name
    return f"{timestamp}_{dataset}_{tf_name}_{lower[0]:.2f}_{upper[0]:.2f}_pad{padding}"


def parse_arguments(argv=None):
    parser = argparse.ArgumentParser(
        description="Calculate image bounds based on given transformations.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    transform = parser.add_argument_group("Transformation")
    transform.add_argument(
        "--transformations",
        type=TransformType,
        nargs="+",
        action=utils.EnumAction,
        required=True,
        help="List of transformations",
    )
    transform.add_argument(
        "--lower",
        type=float,
        nargs="+",
        required=True,
        help="List of corresponding transformation lower bounds",
    )
    transform.add_argument(
        "--upper",
        type=float,
        nargs="+",
        required=True,
        help="List of corresponding transformation upper bounds",
    )
    parser.add_argument_group(transform)

    data = parser.add_argument_group("Data")
    data.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=datasets.get_dataset_choices(),
        help="Dataset to use",
    )
    group = data.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--image-number",
        type=int,
        help="Number of dataset images to process",
    )
    group.add_argument(
        "--path-images-csv",
        type=Path,
        help="Path to CSV file with indices of images to process",
    )
    data.add_argument(
        "--padding",
        type=str,
        default="0.5",
        help="Padding value (float/int) or method (BORDER_REPLICATE, BORDER_REFLECT, BORDER_WRAP) to use for padding when transforming images",
    )
    parser.add_argument_group(data)

    algorithm = parser.add_argument_group("Algorithm")
    algorithm.add_argument(
        "--num-samples",
        type=int,
        default=100,
        help="Number of samples used in initial approximation of pixel value curve",
    )
    algorithm.add_argument(
        "--num-subdomain-samples",
        type=int,
        default=10,
        help="Number of samples for each sub-domain during Lipschitz Optimization",
    )
    algorithm.add_argument(
        "--num-init-splits",
        type=int,
        default=124,
        help="Number of initial sud-domains splits for Lipschitz Optimization",
    )
    algorithm.add_argument(
        "--num-splits",
        type=int,
        default=2,
        help="Number of splits at each iteration of during Lipschitz Optimization",
    )
    algorithm.add_argument(
        "--lipschitz-error",
        type=float,
        default=0.05,
        help="Final error allowed during Lipschitz Optimization",
    )
    algorithm.add_argument(
        "--max-bab-iter",
        type=int,
        default=1_000,
        help="Max. number of branch-and-bound (BaB) iterations",
    )
    algorithm.add_argument(
        "--solver",
        type=Solver,
        default=Solver.GUROBI,
        action=utils.EnumAction,
        help="LP solver to use",
    )
    parser.add_argument_group(algorithm)

    other = parser.add_argument_group("Other")
    other.add_argument(
        "--load-bounds", action="store_true", help="Load existing bounds if available"
    )
    other.add_argument(
        "--num-jobs", type=int, default=1, help="Number of parallel jobs to use"
    )
    other.add_argument(
        "--plot",
        action="store_true",
        help="Generate plots",
    )
    other.add_argument(
        "--logging",
        default=logging.getLevelName(logging.INFO),
        choices=utils.get_logging_levels(),  # Sort by severity
        help="Set logging level",
    )
    parser.add_argument_group(other)

    flags = parser.parse_args(argv)

    # Parse padding (either float, or OpenCV padding method)
    try:
        flags.padding = float(flags.padding)
    except ValueError:
        if not hasattr(cv2, flags.padding):
            raise ValueError(f"Invalid OpenCV padding method: {flags.padding}")

    return flags


def parse_transform(transform_types, lower, upper, dataset):
    assert len(transform_types) == len(lower) == len(upper), (
        "Number of transformations, lower bounds and upper bounds must be the same"
    )
    assert len(transform_types) == 1, "Only one transformation is supported"

    # Get a sample image from the dataset
    img = dataset[0][0]
    h, w = img.shape[-2:]
    xc = w / 2
    yc = h / 2

    # Adjust camera parameters based on dataset
    if isinstance(dataset, datasets.MNIST):
        focal_length_px = 10
        z = -10
    elif isinstance(dataset, datasets.CIFAR10):
        focal_length_px = 10
        z = -2
    elif isinstance(dataset, datasets.MetaRoom):
        focal_length_px = w
        z = -5
    elif isinstance(dataset, datasets.GTSRB):
        focal_length_px = w
        z = -5
    elif isinstance(dataset, datasets.LARD):
        focal_length_px = w
        z = -5
    else:
        raise NotImplementedError(f"Dataset {dataset} has no custom camera parameters")

    tf_type = transform_types[0]
    if tf_type == TransformType.ROTATE:
        tf = Rotation(x0=xc, y0=yc)
    elif tf_type == TransformType.SCALE:
        tf = Scale()
    elif tf_type == TransformType.SHEAR_X:
        tf = ShearX(y0=yc)
    elif tf_type == TransformType.TRANSLATE_X:
        tf = TranslateX()
    elif tf_type == TransformType.TRANSLATE_Y:
        tf = TranslateY()
    elif tf_type == TransformType.H_ROLL:
        tf = HomographyRoll(xc=xc, yc=yc)
    elif tf_type == TransformType.H_PITCH:
        tf = HomographyPitch(f=focal_length_px, xc=xc, yc=yc)
    elif tf_type == TransformType.H_YAW:
        tf = HomographyYaw(f=focal_length_px, xc=xc, yc=yc)
    elif tf_type == TransformType.H_X:
        tf = HomographyX(f=focal_length_px, xc=xc, yc=yc, z=z)
    elif tf_type == TransformType.H_Y:
        tf = HomographyY(f=focal_length_px, xc=xc, yc=yc, z=z)
    elif tf_type == TransformType.H_Z:
        tf = HomographyZ(f=focal_length_px, xc=xc, yc=yc, z=z)
    else:
        raise NotImplementedError(f"Transformation {tf_type} is not supported")

    return TransformWithBounds(tf, lower[0], upper[0])


def main(argv=None):
    args = parse_arguments(argv)

    # Create experiment directory and setup logging
    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    dir_experiment = DIR_RESULTS / get_experiment_name(
        timestamp,
        args.dataset,
        args.transformations,
        args.lower,
        args.upper,
        args.padding,
    )
    dir_experiment.mkdir(parents=True, exist_ok=True)

    path_log = dir_experiment / f"{timestamp}.log"
    utils.setup_logging(level=args.logging, path_log=path_log)
    LOGGER.info("Log file: %s", path_log)
    LOGGER.info("User arguments: %s", args)
    LOGGER.info("Saving results under %s", dir_experiment)

    LOGGER.info("Loading dataset...")
    dataset_cls = getattr(datasets, args.dataset)
    dataset = dataset_cls()

    LOGGER.info("Loading image indices...")
    if args.image_number is not None:
        image_indices = list(range(args.image_number + 1))
    elif args.path_images_csv:
        path_images_csv = args.path_images_csv.expanduser().resolve()
        with path_images_csv.open() as f:
            image_indices = [int(line) for line in f]
    else:
        raise ValueError("Either --image-number or --path-images-csv must be provided")

    LOGGER.info("Parsing transforms...")
    tfwb = parse_transform(args.transformations, args.lower, args.upper, dataset)

    # Create directory structure
    dir_plots = dir_experiment / "plots"
    dir_plots.mkdir(parents=True, exist_ok=True)
    dir_timing = dir_experiment / "timing"
    dir_timing.mkdir(parents=True, exist_ok=True)
    dir_bab = dir_experiment / "bab"
    dir_bab.mkdir(parents=True, exist_ok=True)

    LOGGER.info("Iterating over images")
    LOGGER.info("Processing %d images: %s", len(image_indices), image_indices)

    completed_images = []
    failed_images = []

    with logging_redirect_tqdm():
        pbar = tqdm(image_indices)
        for idx_img in pbar:
            try:
                pbar.set_description(f"Processing image {args.dataset}[{idx_img}]")
                LOGGER.info("Starting processing of image %d", idx_img)
                img = dataset[idx_img][0].numpy()
                assert len(img.shape) == 3, "Image shape is not (C,H,W)"

                # Plot samples
                LOGGER.info("Plotting samples")
                dir_img = dir_plots / str(idx_img)
                dir_img.mkdir(parents=True, exist_ok=True)
                samples, params = generate_samples(
                    img,
                    tfwb.transform,
                    lower_bound=tfwb.lower_bound,
                    upper_bound=tfwb.upper_bound,
                    num_samples=args.num_samples,
                    padding=args.padding,
                )

                fig = plot_samples(samples)
                fig.savefig(dir_img / "samples.pdf", bbox_inches="tight", pad_inches=0)
                plt.close(fig)

                path_bounds = dir_experiment / f"{idx_img}.pkl"
                if args.load_bounds and path_bounds.exists():
                    LOGGER.info("Loading existing bounds from %s", path_bounds)
                    with open(path_bounds, "rb") as f:
                        bounds = pickle.load(f)["bounds"]
                else:
                    LOGGER.info("Calculating bounds")
                    start_time = time.time()
                    bounds = calculate_bounds(
                        img=img,
                        transform_with_bounds=tfwb,
                        padding=args.padding,
                        num_samples=args.num_samples,
                        lipschitz_error=args.lipschitz_error,
                        num_init_splits=args.num_init_splits,
                        num_splits=args.num_splits,
                        num_subdomains=args.num_subdomain_samples,
                        max_iterations=args.max_bab_iter,
                        num_jobs=args.num_jobs,
                        solver=args.solver,
                    )
                    elapsed_time = time.time() - start_time
                    pd.DataFrame({"time_s": [elapsed_time]}).to_csv(
                        dir_timing / f"{args.num_init_splits}_{idx_img}.csv",
                        index=False,
                    )

                    LOGGER.info("Saving bounds to %s", path_bounds)
                    result = {
                        "dataset": args.dataset,
                        "idx_img": idx_img,
                        "transform_with_bounds": str(tfwb),
                        "bounds": bounds,
                    }
                    with path_bounds.open("wb") as f:
                        pickle.dump(result, f)

                    df = pd.DataFrame(
                        bounds["linear"]["sound"]["num_bab_lb"]
                        + bounds["linear"]["sound"]["num_bab_ub"]
                        + bounds["pwl"]["sound"]["num_bab_lb"]
                        + bounds["pwl"]["sound"]["num_bab_ub"],
                        columns=["i", "j", "c", "num_bab"],
                    )
                    df.to_csv(
                        dir_bab / f"{args.num_init_splits}_{idx_img}.csv",
                        index=False,
                    )

                if args.plot:
                    LOGGER.info("Plotting bounds")
                    dir_bounds = dir_img / "bounds"
                    dir_bounds.mkdir(parents=True, exist_ok=True)

                    def save_plot(c, i, j):
                        fig = plot_bounds(c, i, j, samples, params, tfwb, bounds)
                        fig.savefig(
                            dir_bounds / f"{c}_{i}_{j}.pdf",
                            bbox_inches="tight",
                            pad_inches=0,
                        )
                        plt.close(fig)

                    num_channels, num_rows, num_cols = img.shape
                    Parallel(n_jobs=args.num_jobs)(
                        delayed(save_plot)(c, i, j)
                        for c in range(num_channels)
                        for i in range(num_rows)
                        for j in range(num_cols)
                    )

                completed_images.append(idx_img)
                LOGGER.info("Successfully completed processing of image %d", idx_img)

            except Exception as e:
                failed_images.append(idx_img)
                LOGGER.error("Failed to process image %d: %s", idx_img, str(e))
                LOGGER.exception("Exception details:")
                # Continue with the next image instead of stopping
                continue

    LOGGER.info("Processing completed!")
    LOGGER.info(
        "Successfully processed %d images: %s", len(completed_images), completed_images
    )
    if failed_images:
        LOGGER.warning(
            "Failed to process %d images: %s", len(failed_images), failed_images
        )
    LOGGER.info("Log file saved at %s", path_log)
    LOGGER.info("Results saved under %s", dir_experiment)


if __name__ == "__main__":
    main()
