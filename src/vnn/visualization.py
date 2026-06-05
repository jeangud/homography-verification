"""Utilities for plotting and visualizations."""

import logging
import shutil

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patches
from mpl_toolkits.axes_grid1.inset_locator import mark_inset, zoomed_inset_axes
from scipy import constants

from vnn.pwl import generate_samples
from vnn.transforms import (
    Transform,
    TransformWithBounds,
    Rotation,
    HomographyRoll,
    HomographyPitch,
    HomographyYaw,
)


LOGGER = logging.getLogger(__name__)

# Material Design Colors
# From: https://materialui.co/colors
COLOR_HIGHLIGHT = "#F44336"
COLOR_Q11 = "#4CAF50"
COLOR_Q12 = "#FFC107"
COLOR_Q21 = "#F44336"
COLOR_Q22 = "#2196F3"

COLOR_PIXEL_VALUES = "magenta"
COLOR_LINEAR_CONSTRAINTS = "#9E9E9E"
COLOR_PWL_CONSTRAINTS = "#009688"

# LaTeX document width, in points
# TEXTWIDTH_PT = 397.48499  # Neurips
TEXTWIDTH_PT = 496.85625 / 2  # CVPR Use `\the\textwidth` in LaTeX to print this value


def setup_latex():
    """Add matplotlib options for LaTeX rendering.

    Falls back to non-LaTeX serif fonts if LaTeX is not installed.
    """
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 9,
            "axes.labelsize": 9,
            "axes.titlesize": 9,
            "legend.fontsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
        }
    )

    if shutil.which("latex") is None:
        LOGGER.warning(
            "LaTeX is not available — figures will use default fonts. "
            "Install LaTeX for better rendering: https://www.latex-project.org/get/"
        )
        return

    plt.rc("text", usetex=True)
    plt.rc("text.latex", preamble=r"\usepackage{gensymb}")


def get_figure_size(width_pt, fraction=1, aspect_ratio=1):
    """Set figure dimensions to avoid scaling in LaTeX.

    Parameters
    ----------
    width: float or string
            Document width in points, or string of predined document type
    fraction: float, optional
            Fraction of the width which you wish the figure to occupy
    subplots: array-like, optional
            The number of rows and columns of subplots.
    Returns
    -------
    fig_dim: tuple
            Dimensions of figure in inches
    """
    fig_width_pt = width_pt * fraction

    inches_per_pt = (
        constants.point / constants.inch
    )  # meters_per_pt / meters_per_inch = inch / pt
    fig_width_in = fig_width_pt * inches_per_pt

    fig_height_in = fig_width_in / aspect_ratio

    return (fig_width_in, fig_height_in)


def plot_samples(samples, num_images: int = 9, i=None, j=None):
    """Plots a few image samples over the given range of parameters"""
    setup_latex()
    fig_size = get_figure_size(width_pt=TEXTWIDTH_PT)
    fig = plt.figure(figsize=fig_size)

    # Pick `num_images` samples to plot
    idx_images = np.linspace(0, len(samples) - 1, num_images, dtype=int)

    # Keep color scale constant for each sample
    vmin = samples[0].min()
    vmax = samples[0].max()

    for idx_grid, idx_image in enumerate(idx_images):
        plt.subplot(1, num_images, idx_grid + 1)
        plt.imshow(
            samples[idx_image].transpose(1, 2, 0), cmap="gray", vmin=vmin, vmax=vmax
        )
        plt.axis("off")

        # Highlight pixel of interest if needed
        if i is not None and j is not None:
            plt.plot(j, i, "rs")

    plt.tight_layout()

    return fig


def plot_transform(
    img: np.ndarray,
    transform: Transform,
    p_min,
    p_max,
    i: float,
    j: float,
    padding_value,
    num_samples: int = 100,
) -> None:
    """Plots the effect of a geometric transform over the values of a pixel."""
    setup_latex()
    text_width = 2 * TEXTWIDTH_PT  # We use 2-column-wide figure for this one
    fig_size = get_figure_size(width_pt=text_width)
    fig = plt.figure(figsize=fig_size)

    tfwb = TransformWithBounds(transform, p_min, p_max)

    # Save range of original image for consistent color scale
    img_original = img.transpose((1, 2, 0))
    img_original = img_original[..., 0]  # Force grayscale for this visualization
    vmin = img_original.min()
    vmax = img_original.max()

    ###########################################################################
    # Start from the transformed image
    ###########################################################################
    ax1 = plt.subplot(131)
    samples, params = generate_samples(
        img,
        tfwb.transform,
        lower_bound=tfwb.lower_bound,
        upper_bound=tfwb.upper_bound,
        num_samples=num_samples,
        padding=padding_value,
    )
    img_transformed = samples[-1].transpose((1, 2, 0))
    img_transformed = img_transformed[..., 0]  # Force grayscale for this visualization
    ax1.imshow(img_transformed, cmap="gray", vmin=vmin, vmax=vmax)

    # Highlight the pixel of interest
    h, w = img.shape[-2:]
    ax1.plot(
        j,
        i,
        "s",
        color=COLOR_HIGHLIGHT,
        markerfacecolor="none",
        label="Pixel of interest",
    )

    # Plot pixel locations as parameter varies to current value
    Is, Js = transform.get_original_pixel_locations(img, params)
    Is = Is.reshape(num_samples, h, w)
    Js = Js.reshape(num_samples, h, w)
    i_values = Is[:, i, j]
    j_values = Js[:, i, j]
    c = 0
    pixel_values = samples[:, c, i, j]

    i0 = i_values[-1]
    j0 = j_values[-1]
    ax1.plot(j_values, i_values, color=COLOR_HIGHLIGHT, label="Pixel trajectory")
    ax1.plot(
        j0,
        i0,
        "x",
        color=COLOR_HIGHLIGHT,
        markerfacecolor="none",
        label="Original position",
    )

    # ax1.set_title('Transformed')
    ax1.legend(bbox_to_anchor=(0, 1.15), loc="lower left")
    ax1.xaxis.tick_top()
    ax1.xaxis.set_label_position("top")

    ###########################################################################
    # Link to original image
    ###########################################################################
    ax2 = plt.subplot(132)
    ax2.imshow(img_original, cmap="gray", vmin=vmin, vmax=vmax)
    (h_path,) = ax2.plot(
        j_values, i_values, "-", color=COLOR_PIXEL_VALUES, label="Pixels spanned"
    )

    ax2.plot(
        j0,
        i0,
        "x",
        color=COLOR_HIGHLIGHT,
        markerfacecolor="none",
        label=f"Pixel ({i0:.1f},{j0:.1f})",
    )

    # ax2.set_title('Original')
    ax2.xaxis.tick_top()
    ax2.xaxis.set_label_position("top")
    ax2.yaxis.set_visible(False)

    ###########################################################################
    # Plot pixel values
    ###########################################################################
    ax3 = plt.subplot(133)
    # ax3.set_title('Pixel values')
    ax3.plot(params, pixel_values, color=COLOR_PIXEL_VALUES)

    ax3.set_xlabel(r"Transform parameter $\kappa$")
    ax3.set_ylabel("Pixel value")
    ax3.yaxis.tick_right()
    ax3.yaxis.set_label_position("right")
    ax3.grid()
    ax3.set_ylim(0, 1)

    # Fix aspect ratio to match the height of the images
    aspect_ratio_plot = np.diff(ax3.get_xlim())[0] / np.diff(ax3.get_ylim())[0]
    aspect_ratio_img = np.abs(np.diff(ax1.get_xlim())[0] / np.diff(ax1.get_ylim())[0])
    ax3.set_aspect(aspect_ratio_plot / aspect_ratio_img)

    # Do this before adding the cross-axes mess
    fig.tight_layout(pad=0)

    ###########################################################################
    # Add zoom area
    ###########################################################################
    # Add link between plots
    con1 = patches.ConnectionPatch(
        axesA=ax1,
        xyA=(j0, i0),
        coordsA="data",
        axesB=ax2,
        xyB=(j0, i0),
        coordsB="data",
        linestyle="--",
        color="gray",
    )
    ax2.add_artist(con1)

    con2 = patches.ConnectionPatch(
        axesA=ax1,
        xyA=(j0, i0),
        coordsA="data",
        axesB=ax2,
        xyB=(j0, i0),
        coordsB="data",
        linestyle="--",
        color="gray",
    )
    ax2.add_artist(con2)

    zoom = 4 * w / 32  # Zoom 4 works well for for 32x32 images
    axins = zoomed_inset_axes(
        ax2,
        zoom=zoom,
        loc="lower left",
        bbox_to_anchor=(0, 1.15),
        bbox_transform=ax3.transAxes,
        borderpad=0,
    )

    axins.imshow(img_original, cmap="gray", vmin=vmin, vmax=vmax)

    # Get interpolation corners
    x1 = np.floor(j0)
    y1 = np.floor(i0)
    x2 = j0 + 1
    y2 = i0 + 1

    # Plot interpolation areas
    axins.fill(
        [j0, x2, x2, j0], [i0, i0, y2, y2], color=COLOR_Q11, edgecolor="none", alpha=0.5
    )
    axins.fill(
        [j0, x2, x2, j0],
        [i0, i0, y1, y1],
        color=COLOR_Q12,
        edgecolor="none",
        alpha=0.5,
        label="Interpolation regions",
    )
    axins.fill(
        [j0, x1, x1, j0], [i0, i0, y2, y2], color=COLOR_Q21, edgecolor="none", alpha=0.5
    )
    (h_area,) = axins.fill(
        [j0, x1, x1, j0], [i0, i0, y1, y1], color=COLOR_Q22, edgecolor="none", alpha=0.5
    )
    axins.plot(x1, y1, "o", color=COLOR_Q11)
    axins.plot(x1, y2, "o", color=COLOR_Q12)
    axins.plot(x2, y1, "o", color=COLOR_Q21)
    (h_corner,) = axins.plot(x2, y2, "o", color=COLOR_Q22)

    (h_origin,) = axins.plot(
        j0,
        i0,
        "x",
        color=COLOR_HIGHLIGHT,
        markerfacecolor="none",
        label=f"Pixel ({i0:.1f},{j0:.1f})",
    )

    margin = 1.5
    xc = (x1 + x2) / 2
    yc = (y1 + y2) / 2
    axins.set_xlim(xc - margin, xc + margin)
    axins.set_ylim(yc - margin, yc + margin)
    axins.spines["bottom"].set_color("gray")
    axins.spines["left"].set_color("gray")
    axins.tick_params(color="gray", labelcolor="gray", grid_color="gray")
    axins.invert_yaxis()

    # Connect the zoom areas
    mark_inset(ax2, axins, loc1=2, loc2=4, fc="none", linestyle="--", edgecolor="gray")

    # Add this to the original legend
    ax2.legend(
        handles=[h_path, (h_area, h_corner), h_origin],
        labels=[h_path.get_label(), "Interpolation", h_origin.get_label()],
        bbox_to_anchor=(0, 1.15),
        loc="lower left",
    )

    # Remove ticks which might conflict with overlay
    # xlim = ax2.get_xlim()
    # xticks = ax2.get_xticks()
    # num_visible_ticks = int(.75 * len(xticks))
    # ax2.set_xticks(xticks[:num_visible_ticks])
    # ax2.set_xlim(xlim)  # Clip, to hide extra ticks added by matplotlib

    #######################################################################
    # Finalize figure
    #######################################################################
    # fig.subplots_adjust(left=.05, right=.95, wspace=0.05)
    fig.align_titles()

    return fig


def plot_bounds(
    c: int,
    i: int,
    j: int,
    samples: list,
    params: list,
    tfwb: TransformWithBounds,
    bounds: dict,
    alpha: float = 0.25,
    num_samples: int = 1_000,
):
    """Plots the different approximation bounds for the values of a pixel (i,j) on channel c."""
    setup_latex()
    textwidth_pt = 397.48499  # Somehow the Neurips values work better for us here (aspect ratio, etc.)
    fig_size = get_figure_size(width_pt=textwidth_pt, fraction=0.5, aspect_ratio=1.25)
    fig, ax = plt.subplots(figsize=fig_size)

    # Book-keeping
    lb_lin_unsound = bounds["linear"]["unsound"]["lower"]
    ub_lin_unsound = bounds["linear"]["unsound"]["upper"]
    lb_lin_sound = bounds["linear"]["sound"]["lower"]
    ub_lin_sound = bounds["linear"]["sound"]["upper"]
    lb_pwl_unsound = bounds["pwl"]["unsound"]["lower"]
    ub_pwl_unsound = bounds["pwl"]["unsound"]["upper"]
    lb_pwl_sound = bounds["pwl"]["sound"]["lower"]
    ub_pwl_sound = bounds["pwl"]["sound"]["upper"]

    # NOTE: see notations in batten2024verification

    # Resample parameter values to get precise PWL breakpoints
    kappas = np.linspace(params[0], params[-1], num_samples)

    # Use degrees if needed
    kappas_units = kappas  # Kappa values with unit adjusted if needed
    params_units = params
    tf_cls = type(tfwb.transform)
    if tf_cls in [Rotation, HomographyRoll, HomographyPitch, HomographyYaw]:
        xlabel = r"Transform parameter $\kappa$ (\degree)"
        kappas_units = np.rad2deg(kappas_units)
        params_units = np.rad2deg(params_units)
    else:
        xlabel = r"Parameter value $\kappa$"

    # Linear bound
    w_lo, b_lo = lb_lin_sound[c, i, j]
    w_hi, b_hi = ub_lin_sound[c, i, j]

    lb = w_lo * kappas + b_lo
    ub = w_hi * kappas + b_hi

    (h_lin_sound,) = ax.plot(
        kappas_units, lb, color=COLOR_LINEAR_CONSTRAINTS, label="Linear bound"
    )
    ax.plot(kappas_units, ub, color=COLOR_LINEAR_CONSTRAINTS)
    ax.fill_between(kappas_units, lb, ub, color=COLOR_LINEAR_CONSTRAINTS, alpha=alpha)

    # Linear unsound
    w_lo, b_lo = lb_lin_unsound[c, i, j]
    w_hi, b_hi = ub_lin_unsound[c, i, j]
    lb = w_lo * kappas + b_lo
    ub = w_hi * kappas + b_hi
    ax.plot(kappas_units, lb, color=COLOR_LINEAR_CONSTRAINTS, ls="--")
    ax.plot(kappas_units, ub, color=COLOR_LINEAR_CONSTRAINTS, ls="--")
    ax.fill_between(kappas_units, lb, ub, color=COLOR_LINEAR_CONSTRAINTS, alpha=alpha)

    # Piecewise linear unsound
    w_lo = lb_pwl_unsound[c, i, j, :, 0]  # (num_subdomains, 1)
    b_lo = lb_pwl_unsound[c, i, j, :, 1]  # (num_subdomains, 1)
    lb = (
        w_lo[:, np.newaxis] * kappas + b_lo[:, np.newaxis]
    )  # (num_subdomains, num_kappas)

    # We use max(lower_bounds) or min(upper_bounds), we don't actually use the segment
    # endpoints which were used to compute the piecewise pieces. This is valid because
    # the optimization cost function also requires the constraints to be satisfied over
    # the whole domain. Thus we know there is no bound violation.
    lb = lb.max(axis=0)  # (num_kappas,) Equation (4) in batten2024verification
    ax.plot(
        kappas_units, lb, color=COLOR_PWL_CONSTRAINTS, ls="-.", label="PWL (unsound)"
    )

    w_hi = ub_pwl_unsound[c, i, j, :, 0]  # (num_subdomains, 1)
    b_hi = ub_pwl_unsound[c, i, j, :, 1]  # (num_subdomains, 1)
    ub = (
        w_hi[:, np.newaxis] * kappas + b_hi[:, np.newaxis]
    )  # (num_subdomains, num_kappas)
    ub = ub.min(axis=0)  # (num_kappas,) Equation (4) in batten2024verification
    ax.plot(kappas_units, ub, color=COLOR_PWL_CONSTRAINTS, ls="-.")

    ax.fill_between(kappas_units, lb, ub, color=COLOR_PWL_CONSTRAINTS, alpha=alpha)

    # Piecewise linear sound
    w_lo = lb_pwl_sound[c, i, j, :, 0]  # (num_subdomains, 1)
    b_lo = lb_pwl_sound[c, i, j, :, 1]  # (num_subdomains, 1)
    lb = (
        w_lo[:, np.newaxis] * kappas + b_lo[:, np.newaxis]
    )  # (num_subdomains, num_kappas)
    lb = lb.max(axis=0)  # (num_kappas,) Equation (4) in batten2024verification
    (h_pwl_sound,) = ax.plot(
        kappas_units, lb, color=COLOR_PWL_CONSTRAINTS, label="Ours"
    )

    w_hi = ub_pwl_sound[c, i, j, :, 0]  # (num_subdomains, 1)
    b_hi = ub_pwl_sound[c, i, j, :, 1]  # (num_subdomains, 1)
    ub = (
        w_hi[:, np.newaxis] * kappas + b_hi[:, np.newaxis]
    )  # (num_subdomains, num_kappas)
    ub = ub.min(axis=0)  # (num_kappas,) Equation (4) in batten2024verification
    ax.plot(kappas_units, ub, color=COLOR_PWL_CONSTRAINTS)

    ax.fill_between(kappas_units, lb, ub, color=COLOR_PWL_CONSTRAINTS, alpha=alpha)

    # Actual pixel value samples
    pixel_values = samples[:, c, i, j]
    (h_samples,) = ax.plot(
        params_units,
        pixel_values,
        color=COLOR_PIXEL_VALUES,
        label="True function",
    )

    # Present legend order differently than plotting/layering order
    ax.legend(handles=[h_samples, h_lin_sound, h_pwl_sound])

    ax.grid()
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Pixel value")
    plt.tight_layout()

    return fig
