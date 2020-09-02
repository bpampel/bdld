#!/usr/bin/env python3
import argparse
from typing import Optional, Union

import numpy as np
import matplotlib.pyplot as plt

from bdld.potential import Potential


def gaussian(x: Union[float, np.ndarray], mu: float, sigma: float) -> float:
    """Standard normal distribution"""
    return (
        1 / (np.sqrt(2 * np.pi) * sigma) * np.exp(-((x - mu) ** 2) / (2 * sigma ** 2))
    )


def get_prob(grid: np.ndarray, kt: float) -> np.ndarray:
    """Get probability of double well potential"""
    pot = Potential(np.array([0, 0.0, -4, 0, 1]))
    prob = np.exp(-pot.calculate_reference(grid) / kt)
    # normalize although that doesn't matter for what is done here
    prob /= np.sum(prob) * (grid[1] - grid[0])
    return prob


def plot_conv(
    grid: np.ndarray,
    gaussgrid: np.ndarray,
    prob: np.ndarray,
    conv: np.ndarray,
    gauss: np.ndarray,
    filename: Optional[str],
) -> None:
    """Plot the convolution figures"""
    fig = plt.figure(figsize=(8, 4), dpi=100)
    ax = plt.axes()
    ax.plot(gaussgrid, gauss, label="K(x)")
    ax.plot(grid, prob, label="π(x)")
    ax.plot(grid, conv, label="K*π")
    ax.legend()
    if filename is not None:
        fig.savefig(filename)
    else:
        plt.show()


def parse_cliargs() -> argparse.Namespace:
    """Use argparse to get cli arguments
    :return: args: Namespace with cli arguments"""

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-kT",
        "--temp",
        type=float,
        dest="kt",
        help="Energy (in units of kT) of the FES file",
        required=True,
    )
    parser.add_argument(
        "-bw",
        "--kernel-bandwidth",
        type=float,
        dest="bw",
        help="Bandwidth for gaussian kernels",
        required=True,
    )
    parser.add_argument(
        "--grid-spacing",
        type=float,
        dest="grid_spacing",
        help="Spacing of the grids used for evaluation. \
        By default it is 0.02 except but at most half the kernel bandwidth",
    )
    parser.add_argument(
        "--conv-image",
        type=str,
        dest="conv_image",
        help="Name for the plot of the conv image",
    )
    parser.add_argument(
        "--show-image",
        action="store_true",
        dest="show_image",
        help="Also show the image directly.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_cliargs()
    kt = args.kt
    bw = args.bw
    grid_spacing = args.grid_spacing
    if grid_spacing is None:
        if 0.5 * bw < 0.02:  # at least 20 points for gaussian
            grid_spacing = 0.5 * bw
        else:
            grid_spacing = 0.02
    conv_image = args.conv_image
    show_image = args.show_image

    grid = np.arange(-2.5, 2.5 + grid_spacing, grid_spacing)
    gaussgrid = np.arange(-5 * bw, 5 * bw + grid_spacing, grid_spacing)

    prob = get_prob(grid, kt)
    gauss = gaussian(gaussgrid, 0, bw)
    conv = np.convolve(prob, gauss, mode="same") * grid_spacing
    if show_image or conv_image:
        plot_conv(grid, gaussgrid, prob, conv, gauss, conv_image)

    first_term = np.log(conv/prob)
    print(first_term)
    second_term = np.trapz(first_term, dx=grid_spacing)
    print(f"max: {max(first_term)}")
    print(f"integral: {second_term}")


if __name__ == "__main__":
    main()
