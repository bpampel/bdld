"""Misc analysis functions"""

from typing import List, Tuple, Union

import numpy as np
import matplotlib.pyplot as plt


def plot_fes(fes, axes, ref=None, plot_domain=None, filename=None, title=None):
    """Show fes with matplotlib

    Does work with 1D and 2D fes, higher dimensions will be ignored without error

    :param fes: the fes to plot
    :param axes: axes of plot
    :param ref: optional reference FES to plot (makes only sense for 1d fes)
    :param plot_domain: optional Tuple with minimum and maximum value to show
    :param filename: optional filename to save figure to
    :param title: optional title for the legend
    """
    fig = plt.figure(figsize=(8, 4), dpi=100)
    if plt.get_backend() == "Qt5Agg": # fix for Qt5Agg
        fig.canvas.setFixedSize(
            *fig.get_size_inches() * fig.dpi
        )  # ensure we really have that size
    ax = plt.axes()
    if len(fes.shape) == 1:
        if ref is not None:
            ax.plot(axes[0], ref, "b-", label="ref")
        ax.plot(axes[0], fes, "r-", label="FES")
        ax.legend(title=title)
        ax.set_ylabel("F (energy units)")
        if plot_domain is not None:
            ax.set_ylim(plot_domain)
        else:  # automatically crop from fes values
            ylim = np.where(np.isinf(fes), 0, fes).max()  # find max that is not inf
            ax.set_ylim([-0.05 * ylim, 1.05 * ylim])  # crop unused parts
    elif len(fes.shape) == 2:
        try:
            vmin, vmax = plot_domain
        except TypeError:  # can't unpack if not set
            vmin = None
            vmax = None
        img = ax.imshow(
            fes,
            origin="lower",
            extent=(axes[0][0], axes[0][-1], axes[-1][0], axes[-1][-1]),
            vmin=vmin,
            vmax=vmax,
        )
        fig.colorbar(img, ax=ax)
    if filename:
        try:
            fig.savefig(filename)
        except ValueError as e:
            print(e)
            save_fig_interactive(fig)
    else:
        fig.show()
        save_fig_interactive(fig)
    plt.close(fig)


def save_fig_interactive(fig):
    """Ask for filename to save file to"""
    while True:
        try:
            filename = input("Save figure to path: (empty for no save) ")
        except ValueError as e:
            print(e)
            continue
        if not filename:
            break
        try:
            fig.savefig(filename)
            break
        except OSError as e:
            print(f"Could not save: {e}")


def calculate_delta_f(fes: np.ndarray, kt: float, masks: List[np.ndarray]):
    """Calculates the free energy difference between states

    If more than two are specified, this returns the difference to the first state for all others

    :param fes: free energy surface to examine
    :param kt: energy in units of kT
    :param masks: a list of boolean numpy arrays resembling the states

    :return delta_F: a list of doubles containing the free energy difference to the first state
    """
    probabilities = np.exp(-fes / float(kt)).reshape((-1))
    state_probs = [np.sum(probabilities[m]) for m in masks]
    delta_f = [
        -kt * np.log(state_probs[i] / state_probs[0])
        for i in range(1, len(state_probs))
    ]
    return delta_f


def count_particles_per_state(
    particles: List[np.ndarray], ranges: List[List[Tuple[float, float]]]
) -> List[int]:
    """Return the number of particles in each state

    This assumes rectangular states

    :param particles: list with all Particles
    :param ranges: list with list of [(min_x, max_x), (min_y, max_y), ...] ranges for each state
    """
    counts = [0] * len(ranges)
    for p in particles:
        for i, state in enumerate(ranges):
            # check if inside in all dimensions
            if all([x_min <= p.pos <= x_max for (x_min, x_max) in state]):
                counts[i] += 1
    return counts
