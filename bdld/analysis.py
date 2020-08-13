#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

def calculate_fes(trajectory, kt, ranges, bins=101, mintozero=True):
    """Calculate free energy surface from trajectory

    :param trajectory: list of positions as numpy arrays
    :param float kt: thermal energy of system
    :param ranges: list of tuples with extent of FES (min and max) per direction
    :param int bins: number of bins for histogram

    :return fes: numpy array with free energy values
    :return axes: list of numpy arrays with axes for the fes values
    """
    hist, axes = np.histogramdd(np.vstack(trajectory),bins=bins, range=ranges, density=True)
    # axes have bin boundaries -> shift to middle of intervals
    axes = [np.array([(axis[i] + axis[i+1]) / 2 for i in range(0,len(axis)-1)]) for axis in axes]
    fes = np.where(hist == 0, np.inf, - kt * np.log(hist, where=(hist!=0)))
    if mintozero:
        fes -= np.min(fes)
    return fes, axes


def calculate_reference(pot, positions):
    """Calculate FES from potential

    :param pot: the potential to evaluate
    :param positions: list of positions to evaluate
    """
    ref = np.fromiter((pot.evaluate(p)[0] for p in positions), float, len(positions))
    ref -= np.min(ref)
    return ref


def plot_fes(fes, axes, ref=None, fesrange=None, filename=None, title=None):
    """Show fes with matplotlib

    :param fes: the fes to plot
    :param axes: axes of plot
    :param ref: optional reference FES to plot (makes only sense for 1d fes)
    :param fesrange: optional list with minimum and maximum value to show
    :param filename: optional filename to save figure to
    :param title: optional title for the legend
    """
    fig = plt.figure(figsize=(8,4),dpi=100)
    if plt.get_backend() == "Qt5Agg":
        fig.canvas.setFixedSize(*fig.get_size_inches()*fig.dpi)  # ensure we really have that size
    ax = plt.axes()
    if len(fes.shape) == 1:
        if ref is not None:
            ax.plot(axes[0],ref,'b-',label='ref')
        ax.plot(axes[0],fes,'r-',label='FES')
        ax.legend(title=title)
        ax.set_ylabel('F (energy units)')
        if fesrange is not None:
            ax.set_ylim(fesrange)
        else:  # automatically crop from fes values
            ylim = np.where(np.isinf(fes), 0, fes).max()  # find max that is not inf
            ax.set_ylim([-0.05*ylim,1.05*ylim])  # crop unused parts
    elif len(fes.shape) == 2:
        img = ax.imshow(fes, origin='lower', extent=(axes[0][0],axes[0][-1],axes[-1][0],axes[-1][-1]))
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
        if filename is None:
            break
        try:
            fig.savefig(filename)
            break
        except OSError as e:
            print(f"Could not save: {e}")


def calculate_delta_F(fes, kt, masks):
    """Calculates the free energy difference between states

    If more than two are specified, this returns the difference to the first state for all others

    :param fes: free energy surface to examine
    :type fes: list or numpy.ndarray
    :param float kt: energy in units of kT
    :param masks: a list of boolean numpy arrays resembling the states

    :return delta_F: a list of doubles containing the free energy difference to the first state
    """
    probabilities = np.exp(- fes / float(kt))
    state_probs = [np.sum(probabilities[m]) for m in masks]
    delta_F = [- kt * np.log(state_probs[i]/state_probs[0]) for i in range(1, len(state_probs))]
    return delta_F


def count_particles_per_state(particles, ranges):
    """Return the number of particles in each state

    currently for 1d only

    :param particles: list with all Particles
    :param ranges: list with [min, max] ranges for each state
    """
    counts = [0] * len(ranges)
    for p in particles:
        for i, ra in enumerate(ranges):
            if ra[0] < p.pos < ra[1]:
                counts[i] += 1
    return counts
