#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import sys

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


def plot_fes(fes, axes, ref=None, fesrange=None, filename=None):
    """Show fes with matplotlib

    :param fes: the fes to plot
    :param axes: axes of plot
    :param ref: optional reference FES to plot (makes only sense for 1d fes)
    :param fesrange: optional list with minimum and maximum value to show
    """
    fig = plt.figure(figsize=(8,4),dpi=100)
    if plt.get_backend() == "Qt5Agg":
        fig.canvas.setFixedSize(*fig.get_size_inches()*fig.dpi)  # ensure we really have that size
    ax = plt.axes()
    if len(fes.shape) == 1:
        if ref is not None:
            ax.plot(axes[0],ref,'b-',label='ref')
        ax.plot(axes[0],fes,'r-',label='FES')
        ax.legend()
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
    else:
        fig.show()
        save_fig_interactive(fig, filename)


def save_fig_interactive(fig, filename=None):
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
