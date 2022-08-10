"""Module holding the FesAction class"""

from collections import OrderedDict
import logging
from typing import Optional, Tuple

import numpy as np

from bdld import analysis, grid, histogram
from bdld.actions.action import Action
from bdld.actions.histogram_action import HistogramAction
from bdld.helpers.misc import backup_if_exists
from bdld.helpers.plumed_header import PlumedHeader


class FesAction(Action):
    """Calculate fes from histogram and save to file

    The actual FES is not a data member of this class but rather a member of the
    Histogram instance of the passed histo_action.
    This is mostly for historical reasons, as the Histogram class existed before
    the code was restructured into actions

    :param fes: Grid instance hosting the FES
    :param kt: thermal energy in units of kT
    """

    def __init__(
        self,
        histo_action: HistogramAction,
        stride: Optional[int] = None,
        filename: Optional[str] = None,
        write_stride: Optional[int] = None,
        write_fmt: Optional[str] = None,
        plot_stride: Optional[int] = None,
        plot_filename: Optional[str] = None,
        plot_domain: Optional[Tuple[float, float]] = None,
        plot_title: str = None,
        ref: Optional[np.ndarray] = None,
    ) -> None:
        """Set up fes calculation for a Histogram

        If no stride is given, this action is not run periodically
        but can manually be triggered.
        If a stride is given, it needs to be a multiple of the update_stride of the
        HistogramAction (write_stride of the TrajectoryAction) to make sense.
        Same is true for the plot_stride and the stride, otherwise the result will
        not represent the current state of the simulation.

        :param histo_action: the histogram to use for the calculations
        :param stride: calculate fes every n time steps, optional
        :param filename: filename to save fes to, optional
        :param write_stride: write to file every n time steps, default None (never)
        :param write_fmt: numeric format for saving the data, default "%14.9f"
        :param plot_filename: filename for plot, optional
        :param plot_domain: specify domain for plots, optional
        :param ref: reference fes for plot, optional
        """
        print("Setting up FES calculation for the histogram\n" + "Parameters:")
        self.histo_action = histo_action
        self.kt = self.histo_action.traj_action.ld.kt
        print(f"  kt = {self.kt}")
        self.fes = histo_action.histo.copy_empty()  # empty Grid with correct points
        self.stride = stride
        if self.stride:
            if self.stride % histo_action.update_stride != 0:
                logging.warning(
                    "The FES stride is no multiple of the Histogram update stride. "
                    "Set a matching write-stride for the [trajectories]"
                )
            print(f"  stride = {self.stride}")
        # writing
        self.filename = filename
        if filename:  # set up header
            fields = histo_action.traj_action.ld.pot.get_fields() + ["fes"]
            constants = OrderedDict()  # makes sure constants are printed in order
            h_grid = histo_action.histo
            for i in range(h_grid.n_dim):
                constants[f"{fields[i]}_min"] = h_grid.ranges[i][0]
                constants[f"{fields[i]}_max"] = h_grid.ranges[i][1]
                constants[f"{fields[i]}_n_bins"] = h_grid.n_points[i]
            self.fileheader = PlumedHeader(fields, constants)
        self.write_fmt = write_fmt or "%14.9f"
        self.write_stride = write_stride
        if self.write_stride:
            if not self.stride:
                e = "Specifying a write_stride but no stride makes no sense"
                raise ValueError(e)
            if self.write_stride % self.stride != 0:
                print("Warning: the write stride is no multiple of the update stride.")
            if not self.filename:
                e = "Specifying a write_stride but no filename makes no sense"
                raise ValueError(e)
            print(
                f"Saving FES every {self.write_stride} time steps to '{filename}_{{step}}'"
            )
        # plotting
        self.plot_filename = plot_filename
        self.plot_domain = plot_domain
        self.plot_title = plot_title or ""
        self.ref = ref
        self.plot_stride = plot_stride
        if self.plot_stride:
            if not self.stride:
                e = "Specifying a plot_stride but no stride makes no sense"
                raise ValueError(e)
            if self.plot_stride % self.stride != 0:
                print("Warning: the plot stride is no multiple of the update stride.")
            if not self.plot_filename:
                e = "Specifying a plot_stride but no plot_filename makes no sense"
                raise ValueError(e)
            print(
                f"Plotting every {self.plot_stride} time steps to '{self.plot_filename}_{{step}}'"
            )
        print()

    def run(self, step: int) -> None:
        """Calculate fes from histogram, write to file and plot if matching strides

        :param step: current simulation step, optional
        """
        if self.stride and step % self.stride == 0:
            self.fes = calculate_fes(self.histo_action.histo, self.kt)
        if self.write_stride and step % self.write_stride == 0:
            self.write(step)
        if self.plot_stride and step % self.plot_stride == 0:
            self.plot(step)

    def final_run(self, step: int) -> None:
        """Same as run() but without stride checks and passing the step number"""
        self.fes = calculate_fes(self.histo_action.histo, self.kt)
        self.write()
        self.plot()

    def write(self, step: int = None) -> None:
        """Write fes to file

        If a step is specified it will be appended to the filename, i.e. it is written
        to a new file.
        If no filename is set, this will do nothing.

        :param step: current simulation step, optional
        """
        if self.filename:
            if step:
                filename = f"{self.filename}_{step}"
            else:
                filename = self.filename
            backup_if_exists(filename)
            self.fes.write_to_file(filename, self.write_fmt, str(self.fileheader))

    def plot(self, step: int = None) -> None:
        """Plot fes with reference and optionally save to file

        If a step is specified it will be appended to the filename, i.e. it is written
        to a new file. It will also be used for the plot title

        :param step: current simulation step, optional
        """
        if self.plot_filename:
            if step:
                plot_filename = f"{self.plot_filename}_{step}"
                plot_title = f"{self.plot_title}_{step}"
            else:
                plot_filename = self.plot_filename
                plot_title = self.plot_title
            analysis.plot_fes(
                self.histo_action.histo.fes,
                self.histo_action.histo.bin_centers(),
                ref=self.ref,
                plot_domain=self.plot_domain,
                filename=plot_filename,
                title=plot_title,
            )


def calculate_fes(
    histo: histogram.Histogram, kt: float, mintozero: bool = True
) -> grid.Grid:
    """Calculate free energy surface from histogram

    :param histo: Histogram instance to calculate FES from
    :param kt: thermal energy of the system
    :param mintozero: shift FES to have minimum at zero

    :return fes: Grid with the fes values as data
    """
    # set bins with 0 count to inf
    fes = np.where(
        histo.data == 0, np.inf, -kt * np.log(histo.data, where=(histo.data != 0))
    )
    if mintozero:
        minimum = np.min(fes)
        if minimum != np.inf:  # otherwise all values become nan
            fes -= np.min(fes)
    # store in new grid instance and return
    fes_grid = histo.copy_empty()
    fes_grid.data = fes
    return fes_grid
