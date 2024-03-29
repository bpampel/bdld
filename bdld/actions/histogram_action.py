"""Module holding the HistogramAction class"""

from collections import OrderedDict
import logging
from typing import List, Optional, Tuple

import numpy as np

from bdld.histogram import Histogram
from bdld.actions.action import Action, get_valid_data
from bdld.actions.trajectory_action import TrajectoryAction
from bdld.helpers.misc import backup_if_exists
from bdld.helpers.plumed_header import PlumedHeader


class HistogramAction(Action):
    """Action collecting trajectory data into a histogram

    The Histogram data member is periodically enhanced with the new data from
    the trajectories.

    :param histo: Histogram data
    :param update_stride: add trajectory data every n time steps
    """

    def __init__(
        self,
        traj_action: TrajectoryAction,
        n_bins: List[int],
        ranges: List[Tuple[float, float]],
        stride: Optional[int] = None,
        reset: Optional[List[int]] = None,
        filename: Optional[str] = None,
        write_stride: Optional[int] = None,
        write_fmt: Optional[str] = None,
    ) -> None:
        """Initialize a histogram for the trajectories

        Remark: the actual "adding to histogram" stride is determined from the
        write_stride of the trajectory because it has to be in sync.

        :param traj_action: trajectories to use for histogramming
        :param n_bins: number of bins of the histogram per dimension
        :param ranges: extent of histogram (min, max) per dimension
        :param stride: add every nth particle position to the histogram, default 1
        :param reset: clear previous counts at given timesteps (reset histogram)
        :param filename: optional filename to save histogram to
        :param write_stride: write to file every n time steps, default None (never)
        :param write_fmt: numeric format for saving the data, default "%14.9f"
        :raise ValueError: if a write_stride but no filename is passed
        :raise ValueError: if histogram and system dimensions do not match
        """
        print(
            f"Setting up histogram for the trajectories\n"
            f"Parameters:\n"
            f"  ranges = {ranges}\n"
            f"  n_bins = {n_bins}"
        )
        n_dim = traj_action.positions.shape[
            -1
        ]  # last dimension of traj array is dim of pot
        if n_dim != len(n_bins):
            e = (
                "Dimensions of histogram bins don't match dimensions of system"
                + f"({len(n_bins)} vs. {n_dim})"
            )
            raise ValueError(e)
        if n_dim != len(ranges):
            e = (
                "Dimensions of histogram ranges don't match dimensions of system "
                + f"({len(ranges)} vs. {n_dim})"
            )
            raise ValueError(e)
        self.traj_action = traj_action
        self.histo = Histogram(n_bins, ranges)
        self.stride = stride or 1
        print(f"  stride = {self.stride}")
        self.reset = reset or []
        self.write_stride = write_stride
        self.write_fmt = write_fmt or "%14.9f"
        self.filename = filename
        if filename:  # set up header
            fields = traj_action.ld.pot.get_fields() + ["histo_count"]
            constants = OrderedDict()  # makes sure constants are printed in order
            for i in range(n_dim):
                constants[f"{fields[i]}_min"] = ranges[i][0]
                constants[f"{fields[i]}_max"] = ranges[i][1]
                constants[f"{fields[i]}_n_bins"] = n_bins[i]
            self.fileheader = PlumedHeader(fields, constants)
        if write_stride:
            if not filename:
                e = "Specifying a write_stride but no filename makes no sense"
                raise ValueError(e)
            print(
                f"Saving histogram every {self.write_stride} time steps to '{filename}'"
            )
        self.update_stride = traj_action.write_stride
        print()

    def run(self, step: int):
        """Add trajectory data to histogram and write to file if strides are matched

        :param step: current simulation step
        """
        if step % self.update_stride == 0:
            data = self.get_traj_data(step)
            # flatten the first 2 dimensions (combine all times)
            self.histo.add(data.reshape(-1, data.shape[-1]))
        if self.write_stride and step % self.write_stride == 0:
            self.write(step)
        if step in self.reset:
            logging.info("Reset of histogram at step %s", step)
            self.histo.clear()

    def final_run(self, step: int):
        """Same as run without stride checks

        :param step: current simulation step
        """
        data = self.get_traj_data(step)
        # flatten the first 2 dimensions (combine all times)
        self.histo.add(data.reshape(-1, data.shape[-1]))
        if self.filename:
            self.write()

    def get_traj_data(self, step: int) -> np.ndarray:
        """Get the right data from the trajectory array

        :param step: current simulation step
        """
        # take into account that the traj might have been written this step
        last_write = (
            self.traj_action.last_write
            if self.traj_action.last_write != step
            else self.traj_action.last_write - self.traj_action.write_stride
        )
        return get_valid_data(
            self.traj_action.positions,
            step,
            self.stride,
            1,  # assumes that traj data is stored even if not written
            last_write,
        )

    def write(self, step: int = None):
        """Write histogram to file

        If a step is specified it will be appended to the filename,
        i.e. it is written to a new file
        If no filename is set, this will do nothing.

        :param step: current simulation step, optional
        """
        if self.filename:
            if step:
                filename = f"{self.filename}_{step}"
            else:
                filename = self.filename
            backup_if_exists(filename)
            self.histo.write_to_file(filename, self.write_fmt, str(self.fileheader))
