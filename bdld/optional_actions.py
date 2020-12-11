"""Optional actions

These are actions that write to file and do analysis in periodic intervals
Each of these needs to inherit from action.Action
"""
from typing import List, Optional, Tuple, Union

import numpy as np

from bdld import analysis
from bdld.action import Action
from bdld.bussi_parinello_ld import BussiParinelloLD
from bdld.histogram import Histogram
from bdld.helpers.plumed_header import PlumedHeader


class TrajectoryAction(Action):
    """Class that stories trajectories and writes them to file"""

    def __init__(
        self,
        ld: BussiParinelloLD,
        stride: int = 1,
        filename: str = "",
        fileheader: Optional[PlumedHeader] = None,
        write_stride: int = 100,
        write_fmt: str = "%14.9",
    ) -> None:
        """Set up trajectory storage action

        :param ld: Langevin Dynamics to track
        :param stride: write every nth time step to file, default 1
        :param filename: base of filename(s) to write to
        :param fileheader: header for the files, the FIELDS line will be overwritten
        :param write_stride: write to file every n time steps, default 100
        :param write_fmt: numeric format for saving the data, default "%14.9"
        """
        n_particles = len(ld.particles)
        self.ld = ld
        # one more per row for storing the time
        self.traj = np.empty((write_stride, n_particles + 1, ld.pot.n_dim))
        self.filenames: Optional[List[str]] = None
        self.stride = stride
        self.write_stride = write_stride
        self.last_write: int = 0
        # write headers
        if filename:
            self.filenames = [f"{filename}.{i}" for i in range(n_particles)]
            self.write_fmt = write_fmt
            if fileheader:
                for i, fname in enumerate(self.filenames):
                    with open(fname, "w") as f:
                        fileheader[0] = f"FIELDS traj.{i}"
                        f.write(str(fileheader) + "\n")

    def run(self, step: int) -> None:
        """Store positions in traj array and write to file if write_stride is matched

        The stride parameters is ignored here and all times are temporarily stored
        This is because a HistogramAction using the data might have a different stride

        :param step: current simulation step
        """
        row = (step % self.write_stride) - 1  # saving starts at step 1
        self.traj[row][0] = step * self.ld.dt
        self.traj[row][1:] = [p.pos for p in self.ld.particles]

        if step % self.write_stride == 0:
            self.write(step)

    def final_run(self, step: int) -> None:
        """Write rest of trajectories to files"""
        self.write(step)

    def write(self, step: int) -> None:
        """Write currently stored trajectory data to file

        This can also be called between regular saves (e.g. at the end of the simulation)
        and will not result in missing or duplicate data
        Because the trajectories are stored in the array independently from the writes
        the function needs to do some arithmetics to find out what to write

        If no filenames were set this function will do nothing but not raise an exception

        :param step: current simulation step
        """
        if self.filenames:
            save_data = self.get_valid_data(step)
            for i, filename in enumerate(self.filenames):
                with open(filename, "ab") as f:
                    np.savetxt(
                        f,
                        save_data[:, (0, i + 1)],
                        delimiter=" ",
                        newline="\n",
                        fmt=self.write_fmt,
                    )
                self.last_write = step

    def get_valid_data(
        self,
        step: int,
        stride: int = None,
        last_write: int = None,
    ) -> np.ndarray:
        """Get the currently valid data as a view of the stored trajectory array

        By default this will use the internal stride and last_write, but other ones can be specified
        (e.g. for histogramming)

        :param step: current simulation step
        :param stride: return only every nth datapoint
        :param last_write: when the data was last written (needs to be less than write_stride ago)
        """
        if not stride:
            stride = self.stride
        if not last_write:
            last_write = self.last_write
        last_element = step % self.write_stride
        already_written = last_write % self.write_stride
        # shortcut for basic case
        if stride == 1 and already_written == 0 and last_element == 0:
            return self.traj
        # shift from stride - number of elements at end that weren't saved last time
        stride_offset = stride - 1 - (last_write % stride)
        first_element = already_written + stride_offset
        if last_element == 0:
            return self.traj[first_element::stride]
        return self.traj[first_element:last_element:stride]


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
        stride: int = 1,
        filename: str = "",
        fileheader: Optional[Union[PlumedHeader, str]] = None,
        write_stride: Optional[int] = None,
        write_fmt: str = "%14.9",
    ) -> None:
        """Initialize a histogram for the trajectories

        Remark: the actual "adding to histogram" stride is determined from the
        write_stride of the trajectory because it has to be in sync.

        :param traj_action: trajectories to use for histogramming
        :param n_bins: number of bins of the histogram per dimension
        :param ranges: extent of histogram (min, max) per dimension
        :param int stride: add every nth particle position to the histogram, default 1
        :param filename: optional filename to save histogram to
        :param fileheader: header for the files
        :param write_stride: write to file every n time steps, default None (never)
        :param write_fmt: numeric format for saving the data, default "%14.9"
        """
        n_dim = traj_action.traj.shape[-1]  # last dimension of traj array is dim of pot
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
        self.stride = stride
        self.write_stride = write_stride
        if write_stride:
            if not filename:
                e = "Specifying a write_stride but no filename makes no sense"
                raise ValueError(e)
            self.filename = filename
            if fileheader:
                self.fileheader = fileheader
            else:
                self.fileheader = ""
            self.write_fmt = write_fmt
        self.update_stride = traj_action.stride

    def run(self, step: int):
        """Add trajectory data to histogram and write to file if strides are matched

        :param step: current simulation step
        """
        if step % self.update_stride == 0:
            data = self.traj_action.get_valid_data(step, self.stride)
            # flatten the first 2 dimensions (combine all times)
            self.histo.add(data.reshape(-1, data.shape[-1]))
        if self.write_stride:
            if step % self.write_stride == 0:
                self.write(step)

    def final_run(self, step: int):
        """Same as run without stride checks"""
        data = self.traj_action.get_valid_data(step, self.stride)
        # flatten the first 2 dimensions (combine all times)
        self.histo.add(data.reshape(-1, data.shape[-1]))
        if self.filename:
            self.write()

    def write(self, step: int = None):
        """Write histogram to file

        If a step is specified it will be appended to the filename,
        i.e. it is written to a new file

        :param step: current simulation step, optional
        """
        if step:
            filename = f"{self.filename}_{step}"
        else:
            filename = self.filename
        self.histo.write_to_file(filename, self.write_fmt, str(self.fileheader))


class FesAction(Action):
    """Calculate fes from histogram and save to file"""

    def __init__(
        self,
        histo_action: HistogramAction,
        kt: float,
        stride: Optional[int] = None,
        filename: str = "",
        fileheader: Optional[Union[PlumedHeader, str]] = None,
        write_stride: Optional[int] = None,
        write_fmt: str = "%14.9",
        plot_stride: Optional[int] = None,
        plot_filename: str = "",
        plot_domain: Optional[Tuple[float, float]] = None,
        plot_title: str = "",
        ref: Optional[np.ndarray] = None,
    ) -> None:
        """Set up fes action for a Histogram

        If no stride is given, this action is not run periodically
        but can manually be triggered.
        If a stride is given, it needs to be a multiple of the update_stride of the
        HistogramAction (write_stride of the TrajectoryAction) to make sense.
        Same is true for the plot_stride and the stride, otherwise the result will
        not represent the current state of the simulation.

        :param histo_action: the histogram to use for the calculations
        :param kt: thermal energy in units of kT
        :param stride: calculate fes every n time steps, optional
        :param filename: filename to save fes to, optional
        :param write_stride: write to file every n time steps, default None (never)
        :param write_fmt: numeric format for saving the data, default "%14.9"
        :param plot_filename: filename for plot, optional
        :param plot_domain: specify domain for plots, optional
        :param ref: reference fes for plot, optional
        """
        self.histo_action = histo_action
        self.kt = kt
        if stride:
            self.stride = stride
            if stride % histo_action.update_stride != 0:
                print("Warning: the FES stride is no multiple of the Histogram stride.")
        # writing
        self.filename = filename
        if write_stride:
            self.write_stride = write_stride
            if write_stride % self.stride != 0:
                print("Warning: the write stride is no multiple of the update stride.")
            if not self.filename:
                e = "Specifying a write_stride but no filename makes no sense"
                raise ValueError(e)
            if fileheader:
                self.fileheader = fileheader
            else:
                self.fileheader = ""
            self.write_fmt = write_fmt
        # plotting
        self.plot_filename = plot_filename
        self.plot_domain = plot_domain
        self.plot_title = plot_title
        self.ref = ref
        if plot_stride:
            self.plot_stride = plot_stride
            if plot_stride % self.stride != 0:
                print("Warning: the plot stride is no multiple of the update stride.")
            if not self.plot_filename:
                e = "Specifying a plot_stride but no plot_filename makes no sense"
                raise ValueError(e)

    def run(self, step: int) -> None:
        """Calculate fes from histogram, write to file and plot if matching strides

        :param step: current simulation step, optional
        """
        if not step or step % self.stride == 0:
            self.histo_action.histo.calculate_fes(self.kt)
        if not step or step % self.write_stride == 0:
            if self.filename:
                self.write(step)
        if not step or step % self.plot_stride == 0:
            if self.plot_filename:
                self.plot(step)

    def run_final(self, step: int) -> None:
        """Same as run without stride checks"""
        self.histo_action.histo.calculate_fes(self.kt)
        if self.filename:
            self.write()
        if self.plot_filename:
            self.plot()

    def write(self, step: int = None) -> None:
        """Write fes to file

        If a step is specified it will be appended to the filename, i.e. it is written
        to a new file.

        :param step: current simulation step, optional
        """
        if step:
            filename = f"{self.filename}_{step}"
        else:
            filename = self.filename
        self.histo_action.histo.get_fes_grid().write_to_file(
            filename, self.write_fmt, str(self.fileheader)
        )

    def plot(self, step: int = None) -> None:
        """Plot fes with reference and optionally save to file

        If a step is specified it will be appended to the filename, i.e. it is written
        to a new file. It will also be used for the plot title

        :param step: current simulation step, optional
        """
        if step:
            filename = f"{self.filename}_{step}"
            plot_title = f"{self.plot_title}_{step}"
        else:
            filename = self.filename
            plot_title = self.plot_title
        analysis.plot_fes(
            self.histo_action.histo.fes,
            self.histo_action.histo.bin_centers(),
            ref=self.ref,
            plot_domain=self.plot_domain,
            filename=filename,
            title=plot_title,
        )
