"""Optional actions

These are actions that write to file and do analysis in periodic intervals
Each of these needs to inherit from action.Action
"""
from typing import List, Optional, Tuple, Union

import numpy as np

from bdld import analysis
from bdld.action import Action, get_valid_data
from bdld.bussi_parinello_ld import BussiParinelloLD
from bdld.histogram import Histogram
from bdld.helpers.plumed_header import PlumedHeader


class TrajectoryAction(Action):
    """Class that stories trajectories and writes them to file

    The write_stride parameter determines how many trajectory
    points are being held in memory even if they are never written.
    This allow other actions to use them.

    :param traj: fixed size numpy array holding the time and positions
                 it is written row-wise (i.e. every row represents a time)
                 and overwritten after being saved to file
    """

    def __init__(
        self,
        ld: BussiParinelloLD,
        stride: Optional[int] = None,
        filename: Optional[str] = None,
        fileheader: Optional[PlumedHeader] = None,
        write_stride: Optional[int] = None,
        write_fmt: Optional[str] = None,
    ) -> None:
        """Set up trajectory storage action

        :param ld: Langevin Dynamics to track
        :param stride: write every nth time step to file, default 1
        :param filename: base of filename(s) to write to
        :param fileheader: header for the files, the FIELDS line will be overwritten
        :param write_stride: write to file every n time steps, default 100
        :param write_fmt: numeric format for saving the data, default "%14.9f"
        """
        n_particles = len(ld.particles)
        self.ld = ld
        self.filenames: Optional[List[str]] = None
        self.stride: int = stride or 1
        self.write_stride: int = write_stride or 100
        # one more per row for storing the time
        self.traj = np.empty((self.write_stride, n_particles + 1, ld.pot.n_dim))
        self.last_write: int = 0
        # write headers
        if filename:
            self.filenames = [f"{filename}.{i}" for i in range(n_particles)]
            self.write_fmt = write_fmt or "%14.9f"
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
        self.traj[row, 0] = step * self.ld.dt
        self.traj[row, 1:] = [p.pos for p in self.ld.particles]

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
            save_data = get_valid_data(
                self.traj, step, self.stride, self.write_stride, self.last_write
            )
            for i, filename in enumerate(self.filenames):
                with open(filename, "ab") as f:
                    np.savetxt(
                        f,
                        save_data[:, (0, i + 1)].reshape((-1, 1 + self.ld.pot.n_dim)),
                        delimiter=" ",
                        newline="\n",
                        fmt=self.write_fmt,
                    )
                self.last_write = step


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
        filename: Optional[str] = None,
        fileheader: Optional[Union[PlumedHeader, str]] = None,
        write_stride: Optional[int] = None,
        write_fmt: Optional[str] = None,
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
        :param write_fmt: numeric format for saving the data, default "%14.9f"
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
        self.stride = stride or 1
        self.write_stride = write_stride
        self.write_fmt = write_fmt or "%14.9f"
        self.fileheader = fileheader or ""
        self.filename = filename
        if write_stride:
            if not filename:
                e = "Specifying a write_stride but no filename makes no sense"
                raise ValueError(e)
        self.update_stride = traj_action.stride

    def run(self, step: int):
        """Add trajectory data to histogram and write to file if strides are matched

        :param step: current simulation step
        """
        if step % self.update_stride == 0:
            data = get_valid_data(self.traj_action.traj, step, self.stride, self.traj_action.write_stride, self.traj_action.last_write)
            # flatten the first 2 dimensions (combine all times)
            self.histo.add(data.reshape(-1, data.shape[-1]))
        if self.write_stride and step % self.write_stride == 0:
            self.write(step)

    def final_run(self, step: int):
        """Same as run without stride checks"""
        data = get_valid_data(self.traj_action.traj, step, self.stride, self.traj_action.write_stride, self.traj_action.last_write)
        # flatten the first 2 dimensions (combine all times)
        self.histo.add(data.reshape(-1, data.shape[-1]))
        if self.filename:
            self.write()

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
            self.histo.write_to_file(filename, self.write_fmt, str(self.fileheader))


class FesAction(Action):
    """Calculate fes from histogram and save to file"""

    def __init__(
        self,
        histo_action: HistogramAction,
        stride: Optional[int] = None,
        filename: Optional[str] = None,
        fileheader: Optional[Union[PlumedHeader, str]] = None,
        write_stride: Optional[int] = None,
        write_fmt: Optional[str] = None,
        plot_stride: Optional[int] = None,
        plot_filename: Optional[str] = None,
        plot_domain: Optional[Tuple[float, float]] = None,
        plot_title: str = None,
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
        :param write_fmt: numeric format for saving the data, default "%14.9f"
        :param plot_filename: filename for plot, optional
        :param plot_domain: specify domain for plots, optional
        :param ref: reference fes for plot, optional
        """
        self.histo_action = histo_action
        self.kt = self.histo_action.traj_action.ld.kt
        self.get_fes_grid = self.histo_action.histo.get_fes_grid
        self.stride = stride
        if self.stride:
            if self.stride % histo_action.update_stride != 0:
                print("Warning: the FES stride is no multiple of the Histogram stride.")
        # writing
        self.filename = filename
        self.write_fmt = write_fmt or "%14.9f"
        self.fileheader = fileheader or ""
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
        # plotting
        self.plot_filename = plot_filename
        self.plot_domain = plot_domain
        self.plot_title = plot_title or ""
        self.ref = ref
        self.plot_stride = plot_stride
        if self.plot_stride:
            if self.plot_stride % self.stride != 0:
                print("Warning: the plot stride is no multiple of the update stride.")
            if not self.plot_filename:
                e = "Specifying a plot_stride but no plot_filename makes no sense"
                raise ValueError(e)

    def run(self, step: int) -> None:
        """Calculate fes from histogram, write to file and plot if matching strides

        :param step: current simulation step, optional
        """
        if self.stride and step % self.stride == 0:
            self.histo_action.histo.calculate_fes(self.kt)
        if self.write_stride and step % self.write_stride == 0:
            self.write(step)
        if self.plot_stride and step % self.plot_stride == 0:
            self.plot(step)

    def final_run(self, step: int) -> None:
        """Same as run without stride checks"""
        self.histo_action.histo.calculate_fes(self.kt)
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
            self.get_fes_grid().write_to_file(
                filename, self.write_fmt, str(self.fileheader)
            )

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


class DeltaFAction(Action):
    """Calculate Delta F from fes and save to file"""

    def __init__(
        self,
        fes_action: FesAction,
        masks: np.ndarray,
        stride: Optional[int] = None,
        filename: Optional[str] = None,
        fileheader: Optional[PlumedHeader] = None,
        write_stride: Optional[int] = None,
        write_fmt: Optional[str] = None,
    ) -> None:
        """Set up action that calculates free energy differences between states

        If no stride is given, this action is not run periodically
        but can manually be triggered.
        If a stride is given, it needs to be a multiple of the stride of the
        FesAction to make sense

        :param fes_action: Fes action to analyise
        :param masks: masks that define the states
        :param stride: calculate delta f every n time steps, optional
        :param filename: filename to save fes to, optional
        :param fileheader: header for wiles
        :param write_stride: write to file every n time steps, default None (never)
        :param write_fmt: numeric format for saving the data, default "%14.9f"
        """
        self.fes_action = fes_action
        self.masks = masks
        self.stride = stride
        if self.stride:
            if not self.fes_action.stride or self.stride % self.fes_action.stride != 0:
                print("Warning: the FES stride is no multiple of the Histogram stride.")
            self.write_stride = write_stride or self.stride * 100
            if self.write_stride % self.stride != 0:
                e = "The write stride must be a multiple of the update stride."
                raise ValueError(e)
            if not filename:
                e = "Specifying a write_stride but no filename makes no sense"
                raise ValueError(e)
            # time + (masks -1) states
            self.delta_f = np.empty((self.write_stride // self.stride, len(self.masks)))
            self.last_write: int = 0
        else:  # just store one data set
            self.delta_f = np.empty((len(self.masks)))
        # writing
        self.filename = filename
        if self.filename:
            if fileheader:
                with open(self.filename, "w") as f:
                    fileheader[0] = "FIELDS +" " ".join(
                        f"delta_f.1-{i}" for i in range(2, len(self.masks) + 1)
                    )
                    f.write(str(fileheader) + "\n")
        self.write_fmt = write_fmt if write_fmt else "%14.9f"

        # copy temp and timestep from LD, shortcut for FES grid
        self.kt = self.fes_action.histo_action.traj_action.ld.kt
        self.dt = self.fes_action.histo_action.traj_action.ld.dt
        self.fes = self.fes_action.histo_action.histo.fes

    def run(self, step: int) -> None:
        """Calculate Delta F from fes and write to file

        :param step: current simulation step, optional
        """
        if self.stride and step % self.stride == 0:
            row = (step % self.write_stride) // self.stride - 1
            self.delta_f[row, 0] = step * self.dt
            self.delta_f[row, 1:] = analysis.calculate_delta_f(
                self.fes, self.kt, self.masks
            )
        if self.write_stride and step % self.write_stride == 0:
            self.write(step)

    def write(self, step: int) -> None:
        """Write fes to file

        If a step is specified it will be appended to the filename, i.e. it is written
        to a new file.
        If no filename is set, this will do nothing.

        :param step: current simulation step, optional
        """
        if self.filename:
            if self.stride:
                save_data = get_valid_data(
                    self.delta_f, step, self.stride, self.write_stride, self.last_write
                )
                self.last_write = step
            else:  # reshape to rows to save as single line
                save_data = self.delta_f.reshape((1, -1))
            with open(self.filename, "ab") as f:
                np.savetxt(
                    f,
                    save_data,
                    delimiter=" ",
                    newline="\n",
                    fmt=self.write_fmt,
                )
