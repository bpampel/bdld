"""Optional actions to use in the BirthDeathLangevinDynamics class

These are actions that write to file and do analysis in periodic intervals
Each of these needs to have a "run(step)" function that performs the action
"""
from typing import List, Optional, Tuple, Union

import numpy as np

from bdld.bussi_parinello_ld import BussiParinelloLD
from bdld.histogram import Histogram
from bdld.helpers.plumed_header import PlumedHeader


class TrajectoryAction:
    """Class that stories trajectories and writes them to file"""

    def __init__(
        self,
        ld: BussiParinelloLD,
        stride: int = 1,
        filename: str = "",
        header: Optional[PlumedHeader] = None,
        write_stride: int = 100,
        write_fmt: str = "%14.9",
    ) -> None:
        """Set up trajectory storage action

        :param ld: Langevin Dynamics to track
        :param stride: write every nth time step to file, default 1
        :param filename: base of filename(s) to write to
        :param header: ???
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
            if header:
                for i, fname in enumerate(self.filenames):
                    with open(fname, "w") as f:
                        header[0] = f"FIELDS traj.{i}"
                        f.write(str(header) + "\n")

    def run(self, step):
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

    def write(self, step: int):
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


class HistogramAction:
    """Action collecting trajectory data into a histogram

    The Histogram data member is periodically enhanced with the new data from
    the trajectories.

    :param histo: Histogram data
    :param update_stide: add trajectory data every n time steps
    """

    def __init__(
        self,
        traj_action: TrajectoryAction,
        n_bins: List[int],
        ranges: List[Tuple[float, float]],
        stride: int = 1,
        filename: str = "",
        header: Optional[Union[PlumedHeader, str]] = None,
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
        :param header: ???
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
            if header:
                self.header = header
            else:
                self.header = ""
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
        self.histo.write_to_file(filename, str(self.header))
