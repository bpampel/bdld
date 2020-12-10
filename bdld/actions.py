from typing import List, Optional

import numpy as np

from bdld.bussi_parinello_ld import BussiParinelloLD


class Trajectories:
    """Class that stories trajectories and writes them to file"""

    def __init__(
        self,
        ld: BussiParinelloLD,
        filename=None,
        header=None,
        stride: int = 1,
        write_stride: int = 100,
        write_fmt: str = "%14.9",
    ) -> None:
        # one more per row for storing the time
        n_particles = len(ld.particles)
        self.traj = np.empty((write_stride, n_particles + 1))
        self.filenames: Optional[List[str]] = None
        self.stride = stride
        self.write_stride = write_stride
        self.write_fmt = write_fmt
        self.last_write: int = 0
        # write headers
        if filename:
            self.filenames = [f"{filename}.{i}" for i in range(n_particles)]
            for i, filename in enumerate(self.filenames):
                header = ld.generate_fileheader([f"traj.{i}"])
                with open(filename, "w") as f:
                    f.write(str(header) + "\n")

    def run(self, ld: BussiParinelloLD, step):
        """Store positions in traj array and write to file if stride is matched"""
        row = (step % self.write_stride) - 1  # saving starts at step 1
        self.traj[row][0] = step * ld.dt
        self.traj[row][1:] = [p.pos for p in ld.particles]

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
