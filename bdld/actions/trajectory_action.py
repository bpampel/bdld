"""Module holding the TrajectoryAction class"""

from typing import List, Optional

import numpy as np

from bdld.actions.action import Action, get_valid_data
from bdld.actions.bussi_parinello_ld import BussiParinelloLD
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
        print("Setting up storage of the trajectories")
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
            print(f"Saving every {self.stride} point to the files {filename}.{{i}}")
        print()

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
            save_data = get_valid_data(self.traj, step, self.stride, 1, self.last_write)
            for i, filename in enumerate(self.filenames):
                with open(filename, "ab") as f:
                    np.savetxt(
                        f,
                        # 3d (times, 1+walkers, pot_dims) to 2d array (times, 1+pot_dims)
                        save_data[:, (0, i + 1)].reshape((-1, 1 + self.ld.pot.n_dim)),
                        delimiter=" ",
                        newline="\n",
                        fmt=self.write_fmt,
                    )
                self.last_write = step
