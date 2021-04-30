"""Module holding the TrajectoryAction class"""

from typing import List, Optional

import numpy as np

from bdld.actions.action import Action, get_valid_data
from bdld.actions.bussi_parinello_ld import BussiParinelloLD
from bdld.helpers.misc import initialize_file, make_ordinal


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
        momentum: Optional[bool] = None,
        write_stride: Optional[int] = None,
        write_fmt: Optional[str] = None,
    ) -> None:
        """Set up trajectory storage action

        :param ld: Langevin Dynamics to track
        :param stride: write every nth time step to file, default 1
        :param filename: base of filename(s) to write to
        :param momentum: also save the momentum of the particles, default False
        :param write_stride: write to file every n time steps, default 100
        :param write_fmt: numeric format for saving the data, default "%14.9f"
        """
        print("Setting up storage of the trajectories")
        n_particles = len(ld.particles)
        self.ld = ld
        self.filenames: Optional[List[str]] = None
        self.stride: int = stride or 1
        self.write_stride: int = write_stride or 100
        # two data members for storing positions and time
        self.positions = np.empty((self.write_stride, n_particles, ld.pot.n_dim))
        self.store_momentum = momentum
        if self.store_momentum:
            self.momentum = np.empty((self.write_stride, n_particles, ld.pot.n_dim))
        self.times = np.empty((self.write_stride, 1))
        self.last_write: int = 0
        # write headers
        if filename:
            self.filenames = [f"{filename}.{i}" for i in range(n_particles)]
            self.write_fmt = write_fmt or "%14.9f"
            fields = ld.pot.get_fields()
            for i, fname in enumerate(self.filenames):
                ifields = ["time"] + [f"pos_{f}.{i}" for f in fields]
                if self.store_momentum:
                    ifields += [f"mom_{f}.{i}" for f in fields]
                initialize_file(fname, ifields)
            if self.stride == 1:
                logstr = f"Saving all positions to the files '{filename}.{{i}}'"
            else:
                logstr = f"Saving every {make_ordinal(self.stride)} position to the files '{filename}.{{i}}'"
            print(logstr)
        print()

    def run(self, step: int) -> None:
        """Store positions in traj array and write to file if write_stride is matched

        The stride parameters is ignored here and all times are temporarily stored
        This is because a HistogramAction using the data might have a different stride

        :param step: current simulation step
        """
        row = (step % self.write_stride) - 1  # saving starts at step 1
        self.times[row] = step * self.ld.dt
        self.positions[row] = [p.pos for p in self.ld.particles]
        if self.store_momentum:
            self.momentum[row] = [p.mom for p in self.ld.particles]

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
            save_times = get_valid_data(self.times, step, self.stride, 1, self.last_write)
            save_pos = get_valid_data(self.positions, step, self.stride, 1, self.last_write)
            if self.store_momentum:
                save_momentum = get_valid_data(self.momentum, step, self.stride, 1, self.last_write)
            for i, filename in enumerate(self.filenames):
                # 3d (times, walkers, pot_dims) to 2d array (times, pot_dims) for saving
                if self.momentum:
                    save_data = np.c_[save_times, save_pos[:, i], save_momentum[:,i]]
                else:
                    save_data = np.c_[save_times, save_pos[:, i]]
                with open(filename, "ab") as f:
                    np.savetxt(
                        f,
                        save_data,
                        delimiter=" ",
                        newline="\n",
                        fmt=self.write_fmt,
                    )
                self.last_write = step
