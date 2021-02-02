"""Module holding action to analyse distribution of particles in states"""

from collections import OrderedDict
import logging
from typing import List, Optional, Tuple

import numpy as np

from bdld.actions.action import Action, get_valid_data
from bdld.actions.bussi_parinello_ld import BpldParticle
from bdld.helpers.misc import initialize_file
from bdld.tools import pos_inside_ranges


class ParticleDistributionAction(Action):
    """Class that stories trajectories and writes them to file

    The write_stride parameter determines how many trajectory
    points are being held in memory even if they are never written.
    This allow other actions to use them.

    :param counts: fixed size numpy array holding the time and particle counts per state
                   it is written row-wise (i.e. every row represents a time)
                   and overwritten after being saved to file
    """

    def __init__(
        self,
        particles: List[BpldParticle],
        states: List[List[Tuple[float, float]]],
        stride: Optional[int] = None,
        filename: Optional[str] = None,
        write_stride: Optional[int] = None,
        write_fmt: Optional[str] = None,
    ) -> None:
        """Set up action to analyse particle distribution

        :param particles: Particle list of LD to analyise
        :param states: List of states to check. Each state is a list of (min, max) ranges per dimension.
        :param stride: write every nth time step to file, default 1
        :param filename: base of filename(s) to write to
        :param write_stride: write to file every n time steps, default 100
        :param write_fmt: numeric format for saving the data, default "%14.9f"
        """
        print("Setting up action to analyse distribution of walkers")
        self.particles = particles
        self.states = states
        self.stride: int = stride or 1
        self.write_stride: int = write_stride or 100

        if self.stride:
            self.write_stride = write_stride or self.stride * 100
            if self.write_stride % self.stride != 0:
                e = "The write stride must be a multiple of the update stride."
                raise ValueError(e)
            if not filename:
                e = "Specifying a write_stride but no filename makes no sense"
                raise ValueError(e)
            # extra col for time
            self.counts = np.empty((self.write_stride // self.stride, len(self.states) + 1))
            self.last_write: int = 0
        else:  # just store one data set
            self.counts = np.empty((len(self.states) +1))
            print(f"Saving every {self.stride} point to '{filename}'")

        # writing
        self.filename = filename
        if self.filename:
            fields = [f"n_particles_{i}" for i in range(len(self.states))]
            constants = OrderedDict()  # add state ranges to file header
            for i, state in enumerate(self.states):
                constants[f"state_{i}"] = str(state)
            initialize_file(self.filename, fields, constants)
        self.write_fmt = write_fmt if write_fmt else "%14.9f"

    def run(self, step: int) -> None:
        """Calculate Delta F from fes and write to file

        :param step: current simulation step, optional
        """
        if self.stride and step % self.stride == 0:
            row = (step % self.write_stride) // self.stride - 1
            self.counts[row, 0] = step
            self.counts[row, 1:] = self.count_particles()
        if self.write_stride and step % self.write_stride == 0:
            self.write(step)

    def final_run(self, step: int) -> None:
        if not self.stride:  # perform analysis once
            self.counts[0] = step
            self.counts[1:] = self.count_particles()
        self.write(step)

    def count_particles(self) -> List[int]:
        """Count particles currently inside the states

        :return counts: list with number of particles per state
        """
        pos = np.array([p.pos for p in self.particles])
        bool_lists = pos_inside_ranges(pos, self.states)
        counts = [sum(state_bools) for state_bools in bool_lists]
        return counts

    def write(self, step: int) -> None:
        """Write fes to file

        If a step is specified it will be appended to the filename, i.e. it is written
        to a new file.
        If no filename is set, this will do nothing.

        :param step: current simulation step
        """
        if self.filename:
            if self.stride:
                save_data = get_valid_data(
                    self.counts, step, self.stride, self.stride, self.last_write
                )
                self.last_write = step
            else:  # reshape to rows to save as single line
                save_data = self.counts.reshape((1, -1))
            with open(self.filename, "ab") as f:
                np.savetxt(
                    f,
                    save_data,
                    delimiter=" ",
                    newline="\n",
                    fmt=self.write_fmt,
                )
