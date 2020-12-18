"""Module holding the DeltaFAction class"""

from typing import Optional

import numpy as np

from bdld import analysis
from bdld.actions.action import Action, get_valid_data
from bdld.actions.fes_action import FesAction
from bdld.helpers.plumed_header import PlumedHeader


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
        ref: Optional[np.ndarray] = None,
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
        :param ref: reference FES to calculate reference values
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

        if ref:
            self.ref_values = analysis.calculate_delta_f(ref, self.kt, self.masks)

    def run(self, step: int) -> None:
        """Calculate Delta F from fes and write to file

        :param step: current simulation step, optional
        """
        if self.stride and step % self.stride == 0:
            row = (step % self.write_stride) // self.stride - 1
            self.delta_f[row, 0] = step * self.dt
            self.delta_f[row, 1:] = analysis.calculate_delta_f(
                self.fes_action.histo_action.histo.fes, self.kt, self.masks
            )
        if self.write_stride and step % self.write_stride == 0:
            self.write(step)

    def final_run(self, step: int) -> None:
        self.delta_f[0] = step * self.dt
        self.delta_f[1:] = analysis.calculate_delta_f(
            self.fes_action.histo_action.histo.fes, self.kt, self.masks
        )
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
                    self.delta_f, step, 1, self.stride, self.last_write
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
