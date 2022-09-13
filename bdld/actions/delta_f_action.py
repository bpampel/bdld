"""Module holding the DeltaFAction class"""

from collections import OrderedDict
from typing import List, Optional

import numpy as np

from bdld.actions.action import Action, get_valid_data
from bdld.actions.fes_action import FesAction
from bdld.helpers.misc import initialize_file, make_ordinal


class DeltaFAction(Action):
    """Calculate Delta F from fes and print or save it to file

    If no filename is specified this still calculates the values (for possible usage
    in other actions) but never actually writes them to file

    """

    def __init__(
        self,
        fes_action: FesAction,
        masks: List[np.ndarray],
        stride: Optional[int] = None,
        filename: Optional[str] = None,
        write_stride: Optional[int] = None,
        write_fmt: Optional[str] = None,
        ref: Optional[np.ndarray] = None,
    ) -> None:
        """Set up action that calculates free energy differences between states

        If no stride is given, this action is not run periodically
        but can manually be triggered (e.g. with final_run()).
        If a stride is given, it needs to be a multiple of the stride of the
        FesAction to make sense

        :param fes_action: Fes action to analyise
        :param masks: List of masks that define the states
        :param stride: calculate delta f every n time steps, optional
        :param filename: filename to save fes to, optional
        :param write_stride: write to file every n time steps, default 0 (never)
        :param write_fmt: numeric format for saving the data, default "%14.9f"
        :param ref: reference FES to calculate reference values
        """
        print("Setting up delta-f action")
        self.fes_action = fes_action
        check_mask_shapes(masks, fes_action.fes.data)
        self.masks = masks
        self.stride = stride or 0
        self.write_stride = 0  # only set this if a stride is specified
        if self.stride:
            if not self.fes_action.stride or self.stride % self.fes_action.stride != 0:
                print("Warning: the DeltaF stride is no multiple of the FES stride.")
            if write_stride and not filename:
                e = "Specifying a write_stride but no filename makes no sense"
                raise ValueError(e)
            self.write_stride = write_stride or self.stride * 100
            if self.write_stride % self.stride != 0:
                e = "The write stride must be a multiple of the update stride."
                raise ValueError(e)
            # time + (masks -1) values per evaluation
            self.delta_f = np.empty((self.write_stride // self.stride, len(self.masks)))
            self.last_write: int = 0
        else:  # just store one data set
            self.delta_f = np.empty((len(self.masks)))

        # copy temp and timestep from LD
        self.kt = self.fes_action.histo_action.traj_action.ld.kt
        self.dt = self.fes_action.histo_action.traj_action.ld.dt

        if ref:
            self.ref_values = calculate_delta_f(ref, self.kt, self.masks)

        # writing
        self.filename = filename
        if self.filename:
            fields = [f"delta_f.1-{i}" for i in range(2, len(self.masks) + 1)]
            constants = OrderedDict([("kt", self.kt)])
            if ref:
                for i, val in enumerate(self.ref_values):
                    constants[f"ref_{fields[i]}"] = val
            initialize_file(self.filename, fields, constants)
            stride_str = f"of every {make_ordinal(self.stride)} time step " if self.stride else ""
            print(
                "Saving delta-f " + stride_str + f"to '{filename}'"
            )
        self.write_fmt = write_fmt if write_fmt else "%14.9f"
        print()

    def run(self, step: int) -> None:
        """Calculate Delta F from fes and write to file

        :param step: current simulation step, optional
        """
        if self.stride and step % self.stride == 0:
            row = (step % self.write_stride) // self.stride - 1
            self.delta_f[row, 0] = step * self.dt
            self.delta_f[row, 1:] = calculate_delta_f(
                self.fes_action.fes.data, self.kt, self.masks
            )
        if self.write_stride and step % self.write_stride == 0:
            self.write(step)

    def final_run(self, step: int) -> None:
        if not self.stride:  # perform analysis once
            self.delta_f[0] = step * self.dt
            self.delta_f[1:] = calculate_delta_f(
                self.fes_action.fes.data, self.kt, self.masks
            )
        self.write(step)

    def write(self, step: int) -> None:
        """Write delta F values to file to file

        Does nothing if self.filename is empty

        :param step: current simulation step
        """
        if self.filename:
            if self.stride:
                save_data = get_valid_data(
                    self.delta_f, step, self.stride, self.stride, self.last_write
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


def check_mask_shapes(masks: List[np.ndarray], fes: np.ndarray) -> None:
    """Check if the shape of all masks is equal to the shape of the FES"""
    if any(mask.shape != fes.shape for mask in masks):
        raise ValueError("Shapes of mask not equal to FES shape")


def calculate_delta_f(fes: np.ndarray, kt: float, masks: List[np.ndarray]):
    """Calculates the free energy difference between states defined by boolean masks

    If more than two are specified, this returns the difference to the first state for all others

    :param fes: free energy surface to examine
    :param kt: energy in units of kT
    :param masks: a list of boolean numpy arrays resembling the states

    :return delta_F: a list of doubles containing the free energy difference to the first state
    :raises IndexError: if the dimensions of the FES and any of the masks do not match
    """
    probabilities = np.exp(-fes / float(kt)).reshape((-1))
    state_probs = [np.sum(probabilities[m]) for m in masks]
    delta_f = [
        -kt * np.log(state_probs[i] / state_probs[0])
        for i in range(1, len(state_probs))
    ]
    return delta_f
