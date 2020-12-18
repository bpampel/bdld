"""Module containing the abstract base class for all actions

Also has helper functions used by multiple actions"""


import numpy as np


class Action:
    """Abstract base class for all actions"""

    def run(self, step: int):
        """Needs to be defined for all actions"""
        raise NotImplementedError()

    def final_run(self, step: int):
        """If not implemented, do nothing"""
        pass


def get_valid_data(
    data: np.ndarray,
    step: int,
    stride: int,
    write_stride: int,
    last_write: int,
) -> np.ndarray:
    """Get the currently valid rows as a view of the data array

    This is required to for actions storing their data continuously that also want to
    be able to write / reset their data array at any time

    This assumes that the array has exactly the size needed to store the data between regular
    writes, i.e. write_stride // stride, and that the data is stored continuously in the array
    ignoring any actions that would invalidate parts of the data

    :param data: data array whose valid rows are returned, each row represents a point in time
    :param step: current simulation step
    :param stride: update stride
    :param write_stride: data is written to file every n time steps
    :param last_write: when the data was last written (needs to be less than write_stride ago)
    """
    if step == last_write:  # manually check because already_written is 0 due to modulo
        return data[:0]  # empty view
    already_written = last_write % write_stride
    last_element = step % write_stride
    # shortcut for basic case
    if stride == 1 and already_written == 0 and last_element == 0:
        return data
    # shift from stride - number of elements at end that weren't saved last time
    stride_offset = stride - 1 - (last_write % stride)
    first_element = already_written + stride_offset
    if last_element == 0:
        return data[first_element::stride]
    return data[first_element:last_element:stride]
