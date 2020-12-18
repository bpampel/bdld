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
    update_stride: int,
    last_write: int,
) -> np.ndarray:
    # thoughts: following the assumptions we don't need all the info
    #    - get either update_stride or write_stride from data.shape[0]
    """Get the currently valid rows as a view of the data array

    This is required to for actions storing their data continuously that also want to
    be able to write / reset their data array at any time

    The data array is required to have the following properties:
    - it has exactly write_stride // update_stride elements
    - it is filled every update_stride, the first step is 1, not 0
    - after each multiple of write_stride it is rewritten from the beginning,
      all previously stored data is no longer valid

    This results in several assumptions on the arguments, which are not checked!
    - the write_stride must be a multiple of the update_stride
    - the last_write argument should not refer to a step before the last rewrite
      from the beginning
    - the stride argument must be a multiple of the update_stride
    if these are not satisfied the returned data will not be correct.

    The stride argument makes the function only return the data of every nth time step.
    It is placed here for calling this function regularly, because if the array size is
    not a multiple, an offset has to be taken into account from the previous times

    :param data: data array whose valid rows are returned, each row represents a point in time
    :param step: current simulation step
    :param stride: Return only data points of every nth time step (not stride of data points!)
    :param update_stride: number of time steps between data points
    :param last_write: when the data was last written (needs to be less than write_stride ago)
    """
    if step == last_write:  # manual check because already_written is 0 due to modulo
        return data[:0]  # empty view

    write_stride = data.shape[0] * update_stride  # wrapping time steps of data
    already_written = (last_write % write_stride) // update_stride
    last_element = (step % write_stride) // update_stride
    # shortcut for basic case
    if stride == update_stride and already_written == 0 and last_element == 0:
        return data
    # omit already written points
    first_element = already_written
    # stride: also omit first n-1 elements of data
    first_element += (stride // update_stride) - 1
    # but shift back by the number that was cut off at the last write
    first_element -= (last_write % stride) // update_stride
    if last_element == 0:
        return data[first_element :: stride // update_stride]
    return data[first_element : last_element : stride // update_stride]
