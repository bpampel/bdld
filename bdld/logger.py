"""Logger"""

# from typing import List, Optional, Tuple, Union
# import numpy as np

from bdld.helpers.plumed_header import PlumedHeader as PlmdHeader
from bdld.helpers.misc import backup_if_existing


class Logger:
    """Class that writes data to a file periodically

    :param filename: file to write to
    :param data_function: function to call to receive data
    :param header: file header
    :param stride:
    :param stride_offset:
    """

    def __init__(self, filename: str, header: str) -> None:
        self.filename = filename
        self.header = header

        # write header
        backup_if_existing(filename)
        with open(self.filename, "w") as f:
            f.write(header + "\n")

    def write(self):
        data = self.data_function()
        with open(filename, "ab") as f:
            np.savetxt(f, data, delimiter=" ", newline="\n")
