"""Collect helper functions that are required by multiple files"""

from typing import List, Tuple

import numpy as np


def pos_inside_ranges(
    pos: np.ndarray, ranges: List[List[Tuple[float, float]]]
) -> List[np.ndarray]:
    """Check if positions are within given ranges
    :param pos: 2d numpy array of the positions. First index is the different positions, second the dimensions
    :param ranges: list of the ranges to check. Each range is a list of (min, max) tuples per dimension
    :return inside_list: List of boolean numpy arrays indicating if position was inside
    """
    n_dim = pos.shape[-1]
    inside_list: List[np.ndarray] = []
    for state in ranges:
        inside = True
        for i in range(n_dim):  # 'bitwise and' over all dimensions
            inside = inside & (pos[:, i] >= state[i][0]) & (pos[:, i] <= state[i][1])
        inside_list.append(inside)
    return inside_list
