"""Custom grid class that holds also the data"""

from typing import List, Union, Tuple
import numpy as np


class Grid:
    """Rectangular grid with evenly distributed points

    :param data: Values at the grid points
    :param ranges: (min, max) of the grid points per dimension
    :param stepsizes: stepsizes between grid points per dimension
    :param n_points: number of points per dimension
    """

    def __init__(self):
        """Create empty, uninitialized class. Should usually not invoked directly"""
        self.data: np.array = None
        self.ranges: List[Tuple[float, float]] = []
        self.n_points: List[int] = []
        self.stepsizes: List[float] = []
        self.n_dim: int = 0

    def axes(self) -> List[np.ndarray]:
        """Return list of grid axes per dimension"""
        return [
            np.linspace(*self.ranges[i], self.n_points[i]) for i in range(self.n_dim)
        ]

    def points(self) -> np.ndarray:
        """Return all grid points as a array of shape (n_points, dim)"""
        return np.array(np.meshgrid(*self.axes())).reshape(
            np.prod(self.n_points), self.n_dim
        )


def from_npoints(
    ranges: List[Tuple[float, float]], n_points: Union[List[int], int]
) -> Grid:
    """Create grid from the number of points per direction

    :param ranges: List with (min,max) positions of the grid
    :param n_points: Either list with points per dimension or single value for all
    """
    grid = Grid()
    grid.n_dim = len(ranges)
    if isinstance(n_points, int):
        if grid.n_dim == 1:
            n_points = [n_points]
        else:  # expand to all dimensions
            n_points = [n_points] * grid.n_dim
    if len(n_points) != grid.n_dim:
        raise ValueError("Dimensions of ranges and number of points do not match")
    grid.n_points = n_points
    grid.stepsizes = [(r[1] - r[0]) / (n_points[i] - 1) for i, r in enumerate(ranges)]
    grid.ranges = ranges
    return grid


def from_stepsizes(
    ranges: List[Tuple[float, float]],
    stepsizes: Union[List[float], float],
    shrink: bool = False,
) -> Grid:
    """Create grid from stepsizes between data points

    If the stepsizes doesn't exactly fit the ranges, the upper grid ranges are extended by default

    :param ranges: List with (min,max) positions of the grid
    :param stepsizes: Either list with stepsizes per dimension or single value for all
    :param shrink: Shrink ranges instead of expanding if stepsizes doesn't fit exactly
    """
    grid = Grid()
    grid.n_dim = len(ranges)
    if isinstance(stepsizes, float):
        if grid.n_dim == 1:
            stepsizes = [stepsizes]
        else:  # expand to all dimensions
            stepsizes = [stepsizes] * grid.n_dim
    grid.stepsizes = stepsizes
    for i, r in enumerate(ranges):
        n_points_tmp = int(np.ceil((r[1] - r[0]) / stepsizes[i])) - int(shrink)
        r = (r[0], r[0] + stepsizes[i] * n_points_tmp)
        grid.n_points.append(n_points_tmp)
    grid.ranges = ranges
    return grid
