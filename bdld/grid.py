"""Custom grid class that holds also the data"""

from typing import Any, Callable, List, Union, Tuple
import operator
import numpy as np
from scipy import signal


class Grid:
    """Rectangular grid with evenly distributed points

    :param data: Values at the grid points
    :param ranges: (min, max) of the grid points per dimension
    :param stepsizes: stepsizes between grid points per dimension
    :param n_points: number of points per dimension
    """

    def __init__(self) -> None:
        """Create empty, uninitialized class. Should usually not invoked directly"""
        self.data: np.array = None
        self.ranges: List[Tuple[float, float]] = []
        self.n_points: List[int] = []
        self.stepsizes: List[float] = []
        self.n_dim: int = 0

    def __add__(self, other):
        """Return new grid instance with sum in data"""
        return self._perform_math_operation(other, operator.add)

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        return self._perform_math_operation(other, operator.sub)

    # no __rsub__ or __rdiv__ as that is ambiguous

    def __mul__(self, other):
        return self._perform_math_operation(other, operator.mul)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        return self._perform_math_operation(other, operator.truediv)

    def __floordiv__(self, other):
        return self._perform_math_operation(other, operator.floordiv)

    def __mod__(self, other):
        return self._perform_math_operation(other, operator.mod)

    def __pow__(self, other):
        return self._perform_math_operation(other, operator.pow)

    def _perform_math_operation(
        self, other: Union[float, int, np.ndarray], oper: Callable[..., Any]
    ):  # -> Grid:
        new_grid = self.copy_empty()
        if isinstance(other, (float, int, np.ndarray)):
            new_grid.data = oper(self.data, other)
        elif isinstance(other, Grid):
            if not (self.n_points == other.n_points) or not (
                self.ranges == other.ranges
            ):
                raise ValueError(
                    "Performing math operations on grids with different points is ambiguous"
                )
            new_grid.data = oper(self.data, other.data)
        else:
            raise ValueError(
                f"Performing math operations with grid and {type(other)} is not supported"
            )
        return new_grid

    def _perform_imath_operation(
        self, other: Union[float, int, np.ndarray], oper: Callable[..., Any]
    ):  # -> Grid:
        """Augmented operations like += change self.data instead of returning new grid"""
        if isinstance(other, (float, int, np.ndarray)):
            self.data = oper(self.data, other)
        elif isinstance(other, Grid):
            if not (self.n_points == other.n_points) or not (
                self.ranges == other.ranges
            ):
                raise ValueError(
                    "Performing math operations on grids with different points is ambiguous"
                )
            self.data = oper(self.data, other.data)
        else:
            raise ValueError(
                f"Performing math operations with grid and {type(other)} is not supported"
            )
        return self

    def axes(self) -> List[np.ndarray]:
        """Return list of grid axes per dimension"""
        return [
            np.linspace(*self.ranges[i], self.n_points[i]) for i in range(self.n_dim)
        ]

    def points(self) -> np.ndarray:
        """Return all grid points as a array of shape (n_points, dim) in row-major order"""
        return np.array(np.meshgrid(*self.axes())).T.reshape(
            np.prod(self.n_points), self.n_dim
        )

    def as_array(self) -> np.ndarray:
        """Return data as structured array in row-major order """
        return self.data.reshape(self.n_points)

    def set_from_func(self, func: Callable[..., float]) -> None:
        """Set data by applying function to all points

        The function must accept the input in"""
        self.data = np.array([func(p) for p in self.points()])

    def copy_empty(self):
        """Get a new grid instance with the same points but without data"""
        return from_npoints(self.ranges, self.n_points)


def convolve(g1: Grid, g2: Grid, mode: str = "valid") -> Grid:
    """Perform convolution between two grids via scipy.signal.convolve

    Grids must have same dimensions and stepsizes.
    The convolution is also correctly normalized by the stepsizes.

    :param g1, g2: grids to convolute
    :param mode: convolution mode, see scipy.signal.convolve for details
    :return grid: New grid containing the convolutin
    """
    if not np.all(g1.stepsizes == g2.stepsizes):
        raise ValueError("Spacing of grids does not match")
    stepsizes = g1.stepsizes
    n_dim = g1.n_dim
    # to get the same values in the continuous limit: multiply by stepsizes
    conv = signal.convolve(g1.as_array(), g2.as_array(), mode=mode) * np.prod(
        g1.stepsizes
    )
    # import pdb; pdb.set_trace()
    # also get the corresponding grid points depending on the method
    if mode == "same":  # easiest case, mirrors grid of first argument
        grid = g1.copy_empty()
    if mode == "valid":  # need to do some math: subtract smaller from larger grid
        n_points = []
        ranges = []
        # sort by points to find smaller grid (conv throws error if not in all dim)
        gs, gl = sorted([g1, g2], key=lambda g: g.n_points)
        for dim in range(n_dim):
            n_points.append(gl.n_points[dim] - gs.n_points[dim] + 1)
            offset = (gs.n_points[dim] - 1) / 2 * stepsizes[dim]
            ranges.append((gl.ranges[dim][0] + offset, gl.ranges[dim][1] - offset))
        grid = from_npoints(ranges, n_points)
    if mode == "full":
        raise ValueError("Currently not implemented")
    grid.data = conv
    return grid


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
        n_points_tmp = int(np.ceil((r[1] - r[0]) / stepsizes[i])) + 1 - int(shrink)
        r = (r[0], r[0] + stepsizes[i] * n_points_tmp)
        grid.n_points.append(n_points_tmp)
    grid.ranges = ranges
    return grid
