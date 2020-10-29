"""Custom grid class that holds also the data"""

from typing import Any, Callable, List, Union, Tuple
import operator
import numpy as np
from scipy import signal
from scipy import interpolate as sp_interpolate

from bdld.helpers.misc import write_2d_sliced_to_file


class Grid:
    """Rectangular grid with evenly distributed points

    :param _data: Values at the grid points
    :param ranges: (min, max) of the grid points per dimension
    :param stepsizes: stepsizes between grid points per dimension
    :param n_points: number of points per dimension
    """

    def __init__(self) -> None:
        """Create empty, uninitialized class. Should usually not invoked directly"""
        self._data: np.ndarray = np.empty(0)
        self.ranges: List[Tuple[float, float]] = []
        self.n_points: List[int] = []
        self.stepsizes: List[float] = []
        self.n_dim: int = 0

    @property
    def data(self):
        """Data values of the grid as numpy array

        Setting data broadcasts the values to the shape given by n_points
        """
        return self._data

    @data.setter
    def data(self, value):
        """Setter checks if given values match grid points"""
        try:
            self._data = value.reshape(self.n_points)
        except ValueError as e:
            raise ValueError("Data does not fit into grid points") from e

    def __pos__(self):
        return self._perform_math_on_self(operator.pos)

    def __neg__(self):
        return self._perform_math_on_self(operator.neg)

    # allow arithmetic operators to manipulate data directly
    def __add__(self, other: Union[float, int, np.ndarray]):
        return self._perform_arithmetic_operation(other, operator.add)

    def __radd__(self, other: Union[float, int, np.ndarray]):
        return self.__add__(other)

    def __sub__(self, other: Union[float, int, np.ndarray]):
        return self._perform_arithmetic_operation(other, operator.sub)

    # no __rsub__ or __rdiv__ as that is ambiguous

    def __mul__(self, other: Union[float, int, np.ndarray]):
        return self._perform_arithmetic_operation(other, operator.mul)

    def __rmul__(self, other: Union[float, int, np.ndarray]):
        return self.__mul__(other)

    def __truediv__(self, other: Union[float, int, np.ndarray]):
        return self._perform_arithmetic_operation(other, operator.truediv)

    def __floordiv__(self, other: Union[float, int, np.ndarray]):
        return self._perform_arithmetic_operation(other, operator.floordiv)

    def __mod__(self, other: Union[float, int, np.ndarray]):
        return self._perform_arithmetic_operation(other, operator.mod)

    def __pow__(self, other: Union[float, int, np.ndarray]):
        return self._perform_arithmetic_operation(other, operator.pow)

    # exp and log implementations allow usage of e.g. np.exp(my_grid)
    def exp(self):
        """Exponentiation of data, relies on numpy"""
        return self._perform_math_on_self(np.exp)

    def log(self):
        """Logarithm of data, relies on numpy"""
        return self._perform_math_on_self(np.log)

    def _perform_math_on_self(self, oper: Callable[..., Any]):  # -> Grid:
        new_grid = self.copy_empty()
        new_grid.data = oper(self.data)
        return new_grid

    def _perform_arithmetic_operation(
        self, other: Union[float, int, np.ndarray], oper: Callable[..., Any]
    ):  # -> Grid::
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

    def set_from_func(self, func: Callable[..., float]) -> None:
        """Set data by applying function to all points

        The function must accept the input in"""
        self.data = np.array([func(p) for p in self.points()])

    def copy_empty(self):
        """Get a new grid instance with the same points but without data"""
        return from_npoints(self.ranges, self.n_points)

    def interpolate(
        self, points: Union[np.ndarray, Tuple[np.ndarray, ...]], method: str = "linear"
    ) -> np.ndarray:
        """Interpolate grid data at the given points

        The interpolation is done via scipy.interpolate.griddata, see there for details

        :param points: the desired points
        :param method: interpolation method to use, defaults to linear
        """
        return sp_interpolate.griddata(
            self.points(), self.data.flatten(), points, method
        )

    def write_to_file(
        self, filename: str, fmt: str = "%.18e", header: str = ""
    ) -> None:
        """Write the grid to file via numpy.savetxt

        For 2d this will write the data in "plumed style", i.e. in C-order with emtpy lines
        between the rows

        :param filename: Path to write to
        :param fmt: format of the data, see np.savetxt
        :param header: String for the header, must start with the appropriate comment char
        """
        if self.n_dim == 1:
            np.savetxt(
                filename,
                np.c_[self.axes()[0], self.data],
                fmt=fmt,
                header=header,
                delimiter=" ",
                newline="\n",
                comments="",
            )
        elif self.n_dim == 2:
            write_2d_sliced_to_file(
                filename,
                np.concatenate(
                    (self.points(), self.data.reshape((np.prod(self.n_points), 1))),
                    axis=1,
                ),
                self.n_points,
                fmt,
                header,
            )


def convolve(g1: Grid, g2: Grid, mode: str = "valid", method: str = "auto") -> Grid:
    """Perform convolution between two grids via scipy.signal.convolve

    Grids must have same dimensions and stepsizes.
    The convolution is also correctly normalized by the stepsizes.

    :param g1, g2: grids to convolute
    :param mode: convolution mode, see scipy.signal.convolve for details
    :return grid: New grid containing the convolutin
    """
    if not g1.stepsizes == g2.stepsizes:
        raise ValueError("Spacing of grids does not match")
    stepsizes = g1.stepsizes
    n_dim = g1.n_dim
    # to get the same values in the continuous limit: multiply by stepsizes
    conv = signal.convolve(g1.data, g2.data, mode=mode, method=method) * np.prod(
        g1.stepsizes
    )
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


def stepsizes_from_npoints(ranges: List[Tuple[float,float]], n_points: List[int]) -> List[float]:
    """Calculate the stepsizes from the number of points and ranges

    :param ranges: Ranges of the grid
    :param n_points: number of points per dimension
    """
    return [(r[1] - r[0]) / (n_points[i] - 1) for i, r in enumerate(ranges)]


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
    grid.stepsizes = stepsizes_from_npoints(ranges, n_points)
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
    if isinstance(stepsizes, (float, int)):
        if grid.n_dim == 1:
            stepsizes = [stepsizes]
        else:  # expand to all dimensions
            stepsizes = [stepsizes] * grid.n_dim
    grid.stepsizes = stepsizes
    for i, r in enumerate(ranges):
        n_points_tmp = int(np.ceil((r[1] - r[0]) / stepsizes[i])) + 1 - int(shrink)
        grid.ranges.append((float(r[0]), r[0] + stepsizes[i] * (n_points_tmp - 1)))
        grid.n_points.append(n_points_tmp)
    return grid
