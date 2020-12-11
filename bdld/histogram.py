"""Implement a simple histogramming and FES calculation class"""

from typing import List, Optional, Union, Tuple
import numpy as np

from bdld import grid


class Histogram(grid.Grid):
    """Histogram data and calculate FES from the histogram

    This uses the Grid class for underlying structure and only adds some histogram functions
    Also explicitely stores the bin edges (as opposed to only the n_points of the Grid class)

    Also allows histogramming over time, i.e. adding more data to the existing histogram

    :param n_points: number of bins for histogramming per dimension
    :param ranges: extent of histogram (min, max) per dimension
    :param bins: bin edges of the histogram per dimension
    :param data: histogram data
    :param fes: the free energy values corresponding to the histogram stored in data
    """

    def __init__(
        self,
        n_bins: Union[List[int], int],
        ranges: List[Tuple[float, float]],
    ):
        """Set up empty histogram instance

        :param n_bins: number of bins for histogramming per dimension
        :param ranges: extent of histogram (min, max) per dimension
        """
        super().__init__()
        if not isinstance(n_bins, list):  # single float
            n_bins = [n_bins]
        self.n_points = n_bins
        self.stepsizes = grid.stepsizes_from_npoints(ranges, n_bins)
        self.ranges = ranges
        self.n_dim = len(ranges)
        self.fes: Optional[np.ndarray] = None
        # create bins from arbitrary value, there doesn't seem to be a function doing it
        self.data, self.bins = np.histogramdd(
            np.zeros((1, len(self.n_points))), bins=self.n_points, range=self.ranges
        )

    def add(self, data: np.ndarray) -> None:
        """Add data to histogram

        :param data: The values to add to the histogram, see numpy's histogramdd for details
        :type data: list (1d), list of lists or numpy.ndarrays (arbitrary dimensions)
        """
        tmp_histo, _ = np.histogramdd(data, bins=self.bins)
        self.data += tmp_histo

    def bin_centers(self):
        """Calculate the centers of the histogram bins from the bin edges

        :return centered_bins: the centers of the histogram bins
        :type centered_bins: list with numpy.ndarray per dimension
        """
        return [
            np.array(
                [(bins_x[i] + bins_x[i + 1]) / 2 for i in range(0, len(bins_x) - 1)]
            )
            for bins_x in self.bins
        ]

    def axes(self):
        """Overwrite function from base class: Axes should return the bin centers

        The base function would return them with points at the borders"""
        return self.bin_centers()

    def calculate_fes(self, kt: float, mintozero: bool = True) -> grid.Grid:
        """Calculate free energy surface from histogram

        Overwrites the fes attribute from the class instance and returns the data
        in plottable form

        :param float kt: thermal energy of the system
        :param bool mintozero: shift FES to have minimum at zero

        :return fes: Grid with the fes values as data
        """
        fes = np.where(
            self.data == 0, np.inf, -kt * np.log(self.data, where=(self.data != 0))
        )
        if mintozero:
            fes -= np.min(fes)
        self.fes = fes
        return self.get_fes_grid()

    def get_fes_grid(self) -> grid.Grid:
        """Returns the fes as Grid instead of numpy array"""
        new_grid = self.grid_from_histo()
        new_grid.data = self.fes
        return new_grid

    def grid_from_histo(self) -> grid.Grid:
        """Return grid with the same points instead of bins"""
        ranges = [(a[0], a[-1]) for a in self.axes()]
        return grid.from_npoints(ranges, self.n_points)
