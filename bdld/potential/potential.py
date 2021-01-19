"""Potential class to be evaluated with md"""

from typing import List, Union, Tuple
import numpy as np

from bdld import grid


class Potential:
    """Base class for potentials

    Collection of some functionality that all should have
    All derived potentials must implement energy() and force() functions
    and set some description in __str__()
    """

    def __init__(self):
        """Define some data members all potentials should set"""
        self.n_dim: int = 0
        self.ranges: List[Tuple[float, float]] = []

    def evaluate(self, pos: Union[List[float], np.ndarray]) -> Tuple[float, np.ndarray]:
        """Get potential energy and forces at position

        :param pos: position to be evaluated
        :return: (energy, forces)
        """
        return (self.energy(pos), self.force(pos))

    def energy(self, pos: Union[List[float], np.ndarray]) -> float:
        """Get energy at position, needs to be overriden by derived class

        :param pos: position to be evaluated
        :return: energy
        """
        raise NotImplementedError()

    def force(self, pos: Union[List[float], np.ndarray]) -> np.ndarray:
        """Get energy at position, needs to be overriden by derived class

        :param pos: position to be evaluated
        :return: array with force per direction
        """
        raise NotImplementedError()

    def __str__(self) -> str:
        """Return some description string for the potential"""
        raise NotImplementedError()

    def calculate_reference(
        self, pos: Union[List[np.ndarray], np.ndarray], mintozero: bool = True
    ) -> np.ndarray:
        """Calculate reference from potential at given positions

        :param pos: positions to evaluate
        :param bool mintozero: shift fes minimum to zero
        :return fes: list numpy array with fes values at positions
        """
        fes = np.fromiter((self.energy(p) for p in pos), np.float64, len(pos))
        if mintozero:
            fes -= np.min(fes)
        return fes

    def calculate_probability_density(
        self,
        kt: float,
        ranges: List[Tuple[float, float]],
        grid_points: List[int],
    ) -> grid.Grid:
        """Calculate the probability density associated with the potential on a grid

        :param kt: Thermal energy of system
        :param ranges: List of ranges of the grid per dimension (min, max)
        :param grid_points: number of points per dimension
        :return grid: meshgrid with the positions of the points
        :return prob: normalized probablities at the grid points
        """
        if len(ranges) != self.n_dim:
            raise ValueError("Dimension of ranges do not match potential")
        if len(grid_points) != self.n_dim:
            raise ValueError("Dimension of grid_points do not match potential")
        fes = grid.from_npoints(ranges, grid_points)
        fes.set_from_func(self.energy)
        prob = np.exp(-fes / kt)
        # normalize with volume element from stepsizes
        prob /= np.sum(prob.data) * np.prod(prob.stepsizes)
        return prob

    def get_fields(self) -> List[str]:
        """Return list of identifiers for the potential dimensions

        Can be overwritten by subclasses to have custom names
        """
        if self.n_dim == 1:
            return ["x"]
        elif self.n_dim == 2:
            return ["x", "y"]
        elif self.n_dim == 3:
            return ["x", "y", "z"]
        else:
            raise ValueError("Class can't be used for more than 3 dimensions")