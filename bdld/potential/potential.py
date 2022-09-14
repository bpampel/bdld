"""Potential class to be evaluated with md"""

import enum
from typing import Callable, List, Union, Tuple
import numpy as np

from bdld import grid


class BoundaryCondition(enum.Enum):
    """Enum for the different boundary conditions"""

    reflective = enum.auto()
    periodic = enum.auto()


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
        self._boundary_condition = None  # default
        self.apply_boundary_condition: Callable = lambda: None
        self._set_boundary_condition_function()

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
        :raises ValueError: if dimensions of points or ranges and potential do not match
        :return prob: grid with normalized probablities
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

    @property
    def boundary_condition(self):
        """Type of boundary condition for the potential"""
        return self._boundary_condition

    @boundary_condition.setter
    def boundary_condition(self, cond: BoundaryCondition):
        """Change the type of boundary condition

        This also updates the apply_boundary_condition function
        """
        self._boundary_condition = cond
        self._set_boundary_condition_function()

    def _set_boundary_condition_function(self) -> None:
        """Set correct apply_boundary_condition function"""
        if self.boundary_condition is None:
            func = lambda pos, force: None  # do nothing
        elif self.boundary_condition == BoundaryCondition.reflective:
            func = self.apply_boundary_condition_reflective
        elif self.boundary_condition == BoundaryCondition.periodic:
            func = self.apply_boundary_condition_periodic
        else:
            raise ValueError("Unknown boundary condition set")
        self.apply_boundary_condition = func

    def apply_boundary_condition_reflective(
        self, pos: np.ndarray, mom: np.ndarray
    ) -> None:
        """Apply reflective boundary condition

        If the particle is outside the potential range, it is set to the boundary
        and its momentum is reversed

        Because position and momentum are numpy arrays, they are passed by reference and
        can be changed in place

        :param pos: position of particle per direction
        :param mom: momentum of particle per direction
        """
        for i, x in enumerate(pos):
            if x < self.ranges[i][0]:
                pos[i] = self.ranges[i][0]
                mom[i] = -mom[i]
            elif x > self.ranges[i][1]:
                pos[i] = self.ranges[i][1]
                mom[i] = -mom[i]

    def apply_boundary_condition_periodic(
        self, pos: np.ndarray, mom: np.ndarray
    ) -> None:
        """Apply periodic boundary condition

        If the particle is outside the potential range, it is moved to the other side of the
        potential range
        Because position and momentum are numpy arrays, they are passed by reference and
        can be changed in place

        :param pos: position of particle per direction
        :param mom: momentum of particle per direction (not actually changed)
        """
        for i, x in enumerate(pos):
            if x < self.ranges[i][0]:
                pos[i] += self.ranges[i][1] - self.ranges[i][0]
            elif x > self.ranges[i][1]:
                pos[i] -= self.ranges[i][1] - self.ranges[i][0]
