"""Potential class to be evaluated with md"""

from typing import Callable, List, Optional, Union, Tuple
import numpy as np

poly = np.polynomial.polynomial


class Potential:
    """Simple class holding a polynomial potential

    :param coeffs: Coefficients of polynomial potential
    :param der: Coefficients of derivative of potential per direction
    :param n_dim: Dimensions of potential
    :param polyval: polyval function of np.polynomial.polynomial to use
    :param ranges: (min, max) values of potential (optional)
    """

    def __init__(
        self,
        coeffs: Union[List[float], np.ndarray],
        ranges: List[Tuple[float, float]] = None,
    ) -> None:
        """Set up from given coefficients

        :param coeffs: The coefficient i,j,k has to be given in coeffs[i,j,k]
        :param ranges: (min, max) values of potential (optional)
        """
        self.coeffs: np.ndarray = np.array(coeffs)
        if self.coeffs.ndim > 3:
            raise ValueError(
                "Class can't be used for potentials in more than 3 dimensions"
            )
        self.n_dim: int = self.coeffs.ndim
        # note: the derivative matrices are larger than needed. Implement trim_zeros for multiple dimensions?
        self.der: List[np.ndarray] = [
            poly.polyder(self.coeffs, axis=d) for d in range(self.n_dim)
        ]
        self.polyval = self.choose_polyval()
        # the ranges are at the moment not actually checked when evaluating but needed for the birth/death
        if ranges is None:
            ranges = []  # mutable default arguments are bad
        self.ranges: List[Tuple[float, float]] = ranges

    def __str__(self) -> str:
        """Give out coefficients"""
        return "polynomial with coefficients " + list(self.coeffs).__str__()

    def choose_polyval(
        self,
    ):  # -> Callable[[np.ndarray, np.ndarray, Optional[bool]], np.ndarray]:
        """Selects polyval function from numpy.polynomial.polynomial depending on self.n_dim"""
        if self.n_dim == 1:
            return poly.polyval
        elif self.n_dim == 2:
            return poly.polyval2d
        elif self.n_dim == 3:
            return poly.polyval3d
        else:
            raise ValueError("No polyval function for more than 3 dimensions")

    def evaluate(
        self, pos: Union[List[float], np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Get potential energy and forces at position

        :param pos: position to be evaluated
        :return: (energy, forces)
        """
        energy = self.polyval(*pos, self.coeffs)
        forces = np.array([-self.polyval(*pos, self.der[d]) for d in range(self.n_dim)])
        return (energy, forces)

    def calculate_reference(
        self, pos: Union[List[float], np.ndarray], mintozero: bool = True
    ) -> np.ndarray:
        """Calculate reference from potential at given positions

        :param pos: positions to evaluate
        :param bool mintozero: shift fes minimum to zero
        :return fes: list numpy array with fes values at positions
        """
        fes = np.fromiter((self.evaluate(p)[0] for p in pos), np.float64, len(pos))
        if mintozero:
            fes -= np.min(fes)
        return fes

    def calculate_probability_density(
        self,
        kt: float,
        ranges: List[Tuple[float, float]],
        grid_points: List[int],
    ) -> Tuple[np.ndarray, np.ndarray]:
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
        axes, stepsizes = zip(
            *(
                np.linspace(*ranges[i], grid_points[i], retstep=True)
                for i in range(self.n_dim)
            )
        )
        grid = np.meshgrid(*axes)
        # reshape to have array of positions
        pos = np.array(grid).T.reshape(np.prod(grid_points), self.n_dim)
        fes = self.calculate_reference(pos)
        prob = np.exp(-fes / kt)
        # normalize with volume element from stepsizes
        prob /= np.sum(prob) * np.prod(stepsizes)
        return grid, prob
