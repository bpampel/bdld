"""Potential class to be evaluated with md"""

from typing import Any, List, Union, Tuple
import numpy as np

poly = np.polynomial.polynomial


class Potential:
    """Simple class holding a polynomial potential

    :param numpy.array coeffs: Coefficients of polynomial potential
    :param list of numpy.array der: Coefficients of derivative of potential per direction
    :param int dimension: Dimensions of potential
    """

    def __init__(self, coeffs: Union[List[float], np.ndarray]) -> None:
        """Set up from given coefficients

        :param coeffs: The coefficient i,j,k has to be given in coeffs[i,j,k]
        :type coeffs: list (1D) or numpy.array with coefficients (2D,3D).
        """
        self.coeffs = np.array(coeffs)
        self.n_dim = self.coeffs.ndim
        # note: the derivative matrices are larger than needed. Implement trim_zeros for multiple dimensions?
        self.der = [poly.polyder(self.coeffs, axis=d) for d in range(self.n_dim)]

    def __str__(self) -> str:
        """Give out coefficients"""
        return "polynomial with coefficients " + list(self.coeffs).__str__()

    def evaluate(
        self, pos: Union[float, List[float], np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Get potential energy and forces at position

        :param pos: position to be evaluated
        :return: (energy, forces)
        """
        pos = np.append(
            pos, [0.0] * (3 - self.n_dim)
        )  #  needed to have 3 elements in pos
        energy = poly.polyval3d(*pos, self.coeffs)
        forces = np.array(
            [-poly.polyval3d(*pos, self.der[d]) for d in range(self.n_dim)]
        )
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
