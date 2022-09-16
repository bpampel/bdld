"""Potential based on polynomial functions to be evaluated with md"""

from typing import Callable, List, Union, Tuple
import numpy as np

from bdld.potential.potential import Potential

poly = np.polynomial.polynomial


class PolynomialPotential(Potential):
    """Simple polynomial potential class defined by coefficients

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
        :raises NotImplementedError: if more than 2 dimensions are passed
        """
        super().__init__()  # not actually needed but enforces having the values

        self.coeffs: np.ndarray = np.array(coeffs)
        if self.coeffs.ndim > 3:
            raise NotImplementedError(
                "Class can't be used for potentials in more than 3 dimensions"
            )
        self.n_dim: int = self.coeffs.ndim
        # note: the derivative matrices are larger than needed. Implement trim_zeros for multiple dimensions?
        self.der: List[np.ndarray] = [
            poly.polyder(self.coeffs, axis=d) for d in range(self.n_dim)
        ]
        self.polyval = self.set_polyval()
        # the ranges are at the moment not actually checked when evaluating but needed for the birth/death
        if ranges is None:
            ranges = []  # mutable default arguments are bad
        self.ranges: List[Tuple[float, float]] = ranges

    def __str__(self) -> str:
        """Give out coefficients"""
        string = "polynomial with coefficients"
        if self.coeffs.ndim == 1:
            string += f" {self.coeffs}"
        else:  # print array in new line
            string += f"\n{np.array(self.coeffs)}\n"
        return string

    def set_polyval(self) -> Callable:
        """Selects polyval function from numpy.polynomial.polynomial depending on self.n_dim"""
        if self.n_dim == 1:
            return poly.polyval
        elif self.n_dim == 2:
            return poly.polyval2d
        elif self.n_dim == 3:
            return poly.polyval3d
        else:
            raise ValueError("No polyval function for more than 3 dimensions")

    def energy(self, pos: Union[List[float], np.ndarray]) -> float:
        """Get energy at position

        :param pos: position to be evaluated (given as list or array even in 1d)
        :return energy: energy at position
        """
        return self.polyval(*pos, self.coeffs)

    def force(self, pos: Union[List[float], np.ndarray]) -> np.ndarray:
        """Get force at position

        :param pos: position to be evaluated (given as list or array even in 1d)
        :return force: array with force per direction
        """
        return np.array([-self.polyval(*pos, self.der[d]) for d in range(self.n_dim)])


def coefficients_from_file(filename: str, n_dim: int) -> np.ndarray:
    """Read coefficients from file

    This should be compatible to the ves_md_linearexpansion files

    The syntax for the files is rows with one coefficient each.
    The first n_dim columns contain the polynomial orders per dimension,
    the next one the coefficient value. Remaining columns are ignored.

    :param filename: path to file holding the coefficients
    :param n_dim: dimensions of potential
    """
    file_data = np.genfromtxt(filename)

    # highest polynomial order per dimension
    shape = [int(max(file_data[:, i])) + 1 for i in range(n_dim)]

    coeffs = np.zeros(shape)
    for row in file_data:  # each row is one coefficient
        indices = row[:n_dim].astype(int)
        coeffs[tuple(indices)] = row[n_dim]  # without tuple gives elements of 1st level

    return coeffs
