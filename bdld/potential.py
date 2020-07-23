"""Potential class to be evaluated with md"""

import numpy as np
poly = np.polynomial.polynomial


class Potential():
    """Simple class holding a polynomial potential

    :param numpy.array coeffs: Coefficients of polynomial potential
    :param list of numpy.array der: Coefficients of derivative of potential per direction
    :param int dimension: Dimensions of potential
    """

    def __init__(self, coeffs):
        """Set up from given coefficients

        :param coeffs: The coefficient i,j,k has to be given in coeffs[i,j,k]
        :type coeffs: list (1D) or numpy.array with coefficients (2D,3D).
        """
        self.coeffs = np.array(coeffs)
        self.dimension = self.coeffs.ndim
        # note: the derivative matrices are larger than needed. Implement trim_zeros for multiple dimensions?
        self.der = [poly.polyder(self.coeffs, axis=d) for d in range(self.dimension)]

    def __str__(self):
        """Give out coefficients"""
        return 'polynomial with coefficients ' + list(self.coeffs).__str__()

    def evaluate(self, pos):
        """Get potential energy and forces at position

        :param pos: position to be evaluated
        :type pos: list or numpy.array
        :return: (energy, forces)
        :rtype: Tuple(float, list of float)
        """
        pos = np.append(pos, [0.0]*(3-self.dimension)) #  needed to have 3 elements in pos
        energy = poly.polyval3d(*pos, self.coeffs)
        forces = np.array([-poly.polyval3d(*pos, self.der[d]) for d in range(self.dimension)])
        return (energy, forces)
