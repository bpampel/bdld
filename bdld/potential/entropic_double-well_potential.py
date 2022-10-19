"""Entropic double-well potential

Described also in Eq. 30 of Faradjian & Elber J. Chem. Phys. 120, 10880 (2004)
https://doi.org/10.1063/1.1738640
"""

from typing import List, Optional, Union
import numpy as np

from bdld.potential.potential import Potential

# coefficients
A = np.array([-200.0, -100.0, -175.0, 15.0])
a = np.array([-1.0, -1.0, -6.5, 0.7])
b = np.array([0.0, 0.0, 11.0, 0.6])
c = np.array([-10.0, -10.0, -6.5, 0.7])
x0 = np.array([1.0, 0.0, -0.5, -1.0])
y0 = np.array([0.0, 0.5, 1.5, 1.0])
pot_shift = +30.33319242243656


class EntropicDoubleWellPotential(Potential):
    """Entropic double-well potential

    This is a 2D potential with two states separated by a entropic
    bottleneck at (0, 0) given by the equation

    f(x,y) = a * (x**6 + y**6 + exp(-(x/sigma_x)**2) * (1-exp(-(y/sigma_y)**2)) )

    where sigma_x, sigma_y are width parameters of the entropic barrier:
    - sigma_x defines the width of the barrier in x direction, sigma
    - sigma_y defines the width of the opening of the barrier in y direction

    Additionally the whole potential is scaled by the factor a

    Described also in Eq. 30 of Faradjian & Elber J. Chem. Phys. 120, 10880 (2004)
    https://doi.org/10.1063/1.1738640

    There sigma_x = sigma_y = 0.1 and a = 1 which is used as default here

    :param sigma_x: barrier width parameter, default 0.1
    :param sigma_y: barrier opening parameter, default 0.1
    :param a: scale potential by float, default 1.0
    """

    def __init__(
        self,
        sigma_x: Optional[float] = None,
        sigma_y: Optional[float] = None,
        scaling_factor: Optional[float] = None,
    ) -> None:
        """Initialize Müller-Brown potential

        :param scaling_factor: Scale potential by float, optional
        """
        super().__init__()  # not actually needed but enforces having the values

        self.n_dim = 2
        self.ranges = [(-1, 1), (-1, 1)]
        self.sigma_x = sigma_x or 0.1
        self.sigma_y = sigma_y or 0.1
        self.a = scaling_factor or 1.0

    def __str__(self) -> str:
        """Give out coefficients"""
        text = (
            "Entropic double-well potential\n"
            + "  Parameters:\n"
            + f"    sigma_x: {self.sigma_x}\n"
            + f"    sigma_x: {self.sigma_x}\n"
            + f"    a: {self.a}"
        )
        return text

    def energy(self, pos: Union[List[float], np.ndarray]) -> float:
        """Get energy at position

        :param pos: position to be evaluated (given as list or array even in 1d)
        :return: energy
        """
        x = pos[0]
        y = pos[1]
        pot = self.a * (
            x**6
            + y**6
            + np.exp(-((x / self.sigma_x) ** 2))
            * (1 - np.exp(-((y / self.sigma_y) ** 2)))
        )
        return pot

    def force(self, pos: Union[List[float], np.ndarray]) -> np.ndarray:
        """Get force at position

        :param pos: position to be evaluated (given as list or array even in 1d)
        :return force: array with force per direction
        """
        force = np.array([0.0, 0.0])
        x = pos[0]
        y = pos[1]
        force[0] = self.a * (
            6 * x**5
            - 2
            * x
            * np.exp(-((x / self.sigma_x) ** 2))
            * (1 - np.exp(-y / self.sigma_y) ** 2)
            / self.sigma_x**2
        )
        force[1] = self.a * (
            6 * y**5
            - 2
            * y
            * np.exp(-((x / self.sigma_x) ** 2) - (y / self.sigma_y) ** 2)
            / self.sigma_y**2
        )
        return force
