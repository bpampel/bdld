"""Müller-Brown Potential class to be evaluated with md"""

from typing import List, Union, Tuple
import numpy as np

from bdld.potential.potential import Potential

# coefficients
A = np.array(  [ -200.0 , -100.0 , -175.0 ,  15.0 ] )
a = np.array(  [   -1.0 ,   -1.0 ,   -6.5 ,   0.7 ] )
b = np.array(  [    0.0 ,    0.0 ,   11.0 ,   0.6 ] )
c = np.array(  [  -10.0 ,  -10.0 ,   -6.5 ,   0.7 ] )
x0 = np.array( [    1.0 ,    0.0 ,   -0.5 ,  -1.0 ] )
y0 = np.array( [    0.0 ,    0.5 ,    1.5 ,   1.0 ] )
pot_shift = +30.33319242243656


class MuellerBrownPotential(Potential):
    """Müller-Brown potential

    This is a 2D potential with three metastable states

    It consists of a sum of four exponential terms:
    -200*exp( -(x-1)^2 -10*y^2 )
    -100*exp( -x^2 -10*(y-0.5)^2 )
    -170*exp( -6.5*(x+0.5)^2 + 11*(x+0.5)*(y-1.5) -6.5*(y-1.5)^2 )
    +15*exp(  0.7*(x+1)^2 +0.6*(x+1)*(y-1) +0.7*(y-1)^2 )

    The potential and its coefficients are described in
    K. Müller and L. D. Brown, Theoretical Chemistry Accounts, 53, 1979 pp. 75–93.

    :param scaling_factor: scale potential by factor
    """

    def __init__(
        self,
        scaling_factor: float,
    ) -> None:
        super().__init__()  # not actually needed but enforces having the values

        self.n_dim = 2
        self.ranges = [(-1.5,1.5), (-0.5,2.5)]
        self.scaling_factor = scaling_factor
        global A
        A *= self.scaling_factor

    def __str__(self) -> str:
        """Give out coefficients"""
        return f"Müller-Brown potential with scaling factor {self.scaling_factor}"

    def evaluate(self, pos: Union[List[float], np.ndarray]) -> Tuple[float, np.ndarray]:
        """Get potential energy and force at position

        Faster than base method because exponentials need to be evaluated only once

        :param pos: position to be evaluated
        :return: (energy, force)
        """
        # from Omar's bdls-code
        pot = 0.0
        force = np.array([ 0.0 , 0.0 ])
        x = pos[0]
        y = pos[1]
        for i in range(4):
            exp_tmp1 = np.exp( a[i]*(x-x0[i])**2 + b[i]*(x-x0[i])*(y-y0[i]) + c[i]*(y-y0[i])**2 )
            pot += A[i] * exp_tmp1
            force[0] += -A[i] * ( 2.0*a[i]*(x-x0[i])+ b[i]*(y-y0[i]) ) * exp_tmp1
            force[1] += -A[i] * ( b[i]*(x-x0[i])+ 2.0*c[i]*(y-y0[i]) ) * exp_tmp1
        pot += pot_shift
        return (pot,force)

    def energy(self, pos: Union[List[float], np.ndarray]) -> float:
        """Get energy at position

        :param pos: position to be evaluated (given as list or array even in 1d)
        :return: energy
        """
        pot = 0.0
        x = pos[0]
        y = pos[1]
        for i in range(4):
            pot += A[i] * np.exp( a[i]*(x-x0[i])**2 + b[i]*(x-x0[i])*(y-y0[i]) + c[i]*(y-y0[i])**2 )
        pot += pot_shift
        return pot

    def force(self, pos: Union[List[float], np.ndarray]) -> np.ndarray:
        """Get force at position

        :param pos: position to be evaluated (given as list or array even in 1d)
        :return force: array with force per direction
        """
        force = np.array([ 0.0 , 0.0 ])
        x = pos[0]
        y = pos[1]
        for i in range(4):
            exp_tmp1 = np.exp( a[i]*(x-x0[i])**2 + b[i]*(x-x0[i])*(y-y0[i]) + c[i]*(y-y0[i])**2 )
            force[0] += -A[i] * ( 2.0*a[i]*(x-x0[i])+ b[i]*(y-y0[i]) ) * exp_tmp1
            force[1] += -A[i] * ( b[i]*(x-x0[i])+ 2.0*c[i]*(y-y0[i]) ) * exp_tmp1
        return force
