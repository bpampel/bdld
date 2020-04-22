'''Potential class to be evaluated with md'''

import numpy as np
poly = np.polynomial.polynomial

class Potential():

    def __init__(self, coeffs=None):
        '''
        :param coeffs: Either list (1D) or numpy array with coefficients (2D,3D).
                       The coefficient i,j,k has to be given in coeffs[i,j,k]
        '''
        self.dimension = None
        self.coeffs = coeffs
        self.set_dimensions()
        # note: the derivative matrices are larger than needed. Implement trim_zeros for multiple dimensions?
        self.der = [poly.polyder(self.coeffs, axis=d) for d in range(self.dimension)]


    def evaluate(self, pos):
        '''Get potential energy / forces at position
        :param pos: current position as list or numpy array
        :return: (energy, forces)
            energy: float
            forces: list with forces per direction
        '''
        pos = np.append(pos, [0.0]*(3-self.dimension)) #  needed to have 3 elements in pos
        energy = poly.polyval3d(*pos, self.coeffs)
        forces = np.array([-poly.polyval3d(*pos, self.der[d]) for d in range(self.dimension)])
        return (energy, forces)

    def set_dimensions(self,dim=None):
        '''Set dimension of Potential either from specified integer
        or from dimensions of coefficients'''
        if dim is None:
            self.dimension = self.coeffs.ndim if self.coeffs is not None else None
        else:
            self.dimension = dim
