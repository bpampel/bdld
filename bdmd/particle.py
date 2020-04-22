"""MD particle class"""

import numpy as np


class Particle:
    """Simple particle class"""
    def __init__(self, pos, mom=None, mass=1):
        self.pos = np.array(pos)
        self.mom = None
        self.mass = mass
        self.init_momentum(mom)

    def init_momentum(self, mom=None):
        """Initialize momentum to either given value or zero array of correct size"""
        if mom is None:
            self.mom = np.array([0.0] * len(self.pos))
        else:
            self.mom = np.array(mom)
        if len(self.pos) != len(self.mom):
            raise ValueError("Dimensions of position and momentum do not match: {} vs {}"
                             .format(len(self.pos), len(self.mom)))
