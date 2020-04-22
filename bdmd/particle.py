"""MD particle class"""

import numpy as np

class Particle:
    """MD particle class"""
    def __init__(self, mass=1, **kwargs):
        self.mass = mass
        self.pos = None
        self.mom = None

        pos = kwargs.get('pos', None)
        mom = kwargs.get('mom', None)
        if pos is not None:
            self.setup(pos, mom)

    def setup(self, pos, mom=None):
        """Set position and momentum of particle
        If momentum is not given it is initialized to zero.
        """
        if mom is None:
            self.mom = np.array([0.0] * len(pos))
        else:
            self.mom = np.array(mom)
        if len(pos) != len(self.mom):
            raise ValueError("Dimensions of position and momentum do not match: {} vs {}"
                             .format(len(pos), len(mom)))
        self.pos = np.array(pos)
