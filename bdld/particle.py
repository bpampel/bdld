"""MD particle class"""

from typing import List, Optional, Union
import numpy as np


class Particle:
    """Simple particle class

    :param numpy.array pos: position
    :param numpy.array mom: momentum
    :param float mass: mass of particle
    """

    def __init__(
        self,
        pos: Union[float, List[float], np.ndarray],
        mom: Optional[Union[float, List[float], np.ndarray]] = None,
        mass: float = 1.0,
    ):
        """Initializes particle with given parameters

        :param pos: position of particle
        :param mom: optional momentum of particle, will be zeroed if not given
        :param mass: mass of particle, defaults to 1.0
        """
        if not isinstance(pos, (list, np.ndarray)):  # single float
            pos = [pos]
        self.pos = np.array(pos, dtype=float)
        self.mom = np.empty(0)
        self.mass = mass
        self.init_momentum(mom)

    def init_momentum(self, mom=None):
        """Initialize momentum to either given value or zero array of correct size"""
        if mom is None:
            self.mom = np.array([0.0] * len(self.pos))
        else:
            if not isinstance(mom, (list, np.ndarray)):  # single float
                mom = [mom]
            self.mom = np.array(mom, dtype=float)
        if len(self.pos) != len(self.mom):
            raise ValueError(
                "Dimensions of position and momentum do not match: %d vs %d"
                % (len(self.pos), len(self.mom))
            )
