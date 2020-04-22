"""Simple MD with Bussi-Parinello thermostat"""

import numpy as np
from potential import Potential
from particle import Particle

class BussiParinelloMD:
    """Perform MD with Bussi-Parinello thermostat"""
    def __init__(self, *args):
        self.pot = None  # potential
        self.p = None  # particle
        self.dt = None  # timestep
        self.rng = None
        self.rand = None  # storage for one random number
        self.forces = None
        self.energy = None
        self.c1 = None  # constants for thermostat
        self.c2 = None
        if args is not None:  # ugly shortcut for setup
            self.setup(*args)

    def step(self):
        """Perform single MD step"""
        # first part of thermostat
        self.p.mom = self.c1 * self.p.mom + self.c2 * self.rand
        # velocity verlet with force evaluation
        self.p.pos += self.p.mom / self.p.mass * self.dt + 0.5 * self.forces / self.p.mass * self.dt
        self.energy, new_forces = self.pot.evaluate(self.p.pos)
        self.p.mom += 0.5 * (self.forces + new_forces) * self.dt
        self.forces = new_forces
        # second part of thermostat
        self.rand = self.rng.standard_normal(self.pot.dimension)
        self.p.mom = self.c1 * self.p.mom + self.c2 * self.rand

    def setup(self, potential, particle, dt, friction, kT, seed):
        """Setup function"""
        self.pot = potential
        self.add_particle(particle)  # has to be set after pot
        self.energy, self.forces = self.pot.evaluate(self.p.pos)
        self.dt = dt

        # coefficients for thermostat
        self.c1 = np.exp(-0.5 * friction * kT)
        self.c2 = np.sqrt((1 - self.c1 * self.c1) * self.p.mass * kT)
        # rng
        self.rng = np.random.default_rng(seed)
        self.rand = self.rng.standard_normal(self.pot.dimension)

    def add_particle(self, p):
        """At the moment this just sets the particle
        but could be extended to add more particles.
        The particle needs to have position and momentum set."""
        if p.pos is None or p.mom is None:
            raise ValueError("Trying to add a non-initialized particle")
        if len(p.pos) != len(p.mom) != self.pot.dimension:
            raise ValueError("Dimensions of particle and potential do not match: {} vs. {}"
                             .format(p.pos, self.pot.dimension))
        self.p = p
