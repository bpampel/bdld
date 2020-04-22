"""Simple MD with Bussi-Parinello thermostat"""

import numpy as np
from particle import Particle


class BpmdParticle(Particle):
    """Particle class that additionally stores MD related variables"""
    def __init__(self, *args):
        super(BpmdParticle, self).__init__(*args)
        self.rand = None  # storage for one random number per dimension
        self.forces = None
        self.energy = None
        self.c2 = None  # second thermostat constant depends on mass


class BussiParinelloMD():
    """Perform MD with Bussi-Parinello thermostat
    Can handle multiple non-interacting particles (= walkers) simultaneously
    """
    def __init__(self, *args):
        self.pot = None  # potential
        self.particles = []  # list of particles
        self.dt = None  # timestep
        self.rng = None
        self.kt = None
        self.c1 = None  # constant for thermostat
        if args is not None:  # ugly shortcut for setup
            self.setup(*args)

    def step(self):
        """Perform single MD step on all particles"""
        for p in self.particles:
            # first part of thermostat
            p.mom = self.c1 * p.mom + p.c2 * p.rand
            # velocity verlet with force evaluation
            p.pos += p.mom / p.mass * self.dt + 0.5 * p.forces / p.mass * self.dt
            p.energy, new_forces = self.pot.evaluate(p.pos)
            p.mom += 0.5 * (p.forces + new_forces) * self.dt
            p.forces = new_forces
            # second part of thermostat
            p.rand = self.rng.standard_normal(self.pot.dimension)
            p.mom = self.c1 * p.mom + p.c2 * p.rand

    def setup(self, potential, dt, friction, kt, seed=None):
        """Setup function"""
        self.pot = potential
        self.dt = dt
        self.kt = kt
        # friction is only needed here and not stored
        self.c1 = np.exp(-0.5 * friction * kt)
        self.rng = np.random.default_rng(seed)

    def add_particle(self, pos, partnum=-1):
        """Add particle to MD
        :param pos: float or array of floats that give the initial position of the particle
        :param partnum: int to specify particle number in list. Default is -1 (add at end)
        """
        if len(pos) != self.pot.dimension:
            raise ValueError("Dimensions of particle and potential do not match: {} vs. {}"
                             .format(pos, self.pot.dimension))
        p = BpmdParticle(pos)
        p.energy, p.forces = self.pot.evaluate(p.pos)
        p.c2 = np.sqrt((1 - self.c1 * self.c1) * p.mass * self.kt)
        p.rand = self.rng.standard_normal(self.pot.dimension) # initialize for first step
        self.particles.append(p)

