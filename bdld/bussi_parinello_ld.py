"""Simple Langevin Dynamics with Bussi-Parinello thermostat"""

from typing import List, Optional, Union
import numpy as np

from bdld.action import Action
from bdld.particle import Particle
from bdld.potential import Potential


class BpldParticle(Particle):
    """Derived Particle class that additionally stores MD related variables

    :param energy: stores last energy evaluation
    :param forces: stores last force evaluation per dimension
    :param float c2: constant for the MD thermostat (mass dependent)
    """

    def __init__(self, *args) -> None:
        """Creates particle from base class with additional attributes"""
        super().__init__(*args)
        self.energy: float = 0.0
        self.forces: np.ndarray = np.empty(0)
        self.c2: float = 0.0  # second thermostat constant depends on mass


class BussiParinelloLD(Action):
    """Perform Langevin Dynamics with Bussi-Parinello thermostat

    Can handle multiple non-interacting particles (= walkers) simultaneously

    :param pot: potential to perform LD on
    :param particles: list with particles (=walkers) of simulation
    :param dt: timestep
    :param friction: friction parameter of langevin equation
    :param kt: thermal energy in units of kt
    :param rng: numpy.random.Generator instance for the thermostat
    :param c1: constant for thermostat
    """

    def __init__(
        self,
        pot: Potential,
        dt: float,
        friction: float,
        kt: float,
        seed: Optional[int] = None,
    ) -> None:
        """Creates Langevin dynamics instance

        :param pot: potential to use
        :param dt: timestep between particle moves
        :param friction: friction parameter of langevin dynamics thermostat
        :param kt: thermal energy in units of kt
        :param seed: seed for rng, optional
        """
        self.pot: Potential = pot
        self.particles: List[BpldParticle] = []  # list of particles
        self.dt: float = dt
        self.kt: float = kt
        self.friction: float = friction
        self.c1: float = np.exp(-0.5 * friction * dt)
        self.rng: np.random.Generator = np.random.default_rng(seed)
        print(
            f"Setting up Langevin dynamics with Bussi-Parinello thermostat\n"
            f"Parameters:\n"
            f"  potential = {self.pot}\n"
            f"  timestep = {self.dt}\n"
            f"  friction = {self.friction}\n"
            f"  kt = {self.kt}"
        )
        if seed:
            print(f"  seed = {seed}")
        print()

    def run(self, step: int = None) -> None:
        """Perform single MD step on all particles"""
        for p in self.particles:
            # first part of thermostat
            p.mom = self.c1 * p.mom + p.c2 * self.rng.standard_normal(self.pot.n_dim)
            # velocity verlet with force evaluation
            p.mom += 0.5 * p.forces * self.dt
            p.pos += (p.mom / p.mass) * self.dt
            p.energy, p.forces = self.pot.evaluate(p.pos)
            p.mom += 0.5 * p.forces * self.dt
            # second part of thermostat
            p.mom = self.c1 * p.mom + p.c2 * self.rng.standard_normal(self.pot.n_dim)

    def add_particle(
        self, pos: Union[List, np.ndarray], partnum: int = -1, overwrite: bool = False
    ) -> None:
        """Add particle to system

        :param pos: list or numpy array with initial position of the particle per dimension
        :param partnum: specifies particle number (position in list). Default is -1 (at end)
        :param overwrite: overwrite existing particle instead of inserting (default False)
        """
        if len(pos) != self.pot.n_dim:
            raise ValueError(
                "Dimensions of particle and potential do not match: {} vs. {}".format(
                    pos, self.pot.n_dim
                )
            )
        p = BpldParticle(pos)
        p.energy, p.forces = self.pot.evaluate(p.pos)
        p.c2 = np.sqrt((1 - self.c1 * self.c1) * p.mass * self.kt)
        if overwrite:
            self.particles[partnum] = p
        else:
            if partnum == -1:
                self.particles.append(p)
            else:
                self.particles.insert(partnum, p)

    def remove_particle(self, partnum: int) -> None:
        """Removes particle from MD

        :param partnum: specifies particle number in list
        """
        del self.particles[partnum]
