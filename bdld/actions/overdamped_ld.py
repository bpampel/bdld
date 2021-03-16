"""Simple overdamped Langevin Dynamics taken from the Lu, Lu, Nolen paper"""

from enum import Enum
from typing import List, Optional, Union
import numpy as np

from bdld.actions.action import Action
from bdld.particle import Particle
from bdld.potential import potential


class LDParticle(Particle):
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


class OverdampedLD(Action):
    """Perform overdamped Langevin Dynamics

    The implementation is according to the algorithm in the Lu, Lu, Nolen paper

    Can handle multiple non-interacting particles (= walkers) simultaneously

    :param pot: potential to perform LD on
    :param dt: timestep
    :param kt: thermal energy. Set to 1 to have no need to modify birth/death
    :param noise_factor: prefactor of the noise term
    :param rng: numpy.random.Generator instance for the thermostat
    """

    def __init__(
        self,
        pot: potential.Potential,
        dt: float,
        seed: Optional[int] = None,
    ) -> None:
        """Creates overdamped Langevin dynamics instance

        :param pot: potential to use
        :param dt: timestep between particle moves
        :param seed: seed for rng, optional
        """
        self.pot: potential.Potential = pot
        self.particles: List[LDParticle] = []  # list of particles
        self.dt: float = dt
        self.kt: float = 1.0
        self.noise_factor: float = np.sqrt(2 * self.dt)  # calculate only once
        self.rng: np.random.Generator = np.random.default_rng(seed)
        print(
            f"Setting up overdamped Langevin dynamics\n"
            f"Parameters:\n"
            f"  potential = {self.pot}\n"
            f"  timestep = {self.dt}"
        )
        if seed:
            print(f"  seed = {seed}")
        print()

    def run(self, step: int = None) -> None:
        """Perform single MD step on all particles

        :param step: Not used, just to have the right function signature"""
        for p in self.particles:
            p.pos += self.dt * p.forces + self.noise_factor * self.rng.standard_normal(
                self.pot.n_dim
            )
            self.pot.apply_boundary_condition(p.pos, p.mom)
            p.energy, p.forces = self.pot.evaluate(p.pos)

    def add_particle(
        self, pos: Union[List, np.ndarray], mass=1.0, partnum: int = -1, overwrite: bool = False
    ) -> None:
        """Add particle to system

        :param pos: list or numpy array with initial position of the particle per dimension
        :param mass: mass of particles (ignored in algorithm)
        :param partnum: specifies particle number (position in list). Default is -1 (at end)
        :param overwrite: overwrite existing particle instead of inserting (default False)
        """
        if len(pos) != self.pot.n_dim:
            raise ValueError(
                "Dimensions of particle and potential do not match: {} vs. {}".format(
                    pos, self.pot.n_dim
                )
            )
        p = LDParticle(pos, None, mass)
        p.energy, p.forces = self.pot.evaluate(p.pos)
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
