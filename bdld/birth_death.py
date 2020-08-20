"""Birth death algorithm"""

import copy
from typing import List, Optional, Union, Tuple

import numpy as np
from scipy.spatial.distance import pdist, sqeuclidean, squareform

from bdld.bussi_parinello_ld import BpldParticle


def walker_density(pos: np.ndarray, bw: np.ndarray) -> np.ndarray:
    """Calculate the local density at each walker (average kernel value)

    This is done by a Kernel density estimate with gaussian kernels.
    The current implementation only works in 1d, because the distance is calculated
    as euclidean for all directions without weighting but would need to take the
    individual bandwidths into account.

    For less than 10000 walkers a spare pdist matrix is calculated and averaged.
    Because the matrix size scales exponentially with the number of walkers for
    more than 10,000 a walker-wise calculation is done that requires less memory
    but calculates each distance twice.

    :param numpy.ndarray pos: positions of particles
    :param float bw: bandwidth parameter of kernel
    :return numpy.ndarray kernel: kernel value matrix
    """
    if len(pos) <= 10000:  # pdist matrix with maximum 10e8 float64 values
        dist = pdist(pos, "sqeuclidean")
        gauss = (
            1 / (2 * np.pi * bw ** 2) ** (pos.ndim / 2) * np.exp(-dist / (2 * bw) ** 2)
        )
        return np.mean(squareform(gauss), axis=0)
    else:
        density = np.empty((len(pos)))
        for i in range(len(pos)):
            dist = np.fromiter(
                (sqeuclidean(pos[i], pos[j]) for j in range(len(pos)) if j != i),
                np.float64,
                len(pos) - 1,
            )
            gauss_dist = (
                1
                / (2 * np.pi * bw ** 2) ** (pos.ndim / 2)
                * np.exp(-dist / (2 * bw) ** 2)
            )
            density[i] = np.mean(gauss_dist)
        return density


class BirthDeath:
    """Birth death algorithm"""

    def __init__(
        self,
        particles: List[BpldParticle],
        dt: float,
        bw: Union[List[float], np.ndarray],
        kt: float,
        seed: Optional[int] = None,
        logging: bool = False,
    ) -> None:
        """Set arguments

        :param particles: list of Particles shared with MD
        :param float dt: timestep of MD
        :param bw: bandwidth for gaussian kernels per direction
        :param float kt: thermal energy of system
        :param int seed: Seed for rng (optional)
        """
        self.particles: List[BpldParticle] = particles
        self.dt: float = dt
        self.bw: np.ndarray = np.array(bw, dtype=float)
        self.inv_kt: float = 1 / kt
        self.rng: np.random.Generator = np.random.default_rng(seed)
        self.logging: bool = logging
        print(
            f"Setting up birth/death scheme\n"
            f"Parameters:\n"
            f"  dt = {self.dt}\n"
            f"  bw = {self.bw}\n"
            f"  kt = {kt}"
        )
        if seed:
            print(f"  seed = {seed}")
        print()
        if self.logging:
            self.dup_count = 0
            self.dup_attempts = 0
            self.kill_count = 0
            self.kill_attempts = 0

    def step(self) -> List[Tuple[int, int]]:
        """Perform birth-death step on particles

        Returns list of succesful birth/death events"""
        pos = np.array([p.pos for p in self.particles])
        ene = np.array([p.energy for p in self.particles])
        bd_events = self.calculate_birth_death(pos, ene)
        for dup, kill in bd_events:
            self.particles[kill] = copy.deepcopy(self.particles[dup])
            # this copies all properties: is this desired?
            # what should be done with the momentum? Keep? Set to 0?
            # -> violates energy conservation!
            # keep the old random number for the initial thermostat step or generate new?
        return bd_events

    def calculate_birth_death(
        self, pos: np.ndarray, ene: np.ndarray
    ) -> List[Tuple[int, int]]:
        """Calculate which particles to kill and duplicate

        The returned tuples are ordered, so the first particle in the tuple
        should be replaced by a copy of the second one

        :param pos: positions of all particles
        :param ene: energy of all particles
        :return bd_events: list of tuples with particles to duplicate and kill per event
        """
        num_part = len(pos)
        dup_list = []
        kill_list = []
        with np.errstate(divide="ignore"):
            # density can be zero and make beta -inf. Filter when averaging in next step
            beta = np.log(walker_density(pos, self.bw)) + ene * self.inv_kt
        beta -= np.mean(beta[beta != -np.inf])
        if self.logging:  # get number of attempts from betas
            curr_kill_attempts = np.count_nonzero(beta > 0)
            self.kill_attempts += curr_kill_attempts
            self.dup_attempts += num_part - curr_kill_attempts

        # evaluate all at same time not sequentially as in original paper
        # does it matter?
        prob = 1 - np.exp(-np.abs(beta) * self.dt)
        rand = self.rng.random(num_part)
        for i in np.where(rand <= prob)[0]:
            if i not in kill_list:
                if beta[i] > 0:
                    kill_list.append(i)
                    dup_list.append(self.random_particle(num_part, [i]))
                    if self.logging:
                        self.kill_count += 1
                elif beta[i] < 0:
                    dup_list.append(i)
                    # prevent killing twice
                    kill_list.append(self.random_particle(num_part, [i]))
                    if self.logging:
                        self.dup_count += 1

        return list(zip(dup_list, kill_list))

    def random_particle(self, num_part: int, excl: List[int]) -> int:
        """Select random particle while excluding list

        :param num_part: total number of particles
        :param excl: particles to exclude
        :return num: random particle
        """
        return self.rng.choice([i for i in range(num_part) if i not in excl])

    def prob_density_grid(self, grid: np.ndarray, energy: np.ndarray) -> np.ndarray:
        """Calculate the density of walkers (kernel density) on a grid

        :param grid: positions to calculate the kernel values
        :param grid: energies of the grid values
        :return array: grid rho beta
        """
        rho = []
        beta = []
        walker_pos = [p.pos for p in self.particles]
        walker_ene = [p.energy for p in self.particles]
        for g, e in zip(grid, energy):
            pos = np.append(g, walker_pos).reshape(
                (len(walker_pos) + 1, walker_pos[1].ndim)
            )  # array with positions as subarrays
            ene = np.append(e, walker_ene)
            # full kernel is needed for probability (normalization)
            rho_g = walker_density(pos, self.bw)
            rho.append(rho_g[0])

            beta_g = np.log(rho_g) + ene * self.inv_kt
            beta.append(beta_g[0] - np.mean(beta_g[beta_g != -np.inf]))

        return np.c_[grid, rho, beta]

    def print_stats(self) -> None:
        """Print birth/death probabilities to screen"""
        if self.logging:
            kill_perc = 100 * self.kill_count / self.kill_attempts
            dup_perc = 100 * self.dup_count / self.dup_attempts
            ratio_succ = self.kill_count / self.dup_count
            ratio_attempts = self.kill_attempts / self.dup_attempts
            print(
                f"Succesful birth events: {self.dup_count}/{self.dup_attempts} ({dup_perc:.4}%)"
            )
            print(
                f"Succesful death events: {self.kill_count}/{self.kill_attempts} ({kill_perc:.4}%)"
            )
            print(
                f"Ratio birth/death: {ratio_succ:.4} (succesful)  {ratio_attempts:.4} (attemps)"
            )
        else:
            raise ValueError("Can't print statistics: Logging is turned off")
