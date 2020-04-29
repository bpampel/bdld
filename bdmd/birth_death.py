"""Birth death algorithm"""

import copy
import numpy as np
from scipy.spatial.distance import pdist, squareform

def kernel(pos, bw):
    """Calculate (gaussian) kernels of all positions

    :param numpy.array pos: positions of particles
    :param float bw: bandwidth parameter of kernel
    :return numpy.array kernel: kernel value matrix
    """
    dist = pdist(pos, 'sqeuclidean')
    gauss = 1 / (2 * np.pi * bw**2)**(pos.ndim/2) * np.exp(-dist / (2*bw)**2 )
    return squareform(gauss)


class BirthDeath():
    """Birth death algorithm"""
    def __init__(self, particles, dt, bw, seed=None):
        """Set arguments

        :param particles: list with Particles shared with MD
        :param float dt: timestep of MD
        :param float bw: bandwidth for gaussian kernels
        :param int seed: Seed for rng (optional)
        """
        self.particles = particles
        self.dt = dt
        self.bw = bw
        self.rng = np.random.default_rng(seed)

    def step(self):
        """Perform birth-death step on particles"""
        num_part = self.particles
        pos = [self.particles[p].self for p in range(num_part)]
        ene = [self.particles[p].ene for p in range(num_part)]
        bd_events = self.calculate_birth_death(pos, ene, self.bw, self.dt)
        for dup, kill in bd_events:
            self.particles[kill] = copy.deepcopy(self.particles[dup])
            # this copies all properties: is this desired?
            # what should be done with the momentum? Keep? Set to 0?
            # -> violates energy conservation!
            # keep the old random number for the initial thermostat step or generate new?

    def calculate_birth_death(self, pos, ene, bw, dt):
        """Calculate which particles to kill and duplicate

        The returned lists are ordered, so the first element of the kill_list
        should be replaced by a copy of the first element in the dup_list

        :param pot: list with positions of all particles
        :param ene: list with energy of all particles
        :param float bw: bandwidth for gaussian kernel
        :param float dt: stepsize of MD
        :return list of tuples (dup, kill): particle to duplicate and kill per event
        """
        num_part = len(pos)
        dup_list = []
        kill_list = []
        beta = np.average(kernel(pos, bw), axis=0) + ene
        beta -= np.average(beta)
        # evaluate all at same time not sequentially as in original paper
        # does it matter?
        prob = 1 - np.exp(- np.abs(beta) * dt)
        rand = self.rng.random(num_part)
        for i in np.where(rand > prob):
            if i not in kill_list:
                if beta[i] > 0:
                    kill_list.append(i)
                    dup_list.append(self.random_particle(num_part, [i]))
                elif beta[i] < 0:
                    dup_list.append(i)
                    # prevent killing twice
                    kill_list.append(self.random_particle(num_part, kill_list + i))

        return list(zip(dup_list, kill_list))

    def random_particle(self, num_part, excl):
        """Select random particle while excluding list

        :param int num_part: total number of particles
        :param list of int excl: particles to exclude
        :return int num: random particle
        """
        return self.rng.choice([i for i in range(num_part) if i not in excl])
