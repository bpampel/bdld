"""Birth death algorithm"""

import copy
import numpy as np
from scipy.spatial.distance import pdist, squareform


def kernel(pos, bw):
    """Calculate (gaussian) kernels of all positions

    :param numpy.ndarray pos: positions of particles
    :param float bw: bandwidth parameter of kernel
    :return numpy.ndarray kernel: kernel value matrix
    """
    dist = pdist(pos, 'sqeuclidean')
    gauss = 1 / (2 * np.pi * bw**2)**(pos.ndim/2) * np.exp(-dist / (2*bw)**2 )
    return squareform(gauss)


class BirthDeath():
    """Birth death algorithm"""
    def __init__(self, particles, dt, bw, seed=None):
        """Set arguments

        :param particles: list of Particles shared with MD
        :param float dt: timestep of MD
        :param bw: bandwidth for gaussian kernels per direction
        :type bw: list or numpy.ndarray
        :param int seed: Seed for rng (optional)
        """
        self.particles = particles
        self.dt = dt
        self.bw = np.array(bw, dtype=float)
        self.rng = np.random.default_rng(seed)

    def step(self):
        """Perform birth-death step on particles"""
        pos = np.array([p.pos for p in self.particles])
        ene = np.array([p.energy for p in self.particles])
        bd_events = self.calculate_birth_death(pos, ene)
        for dup, kill in bd_events:
            self.particles[kill] = copy.deepcopy(self.particles[dup])
            # this copies all properties: is this desired?
            # what should be done with the momentum? Keep? Set to 0?
            # -> violates energy conservation!
            # keep the old random number for the initial thermostat step or generate new?
        print("Duplicated/Killed particles: {}".format(bd_events))

    def calculate_birth_death(self, pos, ene):
        """Calculate which particles to kill and duplicate

        The returned lists are ordered, so the first element of the kill_list
        should be replaced by a copy of the first element in the dup_list

        :param numpy.ndarray pot: positions of all particles
        :param numpy.ndarray ene: energy of all particles
        :return list of tuples (dup, kill): particle to duplicate and kill per event
        """
        num_part = len(pos)
        dup_list = []
        kill_list = []
        beta = np.average(kernel(pos, self.bw), axis=0) + ene
        beta -= np.average(beta)
        # evaluate all at same time not sequentially as in original paper
        # does it matter?
        prob = 1 - np.exp(- np.abs(beta) * self.dt)
        rand = self.rng.random(num_part)
        for i in np.where(rand <= prob)[0]:
            if i not in kill_list:
                if beta[i] > 0:
                    kill_list.append(i)
                    dup_list.append(self.random_particle(num_part, [i]))
                elif beta[i] < 0:
                    dup_list.append(i)
                    # prevent killing twice
                    kill_list.append(self.random_particle(num_part, kill_list + [i]))

        return list(zip(dup_list, kill_list))

    def random_particle(self, num_part, excl):
        """Select random particle while excluding list

        :param int num_part: total number of particles
        :param list of int excl: particles to exclude
        :return int num: random particle
        """
        return self.rng.choice([i for i in range(num_part) if i not in excl])
