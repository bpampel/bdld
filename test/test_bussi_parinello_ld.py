import unittest
import numpy as np

from bdld.actions import bussi_parinello_ld as bp_ld
from bdld.potential import polynomial


class BpldTests(unittest.TestCase):
    """Test dynamics of BussiParinelloLD"""

    def test_run(self):
        """Setup dynamics on simple potential with just 1 particle and run for one step"""
        # first define the parameters
        coeffs = [0,-1] # linear slope moving particle to the right
        dt = 0.1
        friction = 1
        kt = 1
        seed = 1234

        pot = polynomial.PolynomialPotential(coeffs)
        ld = bp_ld.BussiParinelloLD(pot, dt, friction, kt, seed)
        ld.add_particle([0])  # add single particle at origin
        ld.run()

        # now try to calculate dynamics by hand
        rng = np.random.default_rng(seed)  # get same random numbers as ld
        rand_values = rng.standard_normal(2)  # [-1.60383681, 0.06409991]

        # test constants needed for dynamics, calculated by hand
        c1 = 0.951229424500714  # np.exp(-0.5 * dt * friction)
        c2 = 0.308484330175846
        self.assertEqual(c1, ld.c1)
        self.assertEqual(c2, ld.particles[0].c2)

        noise_values = c2 * rand_values
        pos = noise_values[0]*0.1 + 1*0.5*0.1**2  # momentum + force
        mom = c1*1*0.1 + c1*noise_values[0] + noise_values[1]

        self.assertAlmostEqual(pos, ld.particles[0].pos[0])
        self.assertAlmostEqual(mom, ld.particles[0].mom[0])
