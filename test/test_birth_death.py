"""Test the birth-deat action"""
import os
import unittest

from typing import List, Tuple

import numpy as np

from bdld.actions import birth_death as bd
from bdld import grid  # needed for some inputs
from bdld.actions.overdamped_ld import LDParticle


# some helper functions, could also be passed into the setUp() method of the test class
def setup_bd_action(t: "BirthDeathTests") -> bd.BirthDeath:
    """Function that returns a BirthDeath action from the parameters of the class

    :param t: the BirthDeathTests instance
    """
    return bd.BirthDeath(
        t.particles,
        t.md_dt,
        t.stride,
        t.bw,
        t.kt,
        t.rate_fac,
        t.recalc_probs,
        t.approx_variant,
        t.eq_density,
        t.seed,
        t.stats_stride,
        t.stats_filename,
    )


def setup_eq_dens_and_particles(t: "BirthDeathTests") -> None:
    """Set eq_density and particles of test to some actual values

    The actual values are set here somewhat arbitrary, it's just for
    reusing the same code more often
    """
    eq_density = grid.from_npoints([(0, 1)], [11])
    eq_density.data = np.arange(1, 12) * 0.1
    eq_density.data /= np.sum(eq_density.data)

    # set up 4 particles
    particles = []
    positions = np.array([np.array([p]) for p in [0, 0.5, 0.5, 1]])
    # energies from density (only relevant for approximations)
    energies = (
        -np.log(
            eq_density.interpolate(positions).reshape(
                4,
            )
        )
        / t.kt
    )
    for pos, ene in zip(positions, energies):
        part = LDParticle(pos)
        part.energy = ene
        particles.append(part)
    t.particles = particles
    t.eq_density = eq_density


class BirthDeathTests(unittest.TestCase):
    """Test BirthDeath class and associated functions"""

    # define some default parameters for the init function to avoid repetition
    particles = []  # type: ignore
    md_dt = 1.0
    stride = 5
    bw = np.array([1.0])
    kt = 2.0
    rate_fac = 2.0
    recalc_probs = False
    eq_density = None
    approx_variant = bd.ApproxVariant.orig
    seed = None
    stats_stride = None
    stats_filename = None

    def test_init(self):
        """Test if action was set up and some values are set correctly"""
        bd_action = setup_bd_action(self)

        self.assertEqual(bd_action.dt, 5.0)
        self.assertEqual(bd_action.inv_kt, 0.5)
        np.testing.assert_equal(bd_action.bw, np.array([1.0]))
        self.assertEqual(bd_action.stats.dup_count, 0)

    def test_init_approximations(self):
        """Test some more init cases, mostly the approximations setup

        Tests for the grids of the approximations are in individual functions
        """
        self.approx_variant = bd.ApproxVariant.add
        with self.assertRaises(ValueError):
            setup_bd_action(self)  # no eq_density

        self.eq_density = grid.from_npoints([(-1, 1)], [10])
        self.eq_density.data = np.ones(10) * 0.1
        bd_action = setup_bd_action(self)

        self.approx_variant = bd.ApproxVariant.from_str("add")
        bd_action = setup_bd_action(self)

        self.approx_variant = bd.ApproxVariant.from_str("mult")
        bd_action = setup_bd_action(self)

        with self.assertRaises(TypeError): # unknown approximation
            self.approx_variant = bd.ApproxVariant.from_str("err")

    # now test the functions used to calculate the densities and approxs

    def test_kernel_1d(self):
        """Test the gaussian kernel in 1d"""
        bw = np.array([2])
        center = np.array([[0]])
        height = 1 / (np.sqrt(2 * np.pi) * 2)  # expected value
        self.assertEqual(bd.calc_kernel(center, bw), height)

        testdist = np.array([[1]])
        self.assertEqual(bd.calc_kernel(testdist, bw), height * np.exp(-1 / 8))

    def test_kernel_2d(self):
        """Test the gaussian kernel in 2d"""
        bw = np.array([1, 2])
        center = np.array([[0, 0]])
        height = 1 / (4 * np.pi)  # expected value
        self.assertEqual(bd.calc_kernel(center, bw), height)

        testdist = np.array([[1, np.sqrt(2)]])
        self.assertEqual(
            bd.calc_kernel(testdist, bw), height * np.exp(-0.5) * np.exp(-0.25)
        )

    def test_walker_density(self):
        """Test evaluation of walker density with all 3 different methods"""
        bw = np.array([0.5])
        positions = np.array([[0], [0.5], [1]])
        dens_pdist = bd.walker_density(positions, bw)  # chooses pdist variant
        dens_manual = bd._walker_density_manual(positions, bw)

        # expected: (kernel values from Wolfram alpha "N(0,0.25) at x=0 / 0.5 / x=1")
        kernel_values = [0.797885, 0.483941, 0.107982]  # 0, 0.5, 1 distance
        expected = np.empty(3)  # mean of kernel values (distance to positions)
        expected[0] = np.mean(kernel_values)
        expected[1] = np.mean([kernel_values[0]] + 2 * [kernel_values[1]])
        expected[2] = expected[0]  # reversed positions -> same density as first

        np.testing.assert_allclose(dens_pdist, expected, rtol=1e-6)
        np.testing.assert_allclose(dens_manual, expected, rtol=1e-6)

    def test_walker_density_2d(self):
        """Test evaluation of walker density in 2d"""
        bw = np.array([0.5, 2])
        positions = np.array([[0, 0], [0.5, -1], [1, -2]])
        dens_pdist = bd.walker_density(positions, bw)  # chooses pdist variant
        dens_manual = bd._walker_density_manual(positions, bw)

        # expected: (kernel values from Wolfram alpha)
        kernel_values = np.zeros((3, 2))
        # same as in 1D case above
        kernel_values[:, 0] = np.array([0.797885, 0.483941, 0.107982])
        # bw = 2, pos = [0, -1, -2]
        kernel_values[:, 1] = np.array([0.19947114020, 0.17603266338, 0.12098536226])

        expected = np.empty(3)  # mean of kernel values (distance to positions)
        expected[0] = np.mean(np.prod(kernel_values, axis=1))
        kernel_pos1 = np.array(
            [kernel_values[1], kernel_values[0], kernel_values[1]]
        )  # for particle 1
        expected[1] = np.mean(np.prod(kernel_pos1, axis=1))
        expected[2] = expected[0]  # reversed positions -> same density as first

        np.testing.assert_allclose(dens_pdist, expected, rtol=1e-6)
        np.testing.assert_allclose(dens_manual, expected, rtol=1e-6)

    def test_kernel_convolution(self):
        """Test the convolution with the kernel needed for the correction"""
        dens = grid.from_npoints([(-1, 1)], [21])
        dens.data = np.r_[np.arange(11), np.arange(9, -1, -1)]  # triangle
        bw = np.array([0.1])
        # test valid because no padding needed in manual calculation
        conv = bd.dens_kernel_convolution(dens, bw, "valid")
        # calculate expected result by hand
        kernel_points = np.arange(-0.5, 0.6, 0.1).reshape(11, 1)
        kernel = bd.calc_kernel(kernel_points, bw)
        maxvalid = dens.n_points[0] - len(kernel) + 1
        conv_manual = (
            np.array(
                [
                    np.sum([dens.data[m + i] * kernel[m] for m, _ in enumerate(kernel)])
                    for i in range(maxvalid)
                ]
            )
            * 0.1
        )
        np.testing.assert_array_almost_equal(conv.data, conv_manual)

    # test if the algorithm does what we expect

    def test_calc_beta(self):
        """Test if the birth-death probabilities are calculated correctly"""
        # setup the particles and eq_density
        setup_eq_dens_and_particles(self)
        # copy values for manual calculation
        positions = np.array([p.pos for p in self.particles])
        energies = np.array([p.energy for p in self.particles])

        # now alterate over the variants
        self.approx_variant = bd.ApproxVariant.orig
        bd_action = setup_bd_action(self)
        beta_no_corr = bd_action.calc_betas()

        self.approx_variant = bd.ApproxVariant.add
        bd_action = setup_bd_action(self)
        beta_add_corr = bd_action.calc_betas()

        self.approx_variant = bd.ApproxVariant.mult
        bd_action = setup_bd_action(self)
        beta_mult_corr = bd_action.calc_betas()

        # calculate semi-manually
        K_rho = bd.walker_density(positions, self.bw)
        beta_man = np.log(K_rho) + (energies / self.kt)
        beta_man -= np.mean(beta_man)
        np.testing.assert_array_equal(beta_no_corr, beta_man)

        # add the additive correction
        # (this is somewhat close to the actual code, but hard to do by hand)
        K_pi = bd.dens_kernel_convolution(self.eq_density, self.bw, "same")
        integral_term = bd.nd_trapz(
            (np.log(K_pi / self.eq_density) * self.eq_density).data, K_pi.stepsizes
        )
        correction = -np.log(K_pi / self.eq_density) + integral_term
        beta_add_corr_man = beta_man + correction.interpolate(positions).reshape(
            4,
        )
        np.testing.assert_array_almost_equal(beta_add_corr, beta_add_corr_man)

        # for the multiplicative correction: recombine the calculated values
        self.approx_variant = bd.ApproxVariant.mult
        beta_mult_corr_man = np.log(K_rho) - np.log(
            K_pi.interpolate(positions).reshape(
                4,
            )
        )
        beta_mult_corr_man -= np.mean(beta_mult_corr_man)
        np.testing.assert_array_almost_equal(beta_mult_corr, beta_mult_corr_man)

    def test_do_birth_death(self):
        """Test if birth_death evaluation works correctly

        This is a regtest instead of a unittest as otherwise we'd simply repeat the code

        This needs to be run after the previous one, as we use the same particles
        """
        self.approx_variant = bd.ApproxVariant.orig
        self.seed = 1111  # fix seed
        setup_eq_dens_and_particles(self)
        bd_action = setup_bd_action(self)
        bd_events = bd_action.do_birth_death()

        # this is from running it once by hand: duplicate 3 and kill 0
        # might fail if the rng implementation is different?
        bd_events_ref = [(2, 0), (3, 1)]
        self.assertEqual(bd_events, bd_events_ref)

    def test_run(self):
        """Test if everything can be run

        Individual functions have been tested before, so only "does it work"
        """
        self.stats_stride = 5
        self.seed = 1234  # fix seed as before
        setup_eq_dens_and_particles(self)
        bd_action = setup_bd_action(self)
        # import pdb; pdb.set_trace()
        bd_action.run(5)  # test also if stats are printed
        # test some stats (known from running the last test by hand)
        np.testing.assert_equal(bd_action.stats.dup_count, 1)
        np.testing.assert_equal(bd_action.stats.dup_attempts, 3)

    def test_stats(self):
        """Test the Stats class"""
        f_name = "stats"
        self.stats_stride = 5
        self.stats_filename = f_name
        stats = setup_bd_action(self).stats

        # set up, now modify one of the values and print to file
        stats.dup_count = 5
        stats.print(step=1, reset=True)
        statsfile = np.genfromtxt(f_name)
        np.testing.assert_equal(statsfile, np.array([1, 5, 0, 0, 0]))
        os.remove(f_name)

        # check if reset has worked
        self.assertEqual(stats.dup_count, 0)

        # test other branch (print to screen)
        self.stats_filename = None
        stats = setup_bd_action(self).stats
        stats.print(1)


if __name__ == "__main__":
    unittest.main()
