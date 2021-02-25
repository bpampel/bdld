import unittest
import numpy as np

from bdld.actions import birth_death as bd
from bdld import grid  # needed for some inputs


class BirthDeathTests(unittest.TestCase):
    """Test BirthDeath class and associated functions"""

    def test_kernel_1d(self):
        """Test the gaussian kernel in 1d"""
        bw = np.array([2])
        center = np.array([[0]])
        height = 1 / (np.sqrt(2 * np.pi) * 2)  # expected value
        self.assertEqual(bd.kernel_sq_dist(center, bw), height)

        testdist = np.array([[1]])
        self.assertEqual(bd.kernel_sq_dist(testdist, bw), height * np.exp(-1/8))

    def test_kernel_2d(self):
        """Test the gaussian kernel in 2d"""
        bw = np.array([1, 2])
        center = np.array([[0, 0]])
        height = 1 / (4 * np.pi)  # expected value
        self.assertEqual(bd.kernel_sq_dist(center, bw), height)

        testdist = np.array([[1, 2]])
        self.assertEqual(
            bd.kernel_sq_dist(testdist, bw), height * np.exp(-0.5) * np.exp(-0.25)
        )

    def test_walker_density(self):
        """Test evaluation of walker density with all 3 different methods"""
        bw = np.array([0.5])
        positions = np.array([[0],[0.5],[1]])
        dens_pdist = bd.walker_density(positions, bw)  # chooses pdist variant
        dens_kde = bd.walker_density(positions, bw, kde=True)
        dens_manual = bd._walker_density_manual(positions, bw)

        # expected: (kernel values from Wolfram alpha "N(0,0.25) at x=0 / 0.5 / x=1")
        kernel_values = [0.797885, 0.483941, 0.107982]  # 0, 0.5, 1 distance
        expected = np.empty(3)  # mean of kernel values (distance to positions)
        expected[0] = np.mean(kernel_values)
        expected[1] = np.mean([kernel_values[0]] + 2*[kernel_values[1]])
        expected[2] = expected[0]  # reversed positions -> same density as first

        np.testing.assert_allclose(dens_pdist, expected, rtol=1e-6)
        np.testing.assert_allclose(dens_kde, expected, rtol=1e-6)
        np.testing.assert_allclose(dens_manual, expected, rtol=1e-6)

    def test_walker_density_2d(self):
        """Test evaluation of walker density in 2d"""
        bw = np.array([0.5, 2])
        positions = np.array([[0,0],[0.5,-1],[1,-2]])
        dens_pdist = bd.walker_density(positions, bw)  # chooses pdist variant
        dens_kde = bd.walker_density(positions, bw, kde=True)
        dens_manual = bd._walker_density_manual(positions, bw)

        # expected: (kernel values from Wolfram alpha "N(0,0.25) at x=0 / 0.5 / x=1")
        kernel_values = np.zeros((3,2))
        kernel_values[:,0] = np.array([0.797885, 0.483941, 0.107982])
        kernel_values[:,1] = np.array([0.19947114020, 0.17603266338, 0.12098536226])

        expected = np.empty(3)  # mean of kernel values (distance to positions)
        expected[0] = np.mean(np.prod(kernel_values, axis=1))
        kernel_pos1 = np.array([kernel_values[1], kernel_values[0], kernel_values[1]]) # for particle 1
        expected[1] = np.mean(np.prod(kernel_pos1, axis=1))
        expected[2] = expected[0]  # reversed positions -> same density as first

        np.testing.assert_allclose(dens_pdist, expected, rtol=1e-6)
        np.testing.assert_allclose(dens_kde, expected, rtol=1e-6)
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
        kernel = bd.kernel_sq_dist(kernel_points ** 2, bw)
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


if __name__ == "__main__":
    unittest.main()
