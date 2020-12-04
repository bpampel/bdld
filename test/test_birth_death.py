import unittest
import numpy as np

from bdld import birth_death as bd
from bdld import grid  # needed for some inputs


class BirthDeathTests(unittest.TestCase):
    """Test BirthDeath class and associated functions"""

    def test_kernel_1d(self):
        """Test the gaussian kernel in 1d"""
        bw = np.array([1])
        center = np.array([[0]])
        height = 1 / np.sqrt(2 * np.pi)  # expected value
        self.assertEqual(bd.kernel_sq_dist(center, bw), height)

        testdist = np.array([[1]])
        self.assertEqual(bd.kernel_sq_dist(testdist, bw), height * np.exp(-0.5))

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
