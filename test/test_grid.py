import unittest
import numpy as np

from bdld import grid


class GridTests(unittest.TestCase):
    """Test Grid class and associated functions"""

    def test_create(self):
        """Test if methods create grid with same properties"""
        g1 = grid.from_npoints([(0, 1)], 11)
        g2 = grid.from_stepsizes([(0, 1)], 0.1)
        self.assertEqual(g1.ranges, g2.ranges)
        self.assertEqual(g1.n_points, g2.n_points)
        self.assertEqual(g1.stepsizes, g2.stepsizes)
        self.assertEqual(g1.n_dim, g2.n_dim)
        self.assertTrue(g1 == g2)  # test internal __eq__

    def test_axes(self):
        """Test if correct axesare returned"""
        g1 = grid.from_npoints([(0, 1), (0, 10)], 3)
        expected = [np.array([0, 0.5, 1]), np.array([0, 5, 10])]
        np.testing.assert_array_equal(g1.axes(), expected)

    def test_points(self):
        """Test if correct points are returned"""
        g1 = grid.from_npoints([(0, 1), (0, 10)], 3)
        expected = np.array(
            [
                [0, 0],
                [0, 5],
                [0, 10],
                [0.5, 0],
                [0.5, 5],
                [0.5, 10],
                [1, 0],
                [1, 5],
                [1, 10],
            ]
        )
        np.testing.assert_array_equal(g1.points(), expected)

    def test_data(self):
        """Test if data can be set and received"""
        g1 = grid.from_npoints([(0, 1)], 11)
        with self.assertRaises(ValueError):
            g1.data = np.arange(10)  # too small
        g1.data = np.arange(11)  # data.setter
        np.testing.assert_array_equal(g1.data, np.arange(11))

    def test_set_from_func(self):
        """Test if grid data can be set from function"""
        g1 = grid.from_npoints([(0, 10)], 11)
        g1.set_from_func(lambda x: x ** 2)
        np.testing.assert_array_equal(g1.data, np.square(np.arange(0, 11)))

    def test_arithmetics(self):
        """Test some of the aritmetics"""
        g1 = grid.from_npoints([(0, 10)], 11)
        g1.data = np.arange(11)
        g2 = g1 + 1  # __add__
        g2 -= 1  # __rsub__
        self.assertTrue(g1 == g2)

    def test_exp_log(self):
        """Test if passing exp and log to numpy works"""
        g1 = grid.from_npoints([(0, 10)], 11)
        g1.data = np.arange(11)
        g1 = np.exp(g1)
        np.testing.assert_array_equal(g1.data, np.exp(np.arange(11)))
        g2 = np.log(g1)
        np.testing.assert_array_equal(g2.data, np.arange(11))

    def test_interpolation_sparsify(self):
        """Test the linear interpolation and sparsify"""
        g1 = grid.from_npoints([(0, 10)], 11)
        g1.data = np.arange(11)
        g2 = g1.sparsify([5], "linear")
        expected = np.array([0, 2.5, 5, 7.5, 10])
        np.testing.assert_array_equal(g2.data, expected)

    def test_normalize(self):
        """Test grid normalization"""
        g1 = grid.from_npoints([(0, 1)], 5)
        g1.data = np.arange(5)
        integral = 2.0
        g2 = g1.normalize(integral)
        normfactor = 10 / 4  # sum of values / values per integer
        expected = np.arange(5) / normfactor * integral
        np.testing.assert_array_almost_equal(g2.data, expected)


if __name__ == "__main__":
    unittest.main()
