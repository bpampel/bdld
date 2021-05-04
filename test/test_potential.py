import unittest
import numpy as np

from bdld.potential import polynomial


class PotentialTests(unittest.TestCase):
    """Test Potential class"""

    # exemplary coefficients for 1d and 2d
    c1 = [0, 0, 1]
    c2 = [[0, 0, 2], [1, 0, 0]]

    def test_create(self):
        """Initialization of Potential"""
        pot = polynomial.PolynomialPotential(self.c1)
        np.testing.assert_array_equal(pot.coeffs, self.c1)
        np.testing.assert_array_equal(pot.der[0], [0, 2])
        self.assertFalse(pot.ranges)
        self.assertEqual(f"{pot}", "polynomial with coefficients [0 0 1]")
        with self.assertRaises(ValueError):  # more than 3 dim
            pot = polynomial.PolynomialPotential(np.ones((1, 1, 1, 4)))

    def test_evaluate(self):
        """Test if correct values are returned"""
        pot = polynomial.PolynomialPotential(self.c2)
        testpos = [0.5, 2.0]
        energy, forces = pot.evaluate(testpos)
        self.assertEqual(energy, 8.5)  # x+2*y**2
        np.testing.assert_array_equal(forces, [-1, -8])  # -1,-4y

    def test_reference(self):
        """Test reference function"""
        pot = polynomial.PolynomialPotential(self.c1)
        testpos = np.linspace(0, 5, 6).reshape(6, 1)
        fes = pot.calculate_reference(testpos)
        np.testing.assert_array_equal(fes, [0, 1, 4, 9, 16, 25])

    def test_probability_density(self):
        """Test if probabilities are correct"""
        pot = polynomial.PolynomialPotential([0, 0, 1])
        prob = pot.calculate_probability_density(0.5, [(-1, 1)], [3])
        expect = np.array([np.exp(-2), 1, np.exp(-2)])  # not yet normalized
        np.testing.assert_array_almost_equal(prob.data, expect / np.sum(expect))


if __name__ == "__main__":
    unittest.main()
