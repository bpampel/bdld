"""Test some potential functionality"""
import unittest
import numpy as np

from bdld.potential import polynomial, potential


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
        with self.assertRaises(NotImplementedError):  # more than 3 dim
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

    def test_boundary_conditions(self):
        """Test if boundary conditions are implemented correctly"""
        ranges = [[0.5, 1], [-1, 0]]

        def init_pos_and_mom():
            """(Re)initialize positions and momenta"""
            pos1 = np.array([1.1, -0.5])  # right in first direction
            pos2 = np.array([0.7, -1.1])  # left in second direction
            mom1 = np.array([1, -2])
            mom2 = np.array([1, -2])
            return (pos1, pos2, mom1, mom2)

        pot = potential.Potential()
        pot.ranges = ranges

        pos1, pos2, mom1, mom2 = init_pos_and_mom()
        # check reflective condition
        pot.boundary_condition = potential.BoundaryCondition.reflective
        pot.apply_boundary_condition(pos1, mom1)
        pot.apply_boundary_condition(pos2, mom2)

        np.testing.assert_array_almost_equal(pos1, np.array([1, -0.5]))
        np.testing.assert_array_almost_equal(pos2, np.array([0.7, -1]))
        np.testing.assert_array_equal(mom1, np.array([-1, -2]))
        np.testing.assert_array_equal(mom2, np.array([1, 2]))

        # reset and do periodic condition
        pos1, pos2, mom1, mom2 = init_pos_and_mom()
        pot.boundary_condition = potential.BoundaryCondition.periodic
        pot.apply_boundary_condition(pos1, mom1)
        pot.apply_boundary_condition(pos2, mom2)
        np.testing.assert_array_almost_equal(pos1, np.array([0.6, -0.5]))
        np.testing.assert_array_almost_equal(pos2, np.array([0.7, -0.1]))
        np.testing.assert_array_equal(mom1, np.array([1, -2]))
        np.testing.assert_array_equal(mom2, np.array([1, -2]))


if __name__ == "__main__":
    unittest.main()
