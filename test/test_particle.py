import unittest

from bdld import particle


class ParticleTests(unittest.TestCase):
    """Test Particle class"""

    def test_create(self):
        """Initialization of particles"""
        p = particle.Particle(0, 1, 3)
        self.assertEqual(p.pos, 0)
        self.assertEqual(p.mom, 1)
        self.assertEqual(p.mass, 3)

    def test_init_momentum(self):
        """Momentum initialization"""
        p = particle.Particle(0)
        self.assertEqual(p.mom, 0.0)
        with self.assertRaises(ValueError):
            p = particle.Particle([0, 1], [1])  # dimension mismatch


if __name__ == "__main__":
    unittest.main()
