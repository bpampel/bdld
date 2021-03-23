import unittest

from bdld.actions import action

import numpy as np


class ActionTest(unittest.TestCase):
    """Test Action class and functions of action module"""

    def test_action_class(self):
        """Test setting up the base class"""
        act = action.Action()
        with self.assertRaises(NotImplementedError):
            act.run(0)
        self.assertEqual(act.final_run(0), None)

    def test_get_valid_data(self):
        """Test the valid data getter

        Assume some action parameters and check that it returns the right data
        """
        # simulation / action parameters
        step = 63
        update_stride = 2
        write_stride = 40  # not needed, only as memory aid for creating data array
        last_write = 47

        data = np.arange(42, 82, 2)  # data array with time step as elements
        stride = 6  # sparsify data for writing / another action
        valid = action.get_valid_data(data, step, stride, update_stride, last_write)

        expected = np.array([48, 54, 60])  # all multiplicatives of 6 since last_write
        np.testing.assert_array_equal(expected, valid)


if __name__ == "__main__":
    unittest.main()
