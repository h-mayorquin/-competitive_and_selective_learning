"""
Here we have the tests of the CSL algorithm
"""

import unittest
import numpy as np
import numpy.testing as nptest
from unittest import TestCase
from csl import _labels


class TestCslFunctions(TestCase):
    """
    Here we have the tests for all the auxiliary functions of CSL
    """

    def test_dictionary_to_labels(self):
        """
        This tests the functions that transforms the dictionary
        of centers to labels to a vector of labels
        """

        n_samples = 6
        centers_to_data = {0: [0, 1, 3], 1: [2, 4], 2: [5]}
        labels = _labels(centers_to_data, 6)
        result = np.array([0, 0, 1, 0, 1, 2])

        nptest.assert_almost_equal(result, labels)


if __name__ == '__main__':
    unittest.main()
