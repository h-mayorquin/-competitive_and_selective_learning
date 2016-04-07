"""
Here we have the tests of the CSL algorithm
"""

import unittest
import numpy as np
import numpy.testing as nptest
from unittest import TestCase
import csl


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
        labels = csl._labels(centers_to_data, 6)
        result = np.array([0, 0, 1, 0, 1, 2])

        nptest.assert_almost_equal(result, labels)

    def test_min_numbers(self):
        """
        This and the test bellow are wrong (for any other value of s)
        because they assume that the min_numbers returns the minima
        in ordert. It does not.
        """

        s = 2  # Number of minimum to get
        n = 10  # Array size
        array = np.arange(n)
        result = np.arange(s)
        minimum = csl._min_numbers(array, s)
        nptest.assert_almost_equal(result, minimum)

    def test_max_numbers(self):

        s = 2  # Number of minimums to get
        n = 10  # Array size
        array = np.arange(n)
        result = np.arange(n - s, n)
        maximum = csl._max_numbers(array, s)

        nptest.assert_almost_equal(result, maximum)

    def test_modify_centers(self):

        n_clusters = 10
        local_distortions = np.arange(n_clusters) * 10.0

        random_state = np.random

        s_set = random_state.random_integers(0, 5, 100)
        print('s set', s_set)
        for s in s_set:

            # Create 10 centers
            centers = np.arange(n_clusters)
            second_dimension = np.zeros(n_clusters)
            centers_old = np.vstack([centers, second_dimension]).T

            print('s', s)
            # Now lets test algorithm selection
            print('old centers', centers_old)
            print('local dist', local_distortions)
            centers_new = csl._modify_centers(centers_old, local_distortions, s, random_state, std=0)
            print('old centers', centers_old)
            print('new centers', centers_new)

            set_new = set(centers_new[-s:, 0])
            set_old = set(centers_new[:s, 0])

            print('sets', set_new, set_old)

            self.assertEqual(set_new, set_old)


if __name__ == '__main__':
    unittest.main()
