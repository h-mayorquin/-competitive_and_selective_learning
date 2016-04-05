"""
This is the class that contains the implementation of competitive selective learning
as proposed by Ueda and Nakano in 1993.
"""

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import check_array, check_random_state


def calculate_local_distortion(distortions, centers_to_data, normalize=True, gamma=0.8):
    """
    Calculates the local distortion

    Parameters
    ---------------


    Returns
    ---------------
    local_distortions: float ndarray with shape (n_centers, )
        This is a vector that contains the total distortions for each
        center.
    """

    local_distortions = np.zeros(len(centers_to_data))
    for neuron, data in centers_to_data.items():
        local_distortions[neuron] = np.sum(distortions[data])

    # Normalize
    if normalize:
        local_distortions = local_distortions ** gamma / (local_distortions ** gamma).sum()

    return local_distortions


def modify_centers(centers, local_distortions, s):
    """
    Adds the perturbations to nuerons based on the
    vector mu which contains information about which
    neurons have to dissapear and be created.
    """

    Nfeatures = centers.shape[1]

    minimal_distortions = min_numbers(local_distortions, s)
    maximal_distortions = max_numbers(local_distortions, s)

    # You put more neurons in the area where there is maximal distortion
    for min_index, max_index in zip(minimal_distortions, maximal_distortions):
        # You replaced the neurons of small dis with ones from the big dist
        centers[min_index, :] = centers[max_index, :] + np.random.rand(Nfeatures)

    return centers


def max_numbers(vector, N):
    """
    Gives you the index of the N greatest elements in
    vector
    """

    return np.argpartition(-vector, N)[:N]


def min_numbers(vector, N):
    """
    Gives you the indexes of the N smallest elements in
    vector
    """

    return np.argpartition(vector, N)[:N]


def selection_algorithm(centers, distortions, centers_to_data, s):
    """
    This is the selection algorithm, it selects for
    creation and destruction new neurons
    """
    local_distortion = calculate_local_distortion(distortions,
                                                  centers_to_data, normalize=False)

    centers = modify_centers(centers, local_distortion, s)

    return centers


def _competition(X, centers, distortions, centers_to_data, eta):
    """
    Implements the competition part of the SCL algorithm

    It consists in three parts.

    1. Calculates the distances between the data and the centers
    and for each one picks the minimum.

    2. Modifies the position of the centers according to who is
    closest (winner-takes-all mechanism).

    3. Stores the distance for each data point and to which
    center it belongs.

    Parameters
    -------------
    X: array-like of floats, shape (n_samples, n_features)
        The observations to cluster.

    centers: ndarray of floats, shape (n_centers, )
        The centers, neurons or centroids

    distortions: ndarray of floats, shape (n_samples, )
        This numpy array contains the distance between each sample
        and the centroid to which it belongs.

    centers_to_data: dictionary
        The keys of this dictionary are each of the centers and the
        items are all the indexes of X that belong to that center in
        the smallest distance sense.

    eta: float
        The learning rate.

    Returns
    -------------
    centers: ndarray of floats, shape (n_centers, )
        The centers, neurons or centroids

    distortions: ndarray of floats, shape (n_samples, )
        This numpy array contains the distance between each sample
        and the centroid to which it belongs.

    centers_to_data: dictionary
        The keys of this dictionary are each of the centers and the
        items are all the indexes of X that belong to that center in
        the smallest distance sense.

    """

    for x_index, x in enumerate(X):
        # Conventional competitive learning
        distances = np.linalg.norm(centers - x, axis=1)
        closest_center = np.argmin(distances)

        # Modify center positions
        difference = x - centers[closest_center, :]
        centers[closest_center, :] += eta * difference

        # Store the distance to each center
        distortions[x_index] = distances[closest_center]
        centers_to_data[closest_center].append(x_index)

    return centers, distortions, centers_to_data


def scl(X, n_clusters=10, max_iter=300, tol=0.001, eta=0.1, s=1, selection=True, random_state=None):
    """
    Selective and competitive learning. This implements the whole algorithm.

    Parameters:

    """
    # Initialize the centers and the distortion
    n_samples, n_features = X.shape
    centers = random_state.rand(n_clusters, n_features)
    distortions = np.zeros(n_samples)

    # Get the s function
    time = np.arange(0, max_iter)
    s_half_life = max_iter / 4.0
    s_sequence = np.floor(s * np.exp(-time / s_half_life)).astype('int')

    # Initialize the dictionary
    centers_to_data = {}
    for center in range(n_clusters):
        centers_to_data[center] = []

    iterations = 0
    while iterations < max_iter:
        # Competition
        centers, distortions, centers_to_data = _competition(X, centers, distortions, centers_to_data, eta)
        # Selection
        if selection:
            center = selection_algorithm(centers, distortions,
                                         centers_to_data, s_sequence[iterations])

        # Increase iterations
        iterations += 1

        # Implement mechanism for tolerance

    return centers, distortions, centers_to_data


class CSL(BaseEstimator, TransformerMixin):
    """
    This is the main class for the Competitive and Selective learning algorithm
    The algorithm is implemented in the style of Sklearn.
    """

    def __init__(self, n_clusters=10, max_iter=300, tol=0.001, random_state=None):
        """
        Parameters
        ------------
        n_clusters : int, optional, default=10
            The number of clusters or neurons that the algorithm will try to fit.

        max_iter : int, optional, default=300
            Maximum number of iterations of the k-means algorithm for a single run.

        tol: float, default, 1e-4
            Relative tolerance with regards to distortion before decalring convergence.

        random_state: integer or numpy.RandomState, optional
                The generator used to initialize the centers.
                If an integer is given, it fixes the seed. Defaults to the global numpy random
                number generator.

        Attributes:
        ------------
        cluster_centers_: array, [n_clusters, n_features]
            coordinates of clusters or neurons centers.

        distortion_: float
            the expected distortion over all the data set


        """

        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state

    def _check_fit_data(self, X):
        """Verify that the number of samples given is larger than k"""
        X = check_array(X, accept_sparse='csr', dtype=np.float64)
        if X.shape[0] < self.n_clusters:
            raise ValueError("n_samples=%d should be >= n_clusters=%d" % (
                X.shape[0], self.n_clusters))

    def fit(self, X, y=None):
        """
        Computer CSL
        """
        random_state = check_random_state(self.random_state)
        X = self._check_fit_data(X)

