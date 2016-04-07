"""
This is the class that contains the implementation of competitive selective learning
as proposed by Ueda and Nakano in 1993.
"""

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import check_array, check_random_state


def _calculate_local_distortion(distortions, centers_to_data, normalize=True, gamma=0.8):
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


def _modify_centers(centers, local_distortions, s, random_state, std=1):
    """
    This sends eliminates the s centroids with the smallest local
    distortion and instead creates new centroids very close to the
     centroids with the highest distortion.

    Parameters
    ------------
    centers: ndarray of floats, shape (n_centers, )
        The centers, neurons or centroids

    local_distortions: float ndarray with shape (n_centers, )
        This is a vector that contains the total distortions for each
        center.

    s: int,
        number of centers, centroids, clusters, neurons to change

    random_state: integer or numpy.RandomState, optional
                The generator used to initialize the centers.
                If an integer is given, it fixes the seed. Defaults to the global numpy random
                number generator.

    std: float,
        Standard deviation of the noise added for the creation of new examples

    Returns
    --------------
    centers: ndarray of floats, shape (n_centers, )
        The centers, neurons or centroids

    """
    if centers.ndim > 1:
        n_features = centers.shape[1]
    else:
        n_features = 1

    minimal_distortions = _min_numbers(local_distortions, s)
    maximal_distortions = _max_numbers(local_distortions, s)
    print('min dis', minimal_distortions)
    print('max dis', maximal_distortions)
    # You put more neurons in the area where there is maximal distortion
    for min_index, max_index in zip(minimal_distortions, maximal_distortions):
        # You replaced the neurons of small dis with ones from the big dist
        centers[min_index, ...] = centers[max_index, ...] + std * random_state.rand(n_features)

    return centers


def _max_numbers(vector, N):
    """
    Gives you the index of the N greatest elements in
    vector
    """

    return np.argpartition(-vector, N)[:N][::-1]


def _min_numbers(vector, N):
    """
    Gives you the indexes of the N smallest elements in
    vector
    """

    return np.argpartition(vector, N)[:N]


def _selection_algorithm(centers, distortions, centers_to_data, s, random_state=None):
    """
    This is the selection algorithm, it selects for
    creation and destruction new neurons
    """
    local_distortion = _calculate_local_distortion(distortions,
                                                   centers_to_data, normalize=False)

    centers = _modify_centers(centers, local_distortion, s, random_state)

    return centers


def _competition(X, centers, distortions, n_clusters, eta):
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

    n_clusters: int,
        Number of clusters, centroids or neurons.

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
    # Initialize the dictionary
    centers_to_data = {}
    for center in range(n_clusters):
        centers_to_data[center] = []

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


def _s_sequence(n_iter, s0):
    """
    This functions returns the sequence (s) of centroids, neurons
    or clusters that have to be modified at every iteration step.

    Parameters
    ------------------
    n_iter: int
        Number of iterations
    s0: float
        Initial value at the iteration 0.
    Returns
    -----------
    s_sequence: ndarray of floats shape (n_iter, )
         the sequence of s values.
    """
    time = np.arange(0, n_iter)
    s_half_life = n_iter / 2.0
    s_sequence = np.floor(s0 * np.exp(-time / s_half_life)).astype('int')

    return s_sequence


def _labels(centers_to_data, n_samples):
    """
    Get the assigned labels for each sample

    Parameters
    ----------
    centers_to_data: dictionary
        The keys of this dictionary are each of the centers and the
        items are all the indexes of X that belong to that center in
        the smallest distance sense.

    n_samples: int,
        number of samples

    Returns
    -----------
    labels: ndarray of int
        A vector with each element as the assigned label.

    """

    labels = np.zeros(n_samples)

    for key, indexes in centers_to_data.items():
        for index in indexes:
            labels[index] = key

    return labels


def csl(X, n_clusters=10, n_iter=300, tol=0.001, eta=0.1, s0=1, selection=True, random_state=None):
    """
    Selective and competitive learning. This implements the whole algorithm.

    Parameters
    ---------------------
    n_clusters : int, optional, default=10
            The number of clusters or neurons that the algorithm will try to fit.

    n_iter : int, optional, default=300
        Maximum number of iterations of the k-means algorithm for a single run.

    tol: float, default, 1e-4
        Relative tolerance with regards to distortion before decalring convergence.

    eta: float, optional, default 0.1
        The learning rate.

    s0: float, optional, default 1
       The starting value of neurons to change with the selection policy.

    selection: bool, deafult True
        Controls whether the selection part of the algorithm acutally happens.

    random_state: integer or numpy.RandomState, optional
            The generator used to initialize the centers.
            If an integer is given, it fixes the seed. Defaults to the global numpy random
            number generator.

    Returns
    ------------
    centers: ndarray of floats, shape (n_centers, )
        The centers, neurons or centroids

    distortions: ndarray of floats, shape (n_samples, )
        This numpy array contains the distance between each sample
        and the centroid to which it belongs.

    centers_to_data: dictionary
        The keys of this dictionary are each of the centers and the
        items are all the indexes of X that belong to that center in
        the smallest distance sense.

    labels: ndarray of int
        A vector with each element as the assigned label.

    """
    # Initialize the centers and the distortion
    n_samples, n_features = X.shape
    centers = random_state.rand(n_clusters, n_features)
    distortions = np.zeros(n_samples)

    # Get the s function
    s_sequence = _s_sequence(n_iter, s0)
    print('s_sequence', s_sequence)

    iterations = 0
    while iterations < n_iter:
        # Competition
        centers, distortions, centers_to_data = _competition(X, centers, distortions, n_clusters, eta)
        # Selection
        if iterations < (n_iter - 1):
            centers = _selection_algorithm(centers, distortions,
                                           centers_to_data, s_sequence[iterations], random_state)

        # Increase iterations
        iterations += 1

        # Implement mechanism for tolerance

    # Calculate the final labels
    labels = _labels(centers_to_data, n_samples)
    return centers, distortions, centers_to_data, labels


class CSL(BaseEstimator, TransformerMixin):
    """
    This is the main class for the Competitive and Selective learning algorithm
    The algorithm is implemented in the style of Sklearn.
    """

    def __init__(self, n_clusters=10, n_iter=300, tol=0.001, eta=0.1, s0=1, selection=True, random_state=None):
        """
        Parameters
        ------------
        n_clusters : int, optional, default=10
            The number of clusters or neurons that the algorithm will try to fit.

        n_iter : int, optional, default=300
            Maximum number of iterations of the k-means algorithm for a single run.

        tol: float, default, 1e-4
            Relative tolerance with regards to distortion before decalring convergence.

        eta: float, optional, default 0.1
            The learning rate.

        s0: float, optional, default 1
           The starting value of neurons to change with the selection policy.

        random_state: integer or numpy.RandomState, optional
                The generator used to initialize the centers.
                If an integer is given, it fixes the seed. Defaults to the global numpy random
                number generator.

        Attributes:
        ------------
        centers_: array, [n_clusters, n_features]
            coordinates of clusters or neurons centers.


        distortions_: ndarray of floats, shape (n_samples, )
            This numpy array contains the distance between each sample
            and the centroid to which it belongs

        centers_to_data_: dictionary
                The keys of this dictionary are each of the centers and the
                items are all the indexes of X that belong to that center in
                the smallest distance sense.

        Notes
        ----------------
        This algorithm comes from:
        Ueda, Naonori, and Ryohei Nakano. "A new competitive learning approach based on
        an equidistortion principle for designing optimal vector quantizers.
        " Neural Networks 7.8 (1994): 1211-1227.

        """

        self.n_clusters = n_clusters
        self.n_iter = n_iter
        self.tol = tol
        self.eta = eta
        self.s0 = s0
        self.random_state = random_state

    def _check_fit_data(self, X):
        """Verify that the number of samples given is larger than k"""
        X = check_array(X, accept_sparse='csr', dtype=np.float64)
        if X.shape[0] < self.n_clusters:
            raise ValueError("n_samples=%d should be >= n_clusters=%d" % (
                X.shape[0], self.n_clusters))

        return X

    def fit(self, X, y=None):
        """
        Computer CSL
        """
        random_state = check_random_state(self.random_state)
        X = self._check_fit_data(X)

        self.centers_, self.distortions_, self.centers_to_data_, self.labels_ = \
            csl(X, n_clusters=self.n_clusters, n_iter=self.n_iter,
                tol=self.tol, eta=self.eta, s0=self.s0,
                random_state=random_state)

        return self

    def fit_predict(self, X, y=None):
        """Compute cluster centers and predict cluster index for each sample.
        Convenience method; equivalent to calling fit(X) followed by
        predict(X).
        """
        return self.fit(X).labels_