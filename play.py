"""
This is the play 
"""
import numpy as np
import matplotlib.pyplot as plt
import math

from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from functions import selection_algorithm, scl
from csl import CSL

plot = True
verbose = False
tracking = True
selection = False

# Generate the data
n_samples = 1500
random_state = 20  # Does not converge
random_state = 41
random_state = 105  # Does not converge
random_state = 325325
random_state = 1111
n_features = 2
centers = 7

X, y = make_blobs(n_samples, n_features, centers, random_state=random_state)

# The algorithm
N = centers
s = 2  # Number of neurons to change per round
eta = 0.1
T = 100

csl = CSL(n_clusters=N, n_iter=T, tol=0.001, eta=eta, s0=s, random_state=np.random)
csl.fit(X)
neurons = csl.centers_

if False:
    kmeans = KMeans(n_clusters=N)
    kmeans.fit(X)
    neurons = kmeans.cluster_centers_

if plot:
    # Visualize X
    fig = plt.figure(figsize=(16, 12))
    ax = fig.add_subplot(111)
    ax.plot(X[:, 0], X[:, 1], 'x', markersize=6)
    ax.hold(True)
    if True:
        for n in range(N):
            ax.plot(neurons[n, 0], neurons[n, 1], 'o', markersize=12, label='neuron ' + str(n))

        ax.legend()

    # fig.show()
    plt.show()
    
