"""
This is the play 
"""
import numpy as np
import matplotlib.pyplot as plt
import math

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
# random_state = 41
n_features = 2
centers = 3

X, y = make_blobs(n_samples, n_features, centers, random_state=random_state)


# The algorithm
N = 3
s = 2  # Number of neurons to change per round
eta = 0.1
T = 50

csl = CSL(n_clusters=N, max_iter=T, tol=0.001, eta=eta, s0=s, random_state=np.random)
csl.fit(X)
neurons = csl.centers

if plot:
    # Visualize X
    fig = plt.figure(figsize=(16, 12))
    ax = fig.add_subplot(111)
    ax.plot(X[:, 0], X[:, 1], 'x', markersize=6)
    ax.hold(True)
    if True:
        ax.plot(neurons[0, 0], neurons[0, 1], 'o', markersize=12, label='neuron 1')
        ax.plot(neurons[1, 0], neurons[1, 1], 'o', markersize=12, label='neuron 2')
        ax.plot(neurons[2, 0], neurons[2, 1], 'o', markersize=12, label='neuron 3')
        ax.legend()

    fig.show()
    
