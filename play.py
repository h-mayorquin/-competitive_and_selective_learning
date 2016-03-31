"""
This is the play 
"""

import numpy as np
import matplotlib.pyplot as plt
import math

from sklearn.datasets import make_blobs
from functions import selection_algorithm

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

# Seed the random number generator
np.random.seed(random_state)

# The algorithm
N = 3
m = 1
s = 2  # Number of neurons to change per round
D = math.inf
eta = 1.0 / n_samples
eta = 0.1
neurons = np.random.rand(N, n_features)
D_vector = np.zeros(n_samples)
T = 50

# Initialize neuron to data hash with empty list
neuron_to_data = {}
for neuron in range(N):
    neuron_to_data[neuron] = []

follow_neuron_0_x = []
follow_neuron_0_y = []

follow_neuron_1_x = []
follow_neuron_1_y = []

follow_neuron_2_x = []
follow_neuron_2_y = []

total_distortion = []

time = np.arange(T)
s_half_life = 10
s_0 = 2
s_sequence = np.floor(s_0 * np.exp(-time / s_half_life)).astype('int')

for t, s in zip(time, s_sequence):
    # Data loop
    for x_index, x in enumerate(X):
        # Conventional competitive learning
        distances = np.linalg.norm(neurons - x, axis=1)
        closest_neuron = np.argmin(distances)
        # Modify neuron weight
        difference = x - neurons[closest_neuron, :]
        neurons[closest_neuron, :] += eta * difference

        # Store the distance to each
        D_vector[x_index] = np.linalg.norm(neurons[closest_neuron, :] - x)
        neuron_to_data[closest_neuron].append(x_index)


    if tracking: 
        follow_neuron_0_x.append(neurons[0, 0])
        follow_neuron_0_y.append(neurons[0, 1])

        follow_neuron_1_x.append(neurons[1, 0])
        follow_neuron_1_y.append(neurons[1, 1])

        follow_neuron_2_x.append(neurons[2, 0])
        follow_neuron_2_y.append(neurons[2, 1])

    # Selection
    if selection:
        neurons = selection_algorithm(neurons, D_vector, neuron_to_data, s)

    if verbose:
        print('winning neuron', closest_neuron)
        print('distances', distances)
    if (t % 10 == 0):
        print('time', t)

    total_distortion.append(np.sum(D_vector))

if plot:
    # Visualize X
    fig = plt.figure(figsize=(16, 12))
    ax = fig.add_subplot(211)
    ax.plot(X[:, 0], X[:, 1], 'x', markersize=6)
    ax.hold(True)
    if False:
        ax.plot(neurons[0, 0], neurons[0, 1], 'o', markersize=12, label='neuron 1')
        ax.plot(neurons[1, 0], neurons[1, 1], 'o', markersize=12, label='neuron 2')
        ax.plot(neurons[2, 0], neurons[2, 1], 'o', markersize=12, label='neuron 3')
        ax.legend()

    if tracking:
        ax.plot(follow_neuron_0_x, follow_neuron_0_y, 'o-', markersize=12)
        ax.plot(follow_neuron_1_x, follow_neuron_1_y, 'o-', markersize=12)
        ax.plot(follow_neuron_2_x, follow_neuron_2_y, 'o-', markersize=12)
    


    ax2 = fig.add_subplot(212)
    ax2.plot(time, total_distortion)

    fig.show()
    
