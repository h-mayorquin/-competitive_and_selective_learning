"""
This is the play 
"""

import numpy as np
import matplotlib.pyplot as plt
import math

from sklearn.datasets import make_blobs
plot = False
verbose = False
tracking = True

# Generate the data
n_samples = 1500
random_state = 20  # Does not converge
random_state = 41
n_features = 2
centers = 3

X, y = make_blobs(n_samples, n_features, centers, random_state=random_state)

# Seed the random number generator
np.random.seed(random_state)

# The algorithm
N = 3
m = 1
D = math.inf
eta = 1.0 / n_samples
neurons = np.random.rand(N, n_features)
D_vector = np.zeros(n_samples)
T = 10

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

def calculate_local_distortion(D_vector, neuron_to_data, gamma=0.8):
    """
    Calculate thes local distortion
    """

    local_distortions = np.zeros(len(neuron_to_data))
    for neuron, data in neuron_to_data.items():
        local_distortions[neuron] = np.sum(D_vector[data])

    # Normalize

    local_distortions = local_distortions**gamma / (local_distortions**gamma).sum()

    return local_distortions

def max_N_numbers(vector, N):
    """
    Gives you the N greatest elements in
    vector
    """

    return np.argpartition(-vector, N)[:N]

def count_new_neurons(g, s):
    """
    Returns a vector with number of new neurons per 
    distortion
    """

    mu = np.floor(g * s).astype('int')
    print('mu', mu)
    aux = g * s - mu
    print('remaining', aux)
    N_left = s - mu.sum()
    indexes = max_N_numbers(aux, N_left)
    print('indxes', indexes)
    mu[indexes] += 1

    return mu
    
local_distortions = np.array([4.5, 16.0, 3.4, 0.1, 1.0])
g = local_distortions**0.8 / (local_distortions**0.8).sum()
print('--------------g------')
print('new_neurons', count_new_neurons(g, 2))

for t in range(T):
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

    if verbose:
        print('winning neuron', closest_neuron)
        print('distances', distances)
    if (t % 10 == 0):
        print('time', t)

if plot:
    # Visualize X
    fig = plt.figure(figsize=(16, 12))
    ax = fig.add_subplot(111)
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
    
    fig.show()
