import numpy as np


def calculate_local_distortion(D_vector, neuron_to_data, normalize=True, gamma=0.8):
    """
    Calculate thes local distortion
    """

    local_distortions = np.zeros(len(neuron_to_data))
    for neuron, data in neuron_to_data.items():
        local_distortions[neuron] = np.sum(D_vector[data])

    # Normalize
    if normalize:
        local_distortions = local_distortions ** gamma / (local_distortions ** gamma).sum()

    return local_distortions


def max_N_numbers(vector, N):
    """
    Gives you the index of the N greatest elements in
    vector
    """

    return np.argpartition(-vector, N)[:N]


def min_N_numbers(vector, N):
    """
    Gives you the indexes of the N smallest elements in
    vector
    """

    return np.argpartition(vector, N)[:N]


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
    print('indexes', indexes)
    mu[indexes] += 1

    return mu


def modify_neurons_mine(neurons, local_distortions, s):
    """
    Adds the perturbations to nuerons based on the
    vector mu which contains information about which
    neurons have to dissapear and be created.
    """

    Nfeatures = neurons.shape[1]

    minimal_distortions = min_N_numbers(local_distortions, s)
    maximal_distortions = max_N_numbers(local_distortions, s)

    # You put more neurons in the area where there is maximal distortion
    for min_index, max_index in zip(minimal_distortions, maximal_distortions):
        # You replaced the neurons of small dis with ones from the big dist 
        neurons[min_index, :] = neurons[max_index, :] + np.random.rand(Nfeatures)

    return neurons


def selection_algorithm(neurons, D_vector, neuron_to_data, s):
    """
    This is the selection algorithm, it selects for
    creation and destruction new neurons 
    """
    distortion = calculate_local_distortion(D_vector,
                                            neuron_to_data, normalize=False)

    neurons = modify_neurons_mine(neurons, distortion, s)

    return neurons


def calculate_distortion(neurons, data, labels):
    """
    This functions returns the mean distortion
    calculate by taking the distance between the
    each point and the neuron that is labeld with
    """

    point_distortion = np.zeros_like(labels)
    for index, x in enumerate(data):
        distance = np.linalg.norm(x - neurons[labels[index]])
        point_distortion[index] = distance

    return np.mean(point_distortion)


def get_key_to_indexes_dic(labels):
    """
    Builds a dictionary whose keys are the labels and whose
    items are all the indexes that have that particular key
    """

    # Get the unique labels and initialize the dictionary
    label_set = set(labels)
    key_to_indexes = {}

    for label in label_set:
        key_to_indexes[label] = np.where(labels == label)[0]

    return key_to_indexes


def competition(X, centers, distortions, centers_to_data, eta):
    """
    Implements the competition part of the SCL algorithm
    That is, it goes through all the examples and modifies the
    centers of the winners according to this result
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
    Selective and competitive learning
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
        centers, distortions, centers_to_data = competition(X, centers, distortions, centers_to_data, eta)
        # Selection
        if selection:
            center = selection_algorithm(centers, distortions,
                                         centers_to_data, s_sequence[iterations])

        # Increase iterations
        iterations += 1

        # Implement mechanism for tolerance

    return centers, distortions, centers_to_data
