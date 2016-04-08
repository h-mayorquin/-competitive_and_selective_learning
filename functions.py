import numpy as np


def get_key_to_indexes_dic(labels):
    """
    Builds a dictionary whose keys are the labels and whose
    items are all the indexes that have that particular key.
    """

    # Get the unique labels and initialize the dictionary
    label_set = set(labels)
    key_to_indexes = {}

    for label in label_set:
        key_to_indexes[label] = np.where(labels == label)[0]

    return key_to_indexes


def sample_blobs(X, y, sampling_list):
    """
    Sample the data from X in a way that you extract the
    percentages from each cluster in the proportions
    passed in sampling_list.
    """
    sampling_indexes = []

    label_to_index = get_key_to_indexes_dic(y)

    for (label, indexes), coefficient in zip(label_to_index.items(), sampling_list):
        aux = int(len(indexes) * coefficient)
        sampling_indexes.append(indexes[:aux])

    sampling_indexes = np.concatenate(sampling_indexes)

    X_sampled = X[sampling_indexes, :]
    y_sampled = y[sampling_indexes]

    return X_sampled, y_sampled


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