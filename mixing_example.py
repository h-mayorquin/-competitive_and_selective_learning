import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from functions import get_key_to_indexes_dic



# Generate the data
n_samples = 100
random_state = 20  # Does not converge
# random_state = 41
n_features = 2
centers = 3

X, y = make_blobs(n_samples, n_features, centers, random_state=random_state)

label_to_index = get_key_to_indexes_dic(y)

# Let's plot it first
fig = plt.figure(figsize=(16, 12))
ax = fig.add_subplot(121)
ax.plot(X[:, 0], X[:, 1], '*', markersize=12)

def sample_blobs(X, y, sampling_list):
    """
    Sample the data from X in a way that you extractHej,


    """

# Now let's do the mixing
sampling_list = [0.2, 0.8, 0.4]
sampling_indexes = []

for (label, indexes), coefficient in zip(label_to_index.items(), sampling_list):
    aux = int(len(indexes) * coefficient)
    sampling_indexes.append(indexes[:aux])

sampling_indexes = np.concatenate(sampling_indexes)

X_sampled = X[sampling_indexes, :]
y_sampled = y[sampling_indexes]

# Let's plot it first
ax = fig.add_subplot(122)
ax.plot(X[:, 0], X[:, 1], '*', markersize=12)

fig.show()
