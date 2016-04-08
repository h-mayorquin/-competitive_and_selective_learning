import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from functions import get_key_to_indexes_dic, sample_blobs



# Generate the data
n_samples = 20
random_state = 20  # Does not converge
# random_state = 41
n_features = 2
centers = 2

X, y = make_blobs(n_samples, n_features, centers, random_state=random_state)


# Let's plot it first
fig = plt.figure(figsize=(16, 12))
ax = fig.add_subplot(211)
ax.plot(X[:, 0], X[:, 1], '*', markersize=12)

# Now let's do the mixing
sampling_list = [0.2, 0.8, 0.4]
sampling_list = [0.5, 0.5]
X_sampled, y_sampled = sample_blobs(X, y, sampling_list)

# Let's plot it first
ax2 = fig.add_subplot(212)
ax2.plot(X_sampled[:, 0], X_sampled[:, 1], '*', markersize=12)

plt.show()
