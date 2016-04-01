"""
Compare the sklearn algorithms
"""

import numpy as np
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from functions import calculate_distortion, get_key_to_indexes_dic



# Generate the data
n_samples = 1500
random_state = 20  # Does not converge
# random_state = 41
n_features = 2
centers = 3

X, y = make_blobs(n_samples, n_features, centers, random_state=random_state)

mixing_list = [0.2, 0.4, 0.4]


###########
# Let's try k-means
###########
kmeans = KMeans(n_clusters=centers)
kmeans.fit(X)
centers = kmeans.cluster_centers_
labels = kmeans.labels_

# Now let's calculate the distortion
distortion = calculate_distortion(centers, X, labels)

###########
# NOW
###########


