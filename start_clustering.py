"""Start the clustering."""
# pylint: disable=C0103
from time import time
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples
from clustering.kmedoids import pam_npass
from clustering.metrics import (dissimilarity_matrix, _dissimilarity_matrix,
                                euclidean_distance, manhattan_distance,
                                supremum_distance)

# parameters to be changed
K_MIN = 2  # minimum number of clusters to test
K_MAX = 50  # maximum number of clusters to test
METHOD = 'kmedoids'  # method of clustering
DISTANCE = 'manhattan'  # distance for the clustering

# load the data
data = np.loadtxt('data/warped_curves.gz')

# initialize the vectors
k_range = np.arange(K_MIN, K_MAX)
sil_scores = np.zeros(K_MAX - K_MIN)
all_labels = np.empty((K_MAX - K_MIN, data.shape[0]), dtype=np.uint32)
sil_samples = np.empty((K_MAX - K_MIN, data.shape[0]))

# compute the dissimilarity matrix based on the chosen distance
if DISTANCE == 'manhattan':
    dm = _dissimilarity_matrix(manhattan_distance)
diss = dm(data)

# perform the clustering according to the parameters
for k in k_range:
    # keep track of the ongoing process
    print("## {:} ##".format(k))

    # start the clustering and time it
    start = time()
    if METHOD == 'kmedoids':
        labels = pam_npass(diss, k, npass=10)[0]
    print("Elapsed time: {:.4f}".format(time() - start))

    # compute the silhouettes for the clustering
    sil = silhouette_samples(diss, labels, metric='precomputed')
    sil_score = sil.mean()
    print("Silhouette score: {:.6f}".format(sil_score))
    sil_scores[k - K_MIN] = sil_score
    all_labels[k - K_MIN] = labels
    sil_samples[k - K_MIN] = sil

# save the results
np.savetxt(
    'results/clustering/{:}_{:}_k_range.gz'.format(METHOD, DISTANCE),
    k_range
)
np.savetxt(
    'results/clustering/{:}_{:}_sil_scores.gz'.format(METHOD, DISTANCE),
    sil_scores
)
np.savetxt(
    'results/clustering/{:}_{:}_all_labels.gz'.format(METHOD, DISTANCE),
    all_labels
)
np.savetxt(
    'results/clustering/{:}_{:}_sil_samples.gz'.format(METHOD, DISTANCE),
    sil_samples
)
