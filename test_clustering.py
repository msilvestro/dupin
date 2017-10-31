"""Clustering for DoppioGioco."""
# pylint: disable=C0103
from time import time
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from clustering.kmedoids import pam_npass
from clustering.metrics import (dissimilarity_matrix, _dissimilarity_matrix,
                                euclidean_distance, manhattan_distance,
                                supremum_distance)
from scipy.spatial.distance import cityblock
from fastdtw import fastdtw

K_MIN = 2
K_MAX = 51
METHOD = 'kmedoids'
DISTANCE = 'dtw'
DEBUG = False

data = np.loadtxt('data/curves_exp_zeros.gz')
if DEBUG:
    np.random.shuffle(data)
    data = data[:100]

k_range = np.arange(K_MIN, K_MAX)
sil_scores = np.zeros(K_MAX - K_MIN)
all_labels = np.empty((K_MAX - K_MIN, data.shape[0]), dtype=np.uint32)

if DISTANCE == 'euclidean':
    dm = _dissimilarity_matrix(euclidean_distance)
    diss = dm(data)
elif DISTANCE == 'manhattan':
    dm = _dissimilarity_matrix(manhattan_distance)
    diss = dm(data)
elif DISTANCE == 'supremum':
    dm = _dissimilarity_matrix(supremum_distance)
    diss = dm(data)
elif DISTANCE == 'dtw':
    dtw = lambda x, y: fastdtw(x, y, dist=cityblock)[0]
    diss = dissimilarity_matrix(data, dtw)

for k in k_range:
    print("## {:} ##".format(k))

    start = time()
    if METHOD == 'agglomerative_clustering':
        labels = AgglomerativeClustering(k, affinity='precomputed',
                                         linkage='average').fit_predict(diss)
    elif METHOD == 'kmedoids':
        labels = pam_npass(diss, k, npass=10)[0]
    print("Elapsed time: {:.4f}".format(time() - start))

    sil_score = silhouette_score(diss, labels, metric='precomputed')
    print("Silhouette score: {:.6f}".format(sil_score))
    sil_scores[k - K_MIN] = sil_score
    all_labels[k - K_MIN] = labels

np.savetxt('results/{:}_{:}_k_range.gz'.format(METHOD, DISTANCE), k_range)
np.savetxt('results/{:}_{:}_sil_scores.gz'.format(METHOD, DISTANCE), sil_scores)
np.savetxt('results/{:}_{:}_all_labels.gz'.format(METHOD, DISTANCE), all_labels)
