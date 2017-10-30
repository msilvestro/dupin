"""Hierarchical clustering for DoppioGioco."""
# pylint: disable=C0103
from time import time
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from clustering.metrics import dissimilarity_matrix, manhattan_warped_distance

K_MIN = 2
K_MAX = 50
METHOD = "agglomerative_clustering_wd"

csv_file = 'curves_exp.csv'
data = np.array([np.asarray(line.split(','), dtype=np.float)
                           for line in open(csv_file)])
max_len = max([len(ten) for ten in data])
n = len(data)

k_range = np.arange(K_MIN, K_MAX)
sil_scores = np.zeros(K_MAX - K_MIN)
all_labels = np.empty((K_MAX - K_MIN, data.shape[0]), dtype=np.uint32)
diss = dissimilarity_matrix(data, manhattan_warped_distance)

for k in k_range:
    print("## {:} ##".format(k))
    start = time()
    labels = AgglomerativeClustering(k, affinity='precomputed',
                                     linkage='average').fit_predict(diss)
    print("Elapsed time: {:.4f}".format(time() - start))
    sil_score = silhouette_score(diss, labels, metric='precomputed')
    print("Silhouette score: {:.6f}".format(sil_score))
    sil_scores[k - K_MIN] = sil_score
    all_labels[k - K_MIN] = labels

np.savetxt('results/{:}_k_range.gz'.format(METHOD), k_range)
np.savetxt('results/{:}_sil_scores.gz'.format(METHOD), sil_scores)
np.savetxt('results/{:}_all_labels.gz'.format(METHOD), all_labels)
