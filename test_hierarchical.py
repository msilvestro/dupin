"""Hierarchical clustering for DoppioGioco."""
# pylint: disable=C0103
from time import time
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score

K_MIN = 2
K_MAX = 50
# METHOD = "agglomerative_clustering"
METHOD = "agglomerative_clustering_md"

# extract plain tension curves from the CSV file
csv_file = 'tension_curves.csv'
tension_curves = np.array([np.asarray(line.split(','), dtype=np.uint32)
                           for line in open(csv_file)])
# get information about the tension curves dataset
n = tension_curves.shape[0]
max_len = max([len(ten) for ten in tension_curves])
# and compute the weighted moving average from the plain tension curves
data = np.zeros((n, max_len))
for i in range(n):
    for j in range(tension_curves[i].shape[0]):
        w = [2**k for k in range(j+1)]  # exponential weighting
        # w = [k+1 for k in range(j+1)]  # standard weighting
        data[i, j] = np.average(tension_curves[i][:j+1], weights=w)

k_range = np.arange(K_MIN, K_MAX)
sil_scores = np.zeros(K_MAX - K_MIN)
all_labels = np.empty((K_MAX - K_MIN, data.shape[0]))
for k in k_range:
    print("## {:} ##".format(k))
    start = time()
    if METHOD == "agglomerative_clustering":
        labels = AgglomerativeClustering(k).fit_predict(data)
    elif METHOD == "agglomerative_clustering_md":
        labels = AgglomerativeClustering(k, affinity='manhattan',
            linkage='average').fit_predict(data)
    print("Elapsed time: {:.4f}".format(time() - start))
    if METHOD == "agglomerative_clustering": 
        sil_score = silhouette_score(data, labels)
    elif METHOD == "agglomerative_clustering_md":
        sil_score = silhouette_score(data, labels, metric='manhattan')
    print("Silhouette score: {:.6f}".format(sil_score))
    sil_scores[k - K_MIN] = sil_score
    all_labels[k - K_MIN] = labels

np.savetxt('results/{:}_k_range.gz'.format(METHOD), k_range)
np.savetxt('results/{:}_sil_scores.gz'.format(METHOD), sil_scores)
np.savetxt('results/{:}_all_labels.gz'.format(METHOD), all_labels)
