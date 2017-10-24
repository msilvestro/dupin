"""Hierarchical clustering for DoppioGioco."""
# pylint: disable=C0103
from time import time
import numpy as np
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import silhouette_score
from clustering.kmedoids import pam_npass
from clustering.metrics import (_dissimilarity_matrix, euclidean_distance,
                                manhattan_distance)

K_MIN = 2
K_MAX = 50
# METHOD = "agglomerative_clustering_ed"
METHOD = "agglomerative_clustering_md"
# METHOD = "kmeans"
# METHOD = "kmedoids_ed"
# METHOD = "kmedoids_md"

data = np.loadtxt('data.gz')

k_range = np.arange(K_MIN, K_MAX)
sil_scores = np.zeros(K_MAX - K_MIN)
all_labels = np.empty((K_MAX - K_MIN, data.shape[0]), dtype=np.uint32)
if METHOD == "kmedoids_ed":
    dissimilarity_matrix = _dissimilarity_matrix(euclidean_distance)
    diss = dissimilarity_matrix(data)
elif METHOD == "kmedoids_md":
    dissimilarity_matrix = _dissimilarity_matrix(manhattan_distance)
    diss = dissimilarity_matrix(data)
for k in k_range:
    print("## {:} ##".format(k))
    start = time()
    if METHOD == "agglomerative_clustering_ed":
        labels = AgglomerativeClustering(k).fit_predict(data)
    elif METHOD == "agglomerative_clustering_md":
        labels = AgglomerativeClustering(k, affinity='manhattan',
                                         linkage='average').fit_predict(data)
    elif METHOD == "kmeans":
        labels = KMeans(k).fit_predict(data)
    elif METHOD in ("kmedoids_ed", "kmedoids_md"):
        labels = pam_npass(diss, k, npass=10)[0]
    print("Elapsed time: {:.4f}".format(time() - start))
    if METHOD in ("agglomerative_clustering_ed", "kmeans", "kmedoids_ed"):
        sil_score = silhouette_score(data, labels)
    elif METHOD in ("agglomerative_clustering_md", "kmedoids_md"):
        sil_score = silhouette_score(data, labels, metric='manhattan')
    print("Silhouette score: {:.6f}".format(sil_score))
    sil_scores[k - K_MIN] = sil_score
    all_labels[k - K_MIN] = labels

np.savetxt('results/{:}_k_range.gz'.format(METHOD), k_range)
np.savetxt('results/{:}_sil_scores.gz'.format(METHOD), sil_scores)
np.savetxt('results/{:}_all_labels.gz'.format(METHOD), all_labels)
