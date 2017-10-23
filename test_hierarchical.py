"""Hierarchical clustering for DoppioGioco."""
# pylint: disable=C0103
import numpy as np
from time import time
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score


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

k_range = range(2, 40)
for k in k_range:
    print("## {:} ##".format(k))
    data = data[:1000]
    start = time()
    labels = AgglomerativeClustering(k).fit_predict(data)
    print("Elapsed time: {:.4f}".format(time() - start))
    sil_score = silhouette_score(data, labels)
    print("Silhouette score: {:.6f}".format(sil_score))
