from sklearn.datasets import make_blobs
from sklearn.metrics import silhouette_samples, silhouette_score
from clustering.metrics import euclidean_distance, dissimilarity_matrix
from clustering.kmedoids import pam
from clustering.evaluation import silhouette_samples, silhouette_plot

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np


# generating the sample data from make_blobs
X, y = make_blobs(n_samples=500,
                  n_features=2,
                  centers=4,
                  cluster_std=1,
                  center_box=(-10.0, 10.0),
                  shuffle=True)

k = 4

# silhouette plot
diss = dissimilarity_matrix(X, euclidean_distance)
labels = pam(diss, k)[0]
sil = silhouette_samples(diss, labels)
silhouette_plot(sil, labels)

# # 2nd Plot showing the actual clusters formed
colors = cm.spectral(labels.astype(float) / k)
plt.scatter(X[:, 0], X[:, 1], marker='.', s=30, lw=0, alpha=0.7,
            c=colors, edgecolor='k')
plt.show()

# # Labeling the clusters
# centers = clusterer.cluster_centers_
# # Draw white circles at cluster centers
# ax2.scatter(centers[:, 0], centers[:, 1], marker='o',
#             c="white", alpha=1, s=200, edgecolor='k')

# for i, c in enumerate(centers):
#     ax2.scatter(c[0], c[1], marker='$%d$' % i, alpha=1,
#                 s=50, edgecolor='k')

# # ax2.set_title("The visualization of the clustered data.")
# ax2.set_xlabel("Feature space for the 1st feature")
# ax2.set_ylabel("Feature space for the 2nd feature")

# # plt.suptitle(("Silhouette analysis for KMeans clustering on sample data "
# #                "with n_clusters = %d" % n_clusters),
# #                fontsize=14, fontweight='bold')

# plt.savefig('export/silhouette.pdf')
# plt.show()
