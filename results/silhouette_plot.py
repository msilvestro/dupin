"""Show the silhouette plot for a clustering solution."""
# pylint: disable=C0103
# %%
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

# %% Load clustering data.
METHOD = 'kmedoids'
DISTANCE = 'manhattan'

data = np.loadtxt('data/warped_curves.gz')
k_range = np.loadtxt(
    'results/clustering/{:}_{:}_k_range.gz'.format(METHOD, DISTANCE),
    dtype=np.uint32)
all_labels = np.loadtxt(
    'results/clustering/{:}_{:}_all_labels.gz'.format(METHOD, DISTANCE),
    dtype=np.uint32)
sil_samples = np.loadtxt(
    'results/clustering/{:}_{:}_sil_samples.gz'.format(METHOD, DISTANCE))
sil_scores = np.loadtxt(
    'results/clustering/{:}_{:}_sil_scores.gz'.format(METHOD, DISTANCE))

# %% Show the silhouette pot for a given number of clusters.
k = 8

sil_score = sil_scores[8 - k_range[0]]
sil_sample = sil_samples[8 - k_range[0]]
labels = all_labels[8 - k_range[0]]

y_lower = 10
for i, label in enumerate(np.unique(labels)):
    # Aggregate the silhouette scores for samples belonging to the current
    # cluster, and sort them
    cluster_sil_values = sil_sample[labels == label]
    cluster_sil_values.sort()

    cluster_size = cluster_sil_values.shape[0]
    y_upper = y_lower + cluster_size

    color = cm.spectral(float(i) / k)
    plt.fill_betweenx(np.arange(y_lower, y_upper), 0, cluster_sil_values,
                      facecolor=color, edgecolor=color, alpha=0.7)

    # Label the silhouette plots with their cluster numbers at the middle
    plt.text(-0.05, y_lower + 0.5 * cluster_size, str(i+1))

    # Compute the new y_lower for next plot
    y_lower = y_upper + 10  # 10 for the 0 samples

# The vertical line for average silhouette score of all the values
plt.axvline(x=sil_score, color="red", linestyle="--")
figure = plt.gcf() # get current figure
figure.set_size_inches(8, 8) # set size
plt.savefig('export/silhouette_plot.pdf')
plt.show()
