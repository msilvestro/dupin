"""Display clustering results."""
# pylint: disable=C0103
# %%
import numpy as np
import matplotlib.pyplot as plt
from clustering.metrics import manhattan_distance
from clustering.evaluation import silhouette_plot

METHOD = 'kmedoids'
DISTANCE = 'manhattan'

data = np.loadtxt('data/warped_curves.gz')
k_range = np.loadtxt(
    'results/clustering/{:}_{:}_k_range.gz'.format(METHOD, DISTANCE),
    dtype=np.uint32)
all_labels = np.loadtxt(
    'results/clustering/{:}_{:}_all_labels.gz'.format(METHOD, DISTANCE),
    dtype=np.uint32)
sil_scores = np.loadtxt(
    'results/clustering/{:}_{:}_sil_scores.gz'.format(METHOD, DISTANCE))
sil_samples = np.loadtxt(
    'results/clustering/{:}_{:}_sil_samples.gz'.format(METHOD, DISTANCE))

# %% Show silhouette scores for each k.
plt.grid(True)
plt.plot(k_range, sil_scores, '.-')
plt.show()

# %% Show for each cluster the mean curve and the 20 nearest curves.
n, max_len = data.shape
amp = int(np.ceil(max(abs(data.max()), abs(data.min()))))

def _get_nearest_curves(center, curves, metric, num=20):
    distances = np.empty(curves.shape[0])
    for i, curve in enumerate(curves):
        distances[i] = metric(center, curve)
    idx = np.argsort(distances)[:num]
    return curves[idx]

k = 8
labels = all_labels[k - k_range[0]]

sil = sil_samples[k - k_range[0]]
silhouette_plot(sil, labels)

unique_labels = np.unique(labels)
mean_curves = np.zeros((k, max_len))
for i, label in enumerate(unique_labels):
    cluster = data[np.where(labels == label)[0]]
    mean_curves[i] = cluster.mean(axis=0)
    print("Members: {:d} ({:.2f}%)".format(
        cluster.shape[0], cluster.shape[0] / n * 100))
    plt.grid(True)
    plt.ylim((-amp, amp))
    plt.axhline(0, c='g', linewidth=3)
    for curve in _get_nearest_curves(mean_curves[i], cluster,
                                     manhattan_distance, 10):
        plt.plot(curve, ':ob')
    plt.plot(mean_curves[i], '-or')
    # plt.plot(data[label], '-og')
    plt.show()
