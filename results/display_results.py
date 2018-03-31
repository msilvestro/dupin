"""Display results from clustering and survey."""
# pylint: disable=C0103
# %%
import csv
import numpy as np
import matplotlib.pyplot as plt
from clustering.metrics import manhattan_distance

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
sil_scores = np.loadtxt(
    'results/clustering/{:}_{:}_sil_scores.gz'.format(METHOD, DISTANCE))

# %% Show silhoette values in a plot.
plt.xticks(np.linspace(0, 50, 11))
plt.grid(True)
plt.plot(k_range, sil_scores, '.-')
plt.savefig('export/sil_scores.pdf')
plt.show()

# %% Load survey data.
with open(
    'data/linear_stories.txt', 'r', encoding='utf8') as f:
    index_to_story = f.readlines()
index_to_story = [x.strip() for x in index_to_story]

# create a dictionary that associates each story in the survey to the
# corresponding scores
engagement_scores = {}
coherence_scores = {}
rating_scores = {}
read_again_scores = {}
with open(
    'results/survey/results_by_story.csv', 'r', encoding='utf8') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=';')
    csv_it = iter(csv_reader)
    next(csv_it)
    for linear_story in csv_it:
        story, engagement, coherence, rating, read_again = linear_story
        engagement_scores[story] = float(engagement)
        coherence_scores[story] = float(coherence)
        rating_scores[story] = float(rating)
        # transform 'yes' -> 100, 'no' -> 0
        read_again_scores[story] = 100 * (read_again == 'yes')

# %% Display results
k = 8  # we chose 8 as optimal number of clusters
labels = all_labels[k - k_range[0]]

n, max_len = data.shape
# compute the maximum amplitude of the curves, i.e. how much space to reserve
# for all plots to be 
amp = int(np.ceil(max(abs(data.max()), abs(data.min()))))

def _get_nearest_curves(center, curves, metric, num=20):
    distances = np.empty(curves.shape[0])
    for i, curve in enumerate(curves):
        distances[i] = metric(center, curve)
    idx = np.argsort(distances)[:num]
    return curves[idx]

unique_labels = np.unique(labels)
mean_curves = np.zeros((k, max_len))
for i, label in enumerate(unique_labels):
    members = np.where(labels == label)[0]

    eng_score = 0
    coh_score = 0
    rat_score = 0
    rea_score = 0
    count = 0
    for j in members:
        story = index_to_story[j]
        if story in engagement_scores.keys():
            eng_score += engagement_scores[story]
            coh_score += coherence_scores[story]
            rat_score += rating_scores[story]
            rea_score += read_again_scores[story]
            count += 1
    eng_score = eng_score / count
    coh_score = coh_score / count
    rat_score = rat_score / count
    rea_score = rea_score / count
    mean_score = (eng_score + coh_score + rat_score + rea_score)/4
    print("Engagament score: {:.2f} out of {:d}".format(eng_score, count))
    print("Coherence score: {:.2f} out of {:d}".format(coh_score, count))
    print("Rating score: {:.2f} out of {:d}".format(rat_score, count))
    print("Read again score: {:.2f} out of {:d}".format(rea_score, count))
    print("Mean score: {:.2f} out of {:d}".format(mean_score, count))

    cluster = data[members]
    print("Members: {:d} ({:.2f}%)".format(
        cluster.shape[0], cluster.shape[0] / n * 100))
    plt.grid(True)
    plt.ylim((-amp, amp))
    plt.axhline(0, linewidth=3)
    for curve in _get_nearest_curves(mean_curves[i], cluster,
                                     manhattan_distance, 10):
        plt.plot(curve, ':.b')
    plt.plot(data[label], '-or')
    plt.savefig('export/cluster_{:}.pdf'.format(i))
    plt.show()
