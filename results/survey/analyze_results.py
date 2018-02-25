"""Analyze the data obtained via the survey."""
# pylint: disable=C0103
import csv

def compute_score(engagement, coherence, rating, read_again):
    return (engagement + coherence + rating + read_again)/4

output = ""
scores = {}
with open(
    'results/survey/results_by_story.csv', 'r', encoding='utf8'
    ) as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=';')
    csv_it = iter(csv_reader)
    next(csv_it)
    for linear_story in csv_it:
        story, engagement, coherence, rating, read_again = linear_story
        engagement = float(engagement)
        coherence = float(coherence)
        rating = float(rating)
        read_again = 100 * (read_again == 'yes')

        if story in scores.keys():
            print("W: Duplicate! {:}".format(story))
        scores[story] = "{:.2f}".format(
            compute_score(engagement, coherence, rating, read_again)
        )
        output += "{:};{:}\n".format(story, scores[story])

with open('results/survey/story_scores.csv', 'w', encoding='utf8') as f:
    f.write(output)
