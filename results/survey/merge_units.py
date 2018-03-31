"""Merge unit results to have a single score per unit."""
# pylint: disable=C0103
import csv

units = {}
with open(
    'results/survey/results_by_unit.csv', 'r', encoding='utf8'
    ) as csv_file:
    csv_reader = csv.reader(csv_file)
    csv_it = iter(csv_reader)
    next(csv_it)
    for unit_score in csv_it:
        unit, valence, arousal = unit_score
        valence = int(valence)
        arousal = int(arousal)

        if unit in units.keys():
            units[unit][0] += valence
            units[unit][1] += arousal
            units[unit][2] += 1
        else:
            units[unit] = [valence, arousal, 1]

output = "unit,valence,arousal\n"
for unit, scores in units.items():
    output += "{:},{:.2f},{:.2f}\n".format(
        unit, scores[0]/scores[2], scores[1]/scores[2])

with open('results/survey/units_scores.csv', 'w', encoding='utf8') as f:
    f.write(output)
