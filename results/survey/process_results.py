"""Processo information contained into the JSON survey result."""
# pylint: disable=C0103
import json

csv_stories = "story;engagement;coherence;rating;read_again\n"
csv_units = "unit, valence, arousal\n"

with open('results/survey/results.json', 'r', encoding='utf8') as f:
    results = json.load(f)

for result in results:
    # extract all keys to recostruct the linear story unit sequence
    sequence = []
    for key in result.keys():
        if key.startswith("valence_"):
            unit = key[-3:]
            csv_units += "{:}, {:}, {:}\n".format(
                unit,
                result['valence_' + unit],
                result['arousal_' + unit]
            )
            sequence.append(unit)

    csv_stories += "{:};{:};{:};{:};{:}\n".format(
        ','.join(sequence),
        result['engagement'],
        result['coherence'],
        result['rating'],
        result['read_again']
    )

with open('results/survey/results_by_unit.csv', 'w', encoding='utf8') as f:
    f.write(csv_units)
with open('results/survey/results_by_story.csv', 'w', encoding='utf8') as f:
    f.write(csv_stories)
