"""Evaluate emotion consistency."""
# pylint: disable=C0103

# %%
import csv
from interactive_story.dg_story_graph import DoppioGiocoStoryGraph
import numpy as np

# import valence and arousal values from the survey results
valences = {}
arousals = {}
with open(
    'results/survey/units_scores.csv', 'r', encoding='utf8') as csv_file:
    csv_reader = csv.reader(csv_file)
    csv_it = iter(csv_reader)
    next(csv_it)
    for unit in csv_it:
        unit_name, valence, arousal = unit
        valences[unit_name] = float(valence)
        arousals[unit_name] = float(arousal)

doppiogioco = DoppioGiocoStoryGraph()

def get_theoretical_values(unit):
    """Get theoretical values for a unit."""
    tension_teor = doppiogioco.get_unit_tension(unit)
    valence_teor = np.sign(tension_teor)
    arousal_teor = np.abs(tension_teor)

    return valence_teor, arousal_teor

def get_emotion_consistency(unit):
    """Compute emotion consistency for a unit."""
    valence_teor, arousal_teor = get_theoretical_values(unit)

    valence_norm = valences[unit] / 2 - 1
    arousal_norm = arousals[unit]

    perc = (2*np.abs(valence_teor - valence_norm) + np.abs(arousal_teor - arousal_norm)) / 6
    return perc, valence_norm, arousal_norm

threshold = 0.4
count = 0
# display results in a LaTeX table
for unit in sorted(doppiogioco.get_nodes()):
    # get unit emotion
    if unit == '000':
        emotion = ''
    else:
        emotion = doppiogioco.get_unit_emotion(unit)
    # get theoretical values for valence and arousal
    valt, arot = get_theoretical_values(unit)
    if unit in valences.keys():
        # get the score if there is any survey result
        score, vals, aros = get_emotion_consistency(unit)

        if score > threshold:
            count += 1
            fstr = r"{:} & {:} & {:} & {:} & {:.3f} & {:.3f} & \\bfseries {:.2f}\% \\\\"
        else:
            fstr = r"{:} & {:} & {:} & {:} & {:.3f} & {:.3f} & {:.2f}\% \\\\"
        output = fstr.format(
            unit,
            emotion,
            valt,
            arot,
            vals,
            aros,
            score * 100
        )
    else:
        output = "{:} & {:} & {:} & {:} & & & \\\\".format(
            unit,
            emotion,
            valt,
            arot
        )
    print(output)

print("\nCount: {:}".format(count))
