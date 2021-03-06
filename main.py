"""Extract the story graph of DoPPioGioco and perform some analysis."""
# pylint: disable=C0103
# %% Import all necessary modules.
from collections import defaultdict
import statistics
from interactive_story.dg_story_graph import DoppioGiocoStoryGraph
import matplotlib.pyplot as plt  # to display plots
from IPython.display import HTML  # to display HTML pages

doppiogioco = DoppioGiocoStoryGraph()

# %% Display the graph visually.
g = doppiogioco.get_graphviz_graph()
g.render('export/story_graph')

# %% Get all linear stories and show some statistics.
linear_stories = doppiogioco.get_linear_stories()

# %% Story number and story ends statistics.
print("Number of units:\t{}".format(len(doppiogioco.get_nodes())))
print("Number of stories:\t{}".format(len(linear_stories)))
print("Story beninnings:\t{}".format(len(doppiogioco.get_initial_units())))
print("Story endings:\t\t{}".format(len(doppiogioco.get_final_units())))

counts = defaultdict(int)
for story in linear_stories:
    end = story[-1]
    counts[end] += 1
plt.bar(range(len(counts)), counts.values(), align='center')
plt.xticks(range(len(counts)), counts.keys(), rotation='vertical')
# plt.title("Distribution of linear stories among possible endings")
plt.savefig('export/distr_endings.pdf')
plt.show()

# %% Story lengths statistics.
story_lengths = [len(story) for story in linear_stories]

print("Story length average:\t{}".format(statistics.mean(story_lengths)))
print("Story length mode:\t{}".format(statistics.mode(story_lengths)))
print("Story length max:\t{}".format(max(story_lengths)))
print("Story length min:\t{}".format(min(story_lengths)))

plt.hist(story_lengths, bins=range(5, 14), align='left', edgecolor='black')
# plt.title("Histogram of story lengths distribution")
plt.savefig('export/hist_lengths.pdf')
plt.show()

# %% Distribution of linear stories among units.
counts = defaultdict(int)
for story in linear_stories:
    for unit in story:
        counts[unit] += 1
plt.bar(range(len(counts)), counts.values(), align='center')
plt.xticks(range(len(counts)), counts.keys(), rotation='vertical')
plt.title("Distribution of linear stories among units")
plt.show()

# %% Analyze some tension curves.
# remove the first unit, since it is always the starting one, 000, and has
# no emotion associated (hence the first tension value will always be 0)
random_story = doppiogioco.get_random_story()[1:]
print(random_story)
tension_curve = doppiogioco.get_tension_curve_for_story(random_story)
print(tension_curve)
x_grid = range(len(tension_curve))
plt.plot(x_grid, tension_curve, ':o')
plt.grid(True)
plt.xticks(x_grid, random_story, rotation='vertical')
plt.yticks([-2, -1, 0, 1, 2])
plt.title("Tension curve of a random story")
plt.savefig('export/tension_curve.pdf')
plt.show()

# %% Display a random linear story.
random_story = doppiogioco.get_random_story()

HTML(doppiogioco.get_html_linear_story(random_story))

# %% Get all tension curves.
tension_curves = [doppiogioco.get_tension_curve_for_story(story)
                  for story in linear_stories]

# %% Write all linear stories in a text file.
stories_str = ""
for story in linear_stories:
    stories_str += ','.join(story) + '\n'
with open('data/linear_stories.txt', 'w') as text_file:
    text_file.write(stories_str)

# %% Write tension curves in a text file.
curves_str_full = ""  # linear story and tension curve associated
curves_str = ""  # only tension curves
# the first is mainly to check corrispondence between linear stories and
# tension curves
for i, story in enumerate(linear_stories):
    curves_str_full += "{:};{:}\n".format(
        ','.join(story),
        ','.join(str(x) for x in tension_curves[i])
    )
    curves_str += ','.join(str(x) for x in tension_curves[i]) + '\n'
with open('data/tension_curves_full.txt', 'w') as text_file:
    text_file.write(curves_str_full)
with open('data/tension_curves.txt', 'w') as text_file:
    text_file.write(curves_str)
