"""Story graph for the interactive storytelling system DoppioGioco."""
from collections import defaultdict
import csv
import os
from interactive_story.tension_evaluation import get_tension_value
from interactive_story.story_graph import StoryGraph

STORY_GRAPH_CSV = os.path.join(os.path.dirname(__file__),
                               'data/story_graph.csv')
UNIT_EMOTIONS_CSV = os.path.join(os.path.dirname(__file__),
                                 'data/unit_emotions.csv')
UNIT_DETAILS_CSV = os.path.join(os.path.dirname(__file__),
                                'data/unit_details.csv')
UNIT_TEXTS_CSV = os.path.join(os.path.dirname(__file__),
                              'data/unit_texts.csv')

class DoppioGiocoStoryGraph(StoryGraph):
    """Extends the StoryGraph class to fit DoppioGioco."""

    def __init__(self):
        """Initialize the story graph for DoppioGioco."""
        super().__init__()
        self.load_from_csv(STORY_GRAPH_CSV)
        self._emotions = defaultdict()
        self.load_emotions_from_csv(UNIT_EMOTIONS_CSV)
        self.tension_function = get_tension_value
        self._clip_uris = defaultdict()
        self._initials = set()
        self._finals = set()
        self.load_units_details_from_csv(UNIT_DETAILS_CSV)
        self._texts = defaultdict()
        self.load_unit_texts_from_csv(UNIT_TEXTS_CSV)

    def load_emotions_from_csv(self, emotions_csv):
        """Extract the emotions associated with units from a CSV file."""
        with open(emotions_csv, 'r') as csv_file:
            emotions_csv_reader = csv.reader(csv_file, delimiter=',')
            csv_it = iter(emotions_csv_reader)
            next(csv_it)
            for pair in csv_it:
                title, emotion = pair[0], pair[1]
                if self.belongs_to_graph(title):
                    # annotate the emotion only if the unit actually belongs
                    # to the story graph, otherwise it is useless
                    self.annotate_emotion(title, emotion)

    def load_units_details_from_csv(self, details_csv):
        """Load all unit details from a CSV file."""
        with open(details_csv, 'r') as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            csv_it = iter(csv_reader)
            next(csv_it)
            for detail in csv_it:
                title, clip_uri, initial, final = detail
                if self.belongs_to_graph(title):
                    # add unit details only if the unit actually belongs to the
                    # story graph
                    if clip_uri != "NULL":
                        self._clip_uris[title] = clip_uri
                    if int(initial) == 1:
                        self._initials.add(title)
                    if int(final) == 1:
                        self._finals.add(title)

    def load_unit_texts_from_csv(self, texts_csv):
        """Load all unit texts from a CSV file.

        There is a separate CSV for texts for two reasons:
        * texts are very long, hence the details CSV is much smaller without
          them;
        * texts may be problematic for encoding, so it is better to handle them
          separately.
        """
        with open(texts_csv, 'r', encoding='utf8') as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            csv_it = iter(csv_reader)
            next(csv_it)
            for row in csv_it:
                # skip empty lines
                if not row:  # empty sequences are false
                    continue

                title, text = row
                if self.belongs_to_graph(title):
                    # add unit details only if the unit actually belongs to the
                    # story graph
                    if text != "NULL":
                        self._texts[title] = text


    def has_emotion(self, unit):
        """Whether or not the unit has an annotated emotion."""
        return unit in self._emotions.keys()

    def annotate_emotion(self, unit, emotion):
        """Annotate a unit with an emotion."""
        self._emotions[unit] = emotion

    def get_unit_emotion(self, unit):
        """Get the emotion associated with a unit."""
        return self._emotions[unit]

    def get_unit_tension(self, unit):
        """Get the tension value for a single unit."""
        if self.has_emotion(unit):
            tension_value = self.tension_function(self.get_unit_emotion(unit))
        else:
            tension_value = 0
        return tension_value

    def has_clip(self, unit):
        """Whether or not the unit has an associated clip."""
        return unit in self._clip_uris.keys()

    def get_unit_clip(self, unit):
        """Get the clip URI associated to the given unit."""
        return self._clip_uris[unit]

    def has_text(self, unit):
        """Whether or not the unit has an associated text."""
        return unit in self._texts.keys()

    def get_unit_text(self, unit):
        """Get the text associated to the given unit."""
        return self._texts[unit]


    def get_html_linear_story(self, story):
        """Create an HTML page to display a linear story."""
        import html

        html_story = """<html>
<head>
<meta charset="UTF-8">
</head>
<body>
<table>"""
        for unit in story:
            html_story += '<tr>'
            # add unit title
            html_story += '<td>{}</td>\n'.format(html.escape(unit))
            if self.has_text(unit):
                text = self.get_unit_text(unit)
                text = html.escape(text).replace('\n', '<br />')
            else:
                text = "missing"
            # add unit text
            html_story += '<td>{}</td>\n'.format(text)
            html_story += '</tr>\n'
        html_story += """</body>
</html>"""
        return html_story

    def get_graphviz_graph(self):
        """Display the graph in a graphical way, using graphviz."""
        from graphviz import Digraph
        graph = Digraph(name=self.__class__.__name__, format='pdf')
        for unit in sorted(self.get_nodes()):
            if unit == '000':
                color = '#000000'
            else:
                color = self._get_emotion_color(self.get_unit_emotion(unit))
            graph.node(unit, style='filled', color='black',
                       fillcolor=color, fontcolor='white')
        graph.edges(self._get_ordered_edge_list())
        # set orientation to be left to right (LR)
        graph.graph_attr.update(rankdir='LR')
        # node displayed as boxes and not as ellipses
        graph.node_attr.update(shape='circle')
        # group together similar units
        graph.body.append(self._get_unit_ranks())
        return graph

    @staticmethod
    def _get_emotion_color(emotion):
        # helper for drawing the graph, associate a color to each emotion
        positive_high = '#0000ff'
        positive_low = '#ffc0bf'
        negative_low = '#c0c0ff'
        negative_high = '#ff0000'
        emotions_to_color = {
            "joy": positive_high,
            "amusement": positive_high,
            "pride": positive_high,

            "pleasure": positive_low,
            "relief": positive_low,
            "interest": positive_low,

            "hot anger": negative_high,
            "panic fear": negative_high,
            "despair": negative_high,

            "irritation": negative_low,
            "anxiety": negative_low,
            "sadness": negative_low
        }
        return emotions_to_color[emotion]

    @staticmethod
    def _get_unit_ranks():
        # helper for drawing the graph, group together similar units
        return """{rank = same; 001 002 003 004}
{rank = same; 005 006 007 010}
{rank = same; 009 011 012 013}
{rank = same; 014 015 016}
{rank = same; 017 018 019 020 021}
{rank = same; 022 023 024 025}
{rank = same; 061 026 027 028}
{rank = same; 029 030 031 032}
{rank = same; 033 034 035 036}
{rank = same; 037 038 039 040}
{rank = same; 041 042 047 048}
{rank = same; 049 050 051 052}
{rank = same; 053 054 055 056}
{rank = same; 057 058 059 060 062}
{rank = same; 063 064 065 066}
{rank = same; 067 068 069 070}
{rank = same; 071 072 073 074}
{rank = same; 075 076 077 078}
{rank = same; 079 080 081 082}
{rank = same; 083 084 085 086}
{rank = same; 087 088 089 090}
{rank = same; 107 108 109 110}"""
