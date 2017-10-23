"""Extend a directed graph to handle a story graph."""
import csv
from interactive_story.graph import Graph


class StoryGraph(Graph):
    """Extend the Graph class to create a story graph.

    A story graph is simply a directed graph in which each node (from now on
    called unit) is annotated in some way (e.g. emotions, texts).
    The story graph is also provided with a tension evaluation function, which
    associates a numeric value (tension) to a unit.
    Note that it is not necessary for each unit to be annotated.
    """

    def __init__(self, edges=None, tension_function=None):
        """Initialize the story graph with given edges and emotions, if any."""
        # call the superclass constructor to create the underlying graph
        super().__init__(edges)
        # add initial and final units
        self._initials = self.get_source_nodes()
        self._finals = self.get_end_nodes()
        # add the tension function, used to evaluate the tension value of each
        # unit
        self.tension_function = tension_function

    def load_from_csv(self, graph_csv):
        """Extract the story graph from a CSV file."""
        with open(graph_csv, 'r') as csvfile:
            graph_csv_reader = csv.reader(csvfile, delimiter=',')
            # skip first line (because of the header) and read the whole CSV
            # file
            csv_it = iter(graph_csv_reader)
            next(csv_it)
            for pair in csv_it:
                self.add(pair[0], pair[1])

    def get_initial_units(self):
        """Get all initial units."""
        return self._initials

    def get_final_units(self):
        """Get all final units."""
        return self._finals

    def is_initial_unit(self, unit):
        """Whether or not the given unit is an initial one.

        Remember that there may be some discarded initial units, i.e. units
        that are source nodes (i.e. have no incoming edges) but are not an
        initial point for the story.
        """
        return unit in self.get_initial_units()

    def is_final_unit(self, unit):
        """Whether or not the given unit is a final one.

        Remmber that there may be some unfinished story paths, i.e. a path that
        ends in a end node (i.e. a node with no outcoming edges) but that is
        not actually final story units (i.e. unit that completely ends a
        story).
        """
        return unit in self.get_final_units()

    def get_loose_ends(self):
        """Get loose ends.

        Loose ends are units that are end nodes (i.e. have no outcoming
        edges and hence end a story) but are not actually story finals.
        """
        return self.get_end_nodes().difference(self.get_final_units())

    def get_unit_tension(self, unit):
        """Get the tension value for a single unit."""
        pass

    def get_tension_curve_for_story(self, story):
        """Get a list of tension values for a given story."""
        if self.tension_function:
            return [self.get_unit_tension(unit) for unit in story]
        else:
            print("Warninig: no tension function provided.")

    def get_random_story(self, start=None):
        """Get a random story from the graph."""
        if not start:
            initials = self.get_initial_units()
            if len(initials) == 1:
                start = next(iter(initials))
            else:
                print("Error: multiple starting units, specify one of them.")
                return

        return self.get_random_path(start)

    def get_linear_stories(self, start=None, end=None):
        """Get all linear stories from an initial unit to an end one.

        If no initial unit is specified and there is only one possible unit,
        choose that one. Otherwise, if there are more than one to choose from,
        prompts the user to choose one himself.
        If no end unit is specified, all stories from the initial unit to any
        final unit are returned.
        """
        if not start:
            initials = self.get_initial_units()
            if len(initials) == 1:
                start = next(iter(initials))
            else:
                print("Error: multiple starting units, specify one of them.")
                return

        if end:
            end_condition = lambda unit: unit == end
        else:
            # the difference between this function and get_simple_paths is here
            # in fact, the end condition becomes that a unit is a final one,
            # more strict than requiring that a unit is an end node
            end_condition = self.is_final_unit

        return self._get_simple_paths(end_condition, current_path=[start],
                                      visited=set(start), all_paths=[])

    def get_graphviz_graph(self):
        """Display the graph in a graphical way, using graphviz."""
        from graphviz import Digraph
        graph = Digraph(name=self.__class__.__name__, format='pdf')
        graph.edges(self._get_ordered_edge_list())
        # set orientation to be left to right (LR)
        graph.graph_attr.update(rankdir='LR')
        # node displayed as boxes and not as ellipses
        graph.node_attr.update(shape='box')
        return graph

    def __str__(self):
        """Represent the graph as adjacency sets."""
        graph_str = '{}('.format(self.__class__.__name__)
        for unit in sorted(self.get_nodes()):
            graph_str += "\n\t'{}', {}".format(
                unit, self._graph[unit] or 'none')
        graph_str += '\n)'
        return graph_str
