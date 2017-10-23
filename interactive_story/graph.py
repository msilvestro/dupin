"""Implement a directed graph in Python."""
from collections import defaultdict
import random


def get_key_ordered_dictionary_repr(dictionary, delimiter=': ',
                                    decorator_in='{', decorator_out='}'):
    """Get a string representation of a dictionary ordered by keys.

    Parameters
    ----------
    dictionary : dictionary
        Dictionary to be represented as a string.
    delimiter : string
        Delimiter to put between every key-value pair.
    decorator_in : string
        Decorator to put at the beginning of the string.
    decorator_out : string
        Decorator to put at the end of the string.

    Returns
    -------
    dict_str : string
        String representation of the key-ordered dictionary.

    Notes
    -----
    In the string representation the dictionary keys are displayed ordered by
    their value and properly indented, e.g.
    {
        'key1': value1,
        'key2': value2,
    }

    """
    dict_str = decorator_in
    for key in sorted(dictionary.keys()):
        dict_str += "\n\t'{}'{}{}".format(key, delimiter, dictionary[key])
    dict_str += '\n{}'.format(decorator_out)
    return dict_str


class Graph:
    """Simple class to handle a directed graph.

    Check
    https://stackoverflow.com/questions/19472530/representing-graphs-data-structure-in-python
    for more informations.
    """

    def __init__(self, edges=None):
        """Initialize the graph with given connections, if any."""
        self._nodes = set()
        # create an empty dictionary of sets (in a safe way)
        self._graph = defaultdict(set)
        # add the edges given, if any
        if edges:
            self.add_edges(edges)

    def add_edges(self, edges):
        """Add edges (list of tuples pairs) to graph."""
        for node1, node2 in edges:
            self.add(node1, node2)

    def add(self, node1, node2):
        """Add an edge between the two nodes."""
        self._graph[node1].add(node2)
        # keep up to date the nodes set
        self._nodes.add(node1)
        self._nodes.add(node2)

    def get_nodes(self):
        """Get all nodes of the graph."""
        return self._nodes

    def belongs_to_graph(self, node):
        """Whether or not a node belongs to the graph."""
        return node in self._nodes

    def get_edge_list(self):
        """Get the list of edges in the graph."""
        edge_list = []
        for node1 in self._graph.keys():
            for node2 in self._graph[node1]:
                edge_list.append((node1, node2))
        return edge_list

    def _get_ordered_edge_list(self):
        # get an ordered list of edges, so that the image of the graph via
        # graphviz doesn't change every time
        edge_list = []
        for node1 in sorted(self._graph.keys()):
            for node2 in sorted(self._graph[node1]):
                edge_list.append((node1, node2))
        return edge_list

    def get_neighbors(self, node):
        """Get all adjacent nodes from a given node."""
        if node in self._graph.keys():
            neighbors = self._graph[node]
        else:
            neighbors = set()
        return neighbors

    def is_source_node(self, node):
        """Whether or not a node is an source one.

        A source node has no incoming edges.
        """
        nonsource_nodes = set()
        for nonsource_node in self._graph.values():
            nonsource_nodes = nonsource_nodes.union(nonsource_node)
        return node not in nonsource_nodes

    def is_end_node(self, node):
        """Whether or not a node is an end one.

        An end node has no outcoming edges.
        """
        return not self.get_neighbors(node)  # empty sequences are false

    def get_source_nodes(self):
        """Get all source nodes, i.e. nodes with no incoming edges.

        It is simply the set difference between all nodes and all non-source
        nodes. All non-source nodes are all nodes that have at least
        an incoming edge.
        """
        nonsource_nodes = set()
        for node in self._graph.values():
            nonsource_nodes = nonsource_nodes.union(node)
        return self.get_nodes().difference(nonsource_nodes)

    def get_end_nodes(self):
        """Get all end nodes, i.e. nodes with no outcoming edges.

        It is simply the set of all nodes that have no outcoming edges.
        """
        return set([node for node in self.get_nodes()
                    if not self.get_neighbors(node)])  # empty seqs are false

    def get_random_path(self, start, end=None, path=None):
        """Get a random path starting from a given node.

        This is a recursive function.
        The choice of nodes is randomly uniform.
        """
        if not end:
            if not self.get_end_nodes():  # empty sequences are false
                print("Warning: no end nodes, paths will be infinite.")
                return

        # safer than setting as `path` default value directly []
        # see http://effbot.org/zone/default-values.htm
        if path is None:
            path = []
        # add this node to the path so far
        path = path + [start]

        if start not in self._nodes and len(path) == 1:
            # if the node is not in the graph but the path has only one
            # element, i.e. this is the starting point of the whole path,
            # it means that we are starting from a non-existing node
            return None
        if (end and start == end) or (not end and self.is_end_node(start)):
            # if we have an end node specified and such end node is reached,
            # return the path
            # if otherwise we have no end node specified but we reached an end
            # node, return the path
            return path

        # otherwise we start from an existing node: choose the next node
        # uniformly random and go ahead with recursion
        next_node = random.sample(self._graph[start], 1)[0]
        return self.get_random_path(next_node, end, path)

    def get_simple_paths(self, start, end=None):
        """Get all simple paths from a node to another.

        If no end node is specified, all paths from the starting node to any
        end node are returned. Also deals with cycles (hence simple paths),
        keeping track of visited nodes.
        Taken from
        https://www.quora.com/How-should-I-find-all-distinct-simple-paths-between-2-given-nodes-in-an-undirected-graph
        """
        if end:
            end_condition = lambda node: node == end
        else:
            if not self.get_end_nodes():  # empty sequences are false
                print("Warning: no end nodes, paths will be infinite.")
                return

            end_condition = self.is_end_node

        return self._get_simple_paths(end_condition, current_path=[start],
                                      visited=set(start), all_paths=[])

    def _get_simple_paths(self, end_condition, current_path, visited,
                          all_paths):
        # we don't need the start node, it is just the last one in the
        # current path
        last_node = current_path[-1]
        if end_condition(last_node):
            # we reached our end condition, i.e. an end node speficied or any
            # of the end nodes, hence we found a new simple path
            # Note: it is important to use the list function to make a new list
            # out of current_path, otherwise we will always have a reference
            # to the exact same list: we will have, hence, a list of the same
            # list repeated over and over
            all_paths.append(list(current_path))
        else:
            for neighbor in self.get_neighbors(last_node):
                if neighbor not in visited:
                    # continue the path, if the node was not already visited
                    # before
                    current_path.append(neighbor)
                    visited.add(neighbor)
                    # continue the path from this node on
                    self._get_simple_paths(end_condition, current_path,
                                           visited, all_paths)
                    # after that, remove the just explored node so that we
                    # get the initial path and go on with the cycle
                    current_path.pop()
                    visited.remove(neighbor)

        return all_paths

    def get_graphviz_graph(self):
        """Display the graph in a graphical way, using graphviz."""
        from graphviz import Digraph
        graph = Digraph(name=self.__class__.__name__, format='pdf')
        graph.edges(self._get_ordered_edge_list())
        return graph

    def __str__(self):
        """Represent the graph as adjacency sets."""
        return get_key_ordered_dictionary_repr(
            self._graph, delimiter=' -> ',
            decorator_in='{}('.format(self.__class__.__name__),
            decorator_out=')'
        )


def _main():
    """Showcase a sample graph, if this file is called directly."""
    edge_list = [('A', 'E'), ('A', 'G'), ('A', 'J'), ('A', 'B'), ('A', 'C'),
                 ('A', 'L'), ('D', 'C'), ('D', 'Z'), ('D', 'F'), ('D', 'B'),
                 ('F', 'E'), ('F', 'Z'), ('F', 'M'), ('F', 'D'), ('G', 'H'),
                 ('G', 'I'), ('G', 'A'), ('I', 'H'), ('I', 'G'), ('J', 'A'),
                 ('J', 'K'), ('J', 'L'), ('H', 'G'), ('H', 'Z'), ('H', 'I'),
                 ('Z', 'F'), ('Z', 'D'), ('Z', 'H'), ('M', 'F'), ('K', 'J'),
                 ('E', 'F'), ('E', 'A'), ('C', 'D'), ('C', 'A'), ('B', 'D'),
                 ('B', 'A'), ('L', 'J'), ('L', 'A')]
    graph = Graph(edge_list)
    print(graph)


if __name__ == '__main__':
    _main()
