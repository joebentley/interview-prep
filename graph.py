from typing import Dict, Any, List
import random
import pdb
from dataclasses import dataclass
from enum import Enum
from collections import deque

Node = Any

class NeitherIsNodeException(Exception):
    pass

@dataclass
class Edge:
    from_node: Node
    to_node: Node
    directed: bool = False
    
    def __repr__(self):
        if self.directed:
            return f"{self.from_node} -> {self.to_node}"
        else:
            return f"{self.from_node} <-> {self.to_node}"
    
    def __eq__(self, value):
        if self.directed:
            return self.from_node == value.from_node and self.to_node == value.to_node
        else:
            return (self.from_node == value.from_node and self.to_node == value.to_node) or \
                   (self.to_node == value.from_node and self.from_node == value.to_node)
    
    def other(self, node):
        """Get the element of `self` that is not `node`.
        
        Throws `NeitherIsNodeException` if neither are `node`."""
        if self.directed:
            if self.from_node == node:
                return self.to_node
        else:
            if self.from_node == node:
                return self.to_node
            elif self.to_node == node:
                return self.from_node
        
        raise NeitherIsNodeException()

@dataclass
class DepthFirstSearchResults:
    discovered_edges: List[Edge]
    back_edges: List[Edge]

@dataclass
class BreadthFirstSearchMetadata:
    distance: int
    predecessor: Node

BreadthFirstSearchResults = Dict[Node, BreadthFirstSearchMetadata]

class DoesNotConnectException(Exception):
    pass

class Graph:
    def random():
        g = Graph()
        nodes = ["A", "B", "C", "D", "E", "F"]
        for node in nodes:
            g.add_node(node)
        for node in nodes:
            for to_node in random.sample(list(filter(lambda n: n != node, nodes)), random.randint(1, 2)):
                g.add_edge(node, to_node)
        return g
    
    def __init__(self, adjacency_list: Dict[str, str] = {}):
        self.adjacency_list = adjacency_list
    
    def add_node(self, node):
        self.adjacency_list[node] = []

    def add_edge(self, from_node, to_node):
        # self.edges.append(edge)
        if to_node not in self.adjacency_list[from_node]:
            self.adjacency_list[from_node].append(to_node)
            self.adjacency_list[to_node].append(from_node)
    
    def __repr__(self):
        s = ""
        for node in self.adjacency_list.keys():
            s += f"{node} : {self.adjacency_list[node]}\n"
        return s.strip()

    def incident_edges(self, node):
        return list(map(lambda to_node: Edge(node, to_node), self.adjacency_list[node]))
    
    def adjacent_node(self, node, edge: Edge):
        other = edge.other(node)
        if other in self.adjacency_list[node]:
            return other
        else:
            raise DoesNotConnectException(f"Node {node} does not connect to Edge {edge}")

    def depth_first_search(self, initial_node) -> DepthFirstSearchResults:
        explored_nodes = []
        discovered_edges = []
        back_edges = []

        def depth_first_search_recursive(node):
            explored_nodes.append(node)
            for edge in self.incident_edges(node):
                if edge not in discovered_edges and edge not in back_edges:
                    w = self.adjacent_node(node, edge)
                    if w not in explored_nodes:
                        discovered_edges.append(edge)
                        depth_first_search_recursive(w)
                    else:
                        back_edges.append(edge)

        depth_first_search_recursive(initial_node)

        return DepthFirstSearchResults(discovered_edges, back_edges)
    
    def breadth_first_search(self, initial_node) -> BreadthFirstSearchResults:
        class Colour(Enum):
            WHITE = 1
            GREY = 2
            BLACK = 3

        @dataclass
        class _BreadthFirstSearchMetadata:
            colour: Colour
            distance: int
            predecessor: Node

        # Setup default node metadata
        metadata: Dict[Node, _BreadthFirstSearchMetadata] = {}
        for node in self.adjacency_list.keys():
            if node == initial_node:
                metadata[node] = _BreadthFirstSearchMetadata(Colour.GREY, 0, None)
            else:
                metadata[node] = _BreadthFirstSearchMetadata(Colour.WHITE, -1, None)
            
        q = deque()
        q.appendleft(initial_node)
        while len(q) > 0:
            u = q.pop()
            u_metadata = metadata[u]
            for v in self.adjacency_list[u]:
                v_metadata = metadata[v]
                if v_metadata.colour == Colour.WHITE:
                    v_metadata.colour = Colour.GREY
                    v_metadata.distance = u_metadata.distance + 1
                    v_metadata.predecessor = u
                    q.appendleft(v)
            u_metadata.colour = Colour.BLACK

        # Strip out the colour metadata
        return { k: BreadthFirstSearchMetadata(v.distance, v.predecessor) for k, v in metadata.items() }

### Testing
import unittest, sys

class TestEdge(unittest.TestCase):
    # TODO: test directed edges
    def test_other(self):
        e = Edge('A', 'B')
        self.assertEqual(e.other('A'), 'B')
        self.assertEqual(e.other('B'), 'A')

    def test_other_raises(self):
        e = Edge('A', 'B')
        with self.assertRaises(NeitherIsNodeException):
            e.other('C')

class TestGraph(unittest.TestCase):
    def setUp(self):
        a_list = {
            'A': ['B', 'C'],
            'B': ['A'],
            'C': []
        }
        self.graph = Graph(a_list)

    def test_incident_edges(self):
        self.assertListEqual(self.graph.incident_edges('A'), [Edge('A', 'B'), Edge('A', 'C')])
    
    def test_adjacent_node(self):
        # TODO: test directed edges
        self.assertEqual(self.graph.adjacent_node('A', Edge('A', 'B')), 'B')
    
    def test_adjacent_node_raises(self):
        with self.assertRaises(DoesNotConnectException):
            self.graph.adjacent_node('C', Edge('C', 'B'))

    def test_depth_first_search(self):
        # borrowed graph from here https://stackoverflow.com/questions/44494426/back-edges-in-a-graph
        test_graph = Graph({
            0: [5, 1],
            1: [0, 2, 3],
            2: [1, 3, 4],
            3: [2, 4, 1],
            4: [3, 2],
            5: [0]
        })
        results = test_graph.depth_first_search(0)
        # NOTE: result will depend on the order of the edges
        self.assertListEqual(results.discovered_edges,
                             [Edge(0, 5), Edge(0, 1), Edge(1, 2), Edge(2, 3), Edge(3, 4)])
        self.assertListEqual(results.back_edges, [Edge(4, 2), Edge(3, 1)]), 

    def test_breadth_first_search(self):
        # from the algorithms book
        test_graph = Graph({
            's': ['r', 'w'],
            'r': ['s', 'v'],
            'v': ['r'],
            'w': ['s', 't', 'x'],
            't': ['u', 'x', 'w'],
            'u': ['t', 'x', 'y'],
            'x': ['w', 't', 'u', 'y'],
            'y': ['x', 'u']
        })
        results = test_graph.breadth_first_search('s')
        self.assertIsNone(results['s'].predecessor)
        self.assertEqual(results['s'].distance, 0)
        self.assertEqual(results['t'].distance, 2)
        self.assertEqual(results['t'].predecessor, 'w')
        self.assertEqual(results['u'].distance, 3)
        self.assertEqual(results['u'].predecessor, 't')
        self.assertEqual(results['y'].distance, 3)
        self.assertEqual(results['y'].predecessor, 'x')
        self.assertEqual(results['v'].distance, 2)
        self.assertEqual(results['v'].predecessor, 'r')

if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == 'test':
        unittest.main(argv=sys.argv[1:])

    g = Graph.random()
    print(g)

    print(f"Depth first search starting at node {g.nodes[0]}")
    print(g.depth_first_search(g.nodes[0]))