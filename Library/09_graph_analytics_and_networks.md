# Graph Analytics and Networks

---

> **Field** — Network Science, Graph Theory
> **Scope** — Graph structures, centrality measures,
> pathfinding, community detection, and the NetworkX
> library for Python

---

## Overview

A graph is a data structure made of dots (nodes) and
lines connecting them (edges). Graphs represent
relationships: who follows whom on social media, which
airports have flights between them, or which web pages
link to each other. Graph analytics extracts insights
from these relationship structures, revealing which
nodes are most important and how information flows
through a network.

---

## Definitions

### `Graph`

**Definition.**
A graph is a mathematical structure consisting of a
set of nodes (also called vertices) and a set of edges
(also called links) that connect pairs of nodes. It is
the fundamental building block of network analysis.

**Context.**
Graphs show up everywhere in data science. Social
networks, transportation systems, supply chains,
citation networks, and biological pathways are all
naturally modeled as graphs. When your data is about
relationships rather than rows and columns, graph
analysis is often the right tool.

**Example.**
Creating a simple graph in NetworkX:

```python
import networkx as nx

G = nx.Graph()
G.add_nodes_from(["Alice", "Bob", "Carol"])
G.add_edges_from([
    ("Alice", "Bob"),
    ("Bob", "Carol"),
    ("Alice", "Carol")
])

print(f"Nodes: {G.number_of_nodes()}")
print(f"Edges: {G.number_of_edges()}")
# Nodes: 3
# Edges: 3
```

This creates a triangle where all three people
are connected to each other.

---

### `Node`

**Definition.**
A node (or vertex) is a single entity in a graph.
It represents one "thing" in the network, such as
a person, a city, a web page, or a gene. Nodes can
carry attributes like labels, weights, or categories.

**Context.**
Nodes are the fundamental units you analyze. In a
social network, each person is a node. In a flight
network, each airport is a node. The number of nodes
determines the scale of your graph problem.

**Example.**
Adding nodes with attributes:

```python
import networkx as nx

G = nx.Graph()

G.add_node("London", population=9_000_000)
G.add_node("Paris", population=2_100_000)
G.add_node("Berlin", population=3_700_000)

# Access node attributes
print(G.nodes["London"]["population"])
# 9000000

# List all nodes
print(list(G.nodes))
# ['London', 'Paris', 'Berlin']
```

---

### `Edge`

**Definition.**
An edge (or link) is a connection between two nodes
in a graph. It represents a relationship, interaction,
or flow between the two entities. Edges can carry
attributes like weight, distance, or type.

**Context.**
Edges define the structure of your network. Without
edges, you just have a collection of isolated points.
The pattern of edges determines everything: who can
reach whom, where bottlenecks exist, and how
information spreads.

**Example.**
Adding weighted edges:

```python
import networkx as nx

G = nx.Graph()
G.add_edge("London", "Paris", distance=340)
G.add_edge("Paris", "Berlin", distance=878)
G.add_edge("London", "Berlin", distance=930)

# Access edge weight
d = G["London"]["Paris"]["distance"]
print(f"London-Paris: {d} km")

# List all edges with data
for u, v, data in G.edges(data=True):
    print(f"{u} -> {v}: {data}")
```

---

### `Directed Graph (DiGraph)`

**Definition.**
A directed graph (digraph) is a graph where edges
have a direction. Each edge goes from one node to
another, like an arrow. The connection from A to B
does not imply a connection from B to A.

**Context.**
Many real-world networks are directed. Twitter
follows are directed (you can follow someone who
does not follow you back). Web links are directed
(page A links to page B, but page B may not link
back). Citation networks are directed (paper A cites
paper B, not the reverse).

**Example.**
```python
import networkx as nx

# Undirected: friendship (mutual)
G = nx.Graph()
G.add_edge("Alice", "Bob")
# Alice-Bob and Bob-Alice are the same edge

# Directed: Twitter follow (one-way)
D = nx.DiGraph()
D.add_edge("Alice", "Bob")
# Alice follows Bob, but Bob does not
# necessarily follow Alice

print(D.has_edge("Alice", "Bob"))   # True
print(D.has_edge("Bob", "Alice"))   # False
```

---

### `Centrality`

**Definition.**
Centrality is a family of metrics that measure how
important or influential a node is within a graph.
Different centrality measures capture different
notions of importance: being well-connected, being
a bridge, or being close to everyone.

**Context.**
Centrality answers the question "which nodes matter
most?" In a disease network, high-centrality nodes
are super-spreaders. In an organization chart, they
are key decision-makers. Choosing the right centrality
measure depends on what "important" means in your
specific problem.

**Example.**
The most common centrality measures:

```python
import networkx as nx

G = nx.karate_club_graph()

# Degree centrality: how many connections
dc = nx.degree_centrality(G)

# Betweenness: how often on shortest paths
bc = nx.betweenness_centrality(G)

# Closeness: how close to all other nodes
cc = nx.closeness_centrality(G)

# PageRank: importance via link structure
pr = nx.pagerank(G)

# Find the most central node by degree
top = max(dc, key=dc.get)
print(f"Most connected: node {top}")
```

---

### `Betweenness Centrality`

**Definition.**
Betweenness centrality measures how often a node
appears on the shortest path between every pair of
other nodes. A node with high betweenness acts as a
bridge or gatekeeper, controlling the flow of
information through the network.

**Context.**
Betweenness identifies bottlenecks. In a
communication network, a high-betweenness node is
critical infrastructure: if it fails, many paths
break. In a social network, high-betweenness people
are brokers who connect different groups.

**Example.**
```python
import networkx as nx

G = nx.Graph()
G.add_edges_from([
    ("A", "B"), ("B", "C"), ("C", "D"),
    ("B", "D"), ("A", "E"), ("E", "B")
])

bc = nx.betweenness_centrality(G)

for node, score in sorted(
    bc.items(), key=lambda x: -x[1]
):
    print(f"  {node}: {score:.3f}")

# Node B likely has the highest betweenness
# because many shortest paths pass through it
```

Betweenness values range from 0 (never on any
shortest path) to 1 (on every shortest path).

---

### `PageRank`

**Definition.**
PageRank is an algorithm that ranks nodes by
importance based on the structure of incoming links.
A node is important if it is linked to by other
important nodes. It was originally invented by
Google to rank web pages.

**Context.**
PageRank is the foundation of web search ranking.
Beyond web search, it is used to rank academic
papers by citation importance, identify influential
users in social networks, and find key genes in
biological networks. It works on directed graphs.

**Example.**
```python
import networkx as nx

D = nx.DiGraph()
D.add_edges_from([
    ("A", "B"), ("A", "C"),
    ("B", "C"), ("C", "A"),
    ("D", "C")
])

pr = nx.pagerank(D, alpha=0.85)

for node, score in sorted(
    pr.items(), key=lambda x: -x[1]
):
    print(f"  {node}: {score:.4f}")

# Node C likely ranks highest because
# it has the most incoming links
```

The `alpha` parameter (default 0.85) is the
damping factor. It represents the probability
that a random web surfer continues clicking
links rather than jumping to a random page.

---

### `In-degree / Out-degree`

**Definition.**
In a directed graph, the in-degree of a node is the
number of edges pointing into it. The out-degree is
the number of edges pointing away from it. In an
undirected graph, there is just "degree" (the total
number of connections).

**Context.**
Degree is the simplest measure of node importance.
On Twitter, in-degree is your follower count and
out-degree is how many people you follow. On the
web, in-degree is how many pages link to yours.
High in-degree often means popularity or authority.

**Example.**
```python
import networkx as nx

D = nx.DiGraph()
D.add_edges_from([
    ("A", "B"), ("A", "C"),
    ("B", "C"), ("D", "C"),
    ("C", "A")
])

for node in D.nodes():
    i = D.in_degree(node)
    o = D.out_degree(node)
    print(f"  {node}: in={i}, out={o}")

# Node C has high in-degree (3 arrows in)
# Node A has high out-degree (2 arrows out)
```

You can also get degree distributions:

```python
in_degrees = [d for _, d in D.in_degree()]
print(f"Average in-degree: "
      f"{sum(in_degrees)/len(in_degrees):.1f}")
```

---

### `Clustering Coefficient`

**Definition.**
The clustering coefficient of a node measures how
many of its neighbors are also connected to each
other. It ranges from 0 (none of your friends know
each other) to 1 (all of your friends know each
other). It quantifies how "cliquish" a neighborhood
is.

**Context.**
High clustering means tight-knit communities. In
social networks, friend groups tend to have high
clustering (your friends are likely friends with
each other). Low clustering suggests a brokerage
role where a node bridges separate groups.

**Example.**
```python
import networkx as nx

G = nx.Graph()
G.add_edges_from([
    ("A", "B"), ("A", "C"), ("A", "D"),
    ("B", "C"),  # B and C know each other
    # D does not know B or C
])

# Node A's clustering coefficient
cc_a = nx.clustering(G, "A")
print(f"Clustering of A: {cc_a:.3f}")
# A has 3 neighbors (B, C, D)
# Only 1 pair (B-C) is connected
# out of 3 possible pairs
# So clustering = 1/3 = 0.333

# Average clustering for entire graph
avg = nx.average_clustering(G)
print(f"Average clustering: {avg:.3f}")
```

---

### `NetworkX`

**Definition.**
NetworkX is the standard Python library for creating,
manipulating, and analyzing graphs and networks. It
provides data structures for graphs and a large
collection of algorithms for network analysis.

**Context.**
NetworkX is the go-to tool for graph analytics in
Python. It is mature, well-documented, and integrates
with pandas, matplotlib, and scikit-learn. For
small-to-medium graphs (up to tens of thousands of
nodes), it is fast enough. For very large graphs,
tools like graph-tool or Neo4j may be needed.

**Example.**
Common NetworkX workflow:

```python
import networkx as nx
import matplotlib.pyplot as plt

# Load a built-in dataset
G = nx.karate_club_graph()

# Basic statistics
print(f"Nodes: {G.number_of_nodes()}")
print(f"Edges: {G.number_of_edges()}")
print(f"Density: {nx.density(G):.3f}")

# Analyze
pr = nx.pagerank(G)
top_node = max(pr, key=pr.get)
print(f"Most important: {top_node}")

# Visualize
nx.draw(G, with_labels=True,
        node_size=300, font_size=8)
plt.title("Karate Club Network")
plt.show()
```

Install with:

```bash
pip install networkx
```

---

### `Adjacency`

**Definition.**
Two nodes are adjacent if there is an edge directly
connecting them. The adjacency matrix is a square
table where row i, column j is 1 if node i connects
to node j, and 0 otherwise. It is the most basic
way to represent a graph numerically.

**Context.**
Adjacency is the foundation of graph computation.
Many graph algorithms operate on the adjacency
matrix or adjacency list internally. Understanding
adjacency helps you convert between graph
representations and interface with linear algebra
tools.

**Example.**
```python
import networkx as nx
import numpy as np

G = nx.Graph()
G.add_edges_from([
    ("A", "B"), ("B", "C"), ("A", "C")
])

# Adjacency matrix as numpy array
A = nx.to_numpy_array(G)
print(A)
# [[0. 1. 1.]
#  [1. 0. 1.]
#  [1. 1. 0.]]

# Adjacency list
for node in G.nodes():
    neighbors = list(G.neighbors(node))
    print(f"  {node}: {neighbors}")
# A: ['B', 'C']
# B: ['A', 'C']
# C: ['A', 'B']
```

---

### `Shortest Path`

**Definition.**
The shortest path between two nodes is the path
that traverses the fewest edges (or has the lowest
total weight, in a weighted graph). It is the most
efficient route from one node to another through
the network.

**Context.**
Shortest path algorithms are fundamental to graph
analytics. They power GPS navigation, network
routing, social network "degrees of separation"
calculations, and many centrality measures.
Dijkstra's algorithm is the most common method for
weighted graphs; BFS works for unweighted graphs.

**Example.**
```python
import networkx as nx

G = nx.Graph()
G.add_weighted_edges_from([
    ("A", "B", 1), ("B", "C", 2),
    ("A", "C", 10), ("C", "D", 1),
    ("B", "D", 5)
])

# Shortest path (by weight)
path = nx.shortest_path(
    G, "A", "D", weight="weight"
)
print(f"Path: {path}")
# Path: ['A', 'B', 'C', 'D']

# Path length
length = nx.shortest_path_length(
    G, "A", "D", weight="weight"
)
print(f"Distance: {length}")
# Distance: 4  (1 + 2 + 1)

# All shortest paths from A
all_paths = nx.shortest_path(G, "A")
for target, path in all_paths.items():
    print(f"  A -> {target}: {path}")
```

---

### `Connected Component`

**Definition.**
A connected component is a group of nodes where
every node can reach every other node through some
path. If a graph has multiple connected components,
it means some groups of nodes are completely
disconnected from others.

**Context.**
Connected components reveal the natural clusters
in your data. In a social network, separate
components mean groups with no connections between
them. In a supply chain, a disconnected component
means an isolated segment that cannot receive goods
from the rest of the network.

**Example.**
```python
import networkx as nx

G = nx.Graph()
# Group 1
G.add_edges_from([("A", "B"), ("B", "C")])
# Group 2 (disconnected from group 1)
G.add_edges_from([("X", "Y"), ("Y", "Z")])

# Find connected components
components = list(
    nx.connected_components(G)
)
print(f"Number of components: "
      f"{len(components)}")
# Number of components: 2

for i, comp in enumerate(components):
    print(f"  Component {i}: {comp}")
# Component 0: {'A', 'B', 'C'}
# Component 1: {'X', 'Y', 'Z'}

# Largest connected component
largest = max(components, key=len)
subgraph = G.subgraph(largest)
print(f"Largest has "
      f"{subgraph.number_of_nodes()} nodes")
```

For directed graphs, use
`nx.strongly_connected_components(D)`
or `nx.weakly_connected_components(D)`.

---

## See Also

- [Statistical Foundations](./01_statistical_foundations.md)
- [Optimization and Linear Programming](./08_optimization_and_linear_programming.md)
- [Scaling and Distributed Processing](./11_scaling_and_distributed_processing.md)
- [Federated and Distributed Learning](./13_federated_and_distributed_learning.md)

---

> **Author** — Simon Parris | Data Science Reference Library
