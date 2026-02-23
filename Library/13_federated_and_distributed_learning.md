# Federated and Distributed Learning

---

> **Field** — Distributed Machine Learning, Privacy-Preserving AI
> **Scope** — Federated learning, decentralized training,
> multi-agent reinforcement learning, trust aggregation,
> network topologies, and communication-efficient methods

---

## Overview

Federated and distributed learning train machine learning
models across multiple devices or organizations without
centralizing the raw data. Instead of sending all data to
one server, each participant trains locally and shares only
model updates. This preserves privacy, reduces bandwidth,
and enables collaboration between parties who cannot or
will not share their data directly.

---

## Definitions

### `Federated Learning`

**Definition.**
Federated learning is a machine learning approach where
a model is trained across multiple devices or servers
(called clients), each holding its own local data. The
raw data never leaves the client. Instead, each client
trains the model locally and sends only the model
updates (weights or gradients) to a central server,
which aggregates them into a global model.

**Context.**
Federated learning was popularized by Google for
training keyboard prediction models on phones without
collecting users' typing data. It is now used in
healthcare (hospitals sharing model updates without
sharing patient records), finance (banks collaborating
without sharing transaction data), and IoT (edge
devices learning together).

**Example.**
A simplified federated learning round:

```python
def federated_round(global_model, clients):
    """One round of federated learning."""
    local_updates = []

    for client in clients:
        # Client gets a copy of global model
        local_model = copy(global_model)

        # Client trains on its local data
        local_model.fit(
            client.X_train, client.y_train,
            epochs=5
        )

        # Client sends updated weights
        local_updates.append(
            local_model.get_weights()
        )

    # Server averages the updates
    avg_weights = average_weights(local_updates)
    global_model.set_weights(avg_weights)

    return global_model
```

Key property: the central server never sees
any client's raw data.

---

### `Decentralized Learning`

**Definition.**
Decentralized learning is a variant of distributed
learning where there is no central server at all.
Each participant communicates directly with its
neighbors, sharing and aggregating model updates
in a peer-to-peer fashion. The network collectively
converges to a shared model without any single point
of coordination.

**Context.**
Decentralized learning removes the single point of
failure and trust dependency on a central server. It
is useful when no single party can be trusted to
aggregate all updates, or when network topology makes
centralized communication impractical (edge computing,
sensor networks, blockchain-based systems).

**Example.**
```python
def decentralized_round(agents, topology):
    """One round of decentralized learning."""
    new_weights = {}

    for agent in agents:
        # Train locally
        agent.train_local(epochs=3)

        # Get neighbors from topology
        neighbors = topology.get_neighbors(
            agent.id
        )

        # Average own weights with neighbors
        all_weights = [agent.get_weights()]
        for neighbor_id in neighbors:
            all_weights.append(
                agents[neighbor_id].get_weights()
            )

        new_weights[agent.id] = average_weights(
            all_weights
        )

    # All agents update simultaneously
    for agent in agents:
        agent.set_weights(
            new_weights[agent.id]
        )
```

Decentralized learning requires more communication
rounds to converge than federated learning, but has
no single point of failure.

---

### `Non-IID Data`

**Definition.**
Non-IID (non-independent and identically distributed)
data means that data is not uniformly distributed
across participants. Different clients have different
kinds or proportions of data. For example, one hospital
might see mostly elderly patients while another sees
mostly children.

**Context.**
Non-IID data is the biggest challenge in federated
and distributed learning. When each client has a
biased view of the world, their local model updates
pull the global model in different directions. This
slows convergence and can reduce final accuracy.
Handling non-IID data is an active area of research.

**Example.**
```python
import numpy as np

def create_non_iid_split(X, y, n_clients,
                          alpha=0.5):
    """Split data non-IID using Dirichlet
    distribution.

    Lower alpha = more non-IID (more skewed).
    Higher alpha = more IID (more uniform).
    """
    n_classes = len(np.unique(y))
    client_data = {i: [] for i in range(n_clients)}

    for c in range(n_classes):
        class_idx = np.where(y == c)[0]

        # Dirichlet gives skewed proportions
        proportions = np.random.dirichlet(
            [alpha] * n_clients
        )
        # Split class data by proportions
        splits = np.split(
            class_idx,
            (np.cumsum(proportions)[:-1]
             * len(class_idx)).astype(int)
        )
        for i, split in enumerate(splits):
            client_data[i].extend(split)

    return client_data
```

With alpha = 0.1, some clients might have 90% of
one class and almost none of another.
With alpha = 100, all clients have roughly equal
class distributions (nearly IID).

---

### `MARL (Multi-Agent Reinforcement Learning)`

**Definition.**
MARL is a framework where multiple autonomous agents
learn to make decisions by interacting with an
environment and each other. Each agent has its own
observations, actions, and rewards. Agents can
cooperate, compete, or do both depending on the
problem design.

**Context.**
In distributed learning, MARL enables agents to
learn not just a model, but also how to communicate
and cooperate effectively. Each agent learns a policy
for when to share information, whom to share with,
and how much to trust received information. This is
more flexible than fixed aggregation rules.

**Example.**
```python
class MARLAgent:
    def __init__(self, agent_id, n_actions):
        self.id = agent_id
        self.q_table = {}  # state -> action values
        self.epsilon = 0.1  # exploration rate
        self.alpha = 0.1    # learning rate
        self.gamma = 0.9    # discount factor

    def choose_action(self, state):
        """Epsilon-greedy action selection."""
        if np.random.random() < self.epsilon:
            return np.random.randint(
                self.n_actions
            )
        q_values = self.q_table.get(
            state, np.zeros(self.n_actions)
        )
        return np.argmax(q_values)

    def update(self, state, action,
               reward, next_state):
        """Q-learning update."""
        current_q = self.q_table.get(
            state, np.zeros(self.n_actions)
        )
        next_q = self.q_table.get(
            next_state, np.zeros(self.n_actions)
        )
        current_q[action] += self.alpha * (
            reward
            + self.gamma * np.max(next_q)
            - current_q[action]
        )
        self.q_table[state] = current_q
```

---

### `Reward Function`

**Definition.**
A reward function assigns a numerical score to an
agent's action in a given state. Positive rewards
encourage the action; negative rewards (penalties)
discourage it. The agent learns to maximize its
cumulative reward over time.

**Context.**
Designing the reward function is the most critical
and difficult part of reinforcement learning. A
poorly designed reward can cause agents to find
loopholes (reward hacking) or ignore important
objectives. In distributed learning, the reward
typically includes model improvement, communication
cost, and energy usage.

**Example.**
```python
def compute_reward(
    f1_improvement,
    communication_cost,
    energy_used
):
    """Multi-objective reward for a
    distributed learning agent.

    Balances model quality against
    resource usage.
    """
    # Positive: model got better
    quality_reward = f1_improvement * 10.0

    # Negative: communication is expensive
    comm_penalty = -communication_cost * 0.5

    # Negative: energy is expensive
    energy_penalty = -energy_used * 0.3

    total = (quality_reward
             + comm_penalty
             + energy_penalty)

    return total
```

Example rewards for different actions:
- Agent shares with a useful peer and F1
  improves by 0.05: reward = +0.5 - costs
- Agent shares with everyone (wasteful):
  reward = +0.05 - high_costs (net negative)
- Agent does not share at all:
  reward = 0 - 0 = 0 (no improvement,
  no cost either)

---

### `Trust-weighted Aggregation`

**Definition.**
Trust-weighted aggregation is a method for combining
model updates from multiple agents where each
update is weighted by a trust score. Agents that
have provided useful updates in the past receive
higher trust and thus more influence on the combined
model. Agents that sent noise or harmful updates
receive lower trust.

**Context.**
In standard federated averaging, all clients have
equal weight. This is vulnerable to malicious or
unreliable participants. Trust-weighted aggregation
protects the global model by giving less influence
to agents whose contributions have been unhelpful.
Trust scores are updated dynamically based on
observed performance.

**Example.**
```python
import numpy as np

def trust_weighted_aggregate(
    updates, trust_scores
):
    """Aggregate model updates weighted
    by trust.

    updates: list of weight arrays
    trust_scores: list of floats [0, 1]
    """
    # Normalize trust scores to sum to 1
    total_trust = sum(trust_scores)
    if total_trust == 0:
        # Equal weighting as fallback
        weights = [1/len(updates)] * len(updates)
    else:
        weights = [
            t / total_trust
            for t in trust_scores
        ]

    # Weighted average
    aggregated = np.zeros_like(updates[0])
    for update, w in zip(updates, weights):
        aggregated += w * np.array(update)

    return aggregated

# Example usage
updates = [
    [0.1, 0.2, 0.3],  # Agent A (trustworthy)
    [0.1, 0.3, 0.2],  # Agent B (trustworthy)
    [9.0, 9.0, 9.0],  # Agent C (suspicious)
]
trust = [0.9, 0.85, 0.1]

result = trust_weighted_aggregate(
    updates, trust
)
# Agent C's extreme values have little
# influence due to low trust
```

---

### `Trust Score`

**Definition.**
A trust score is a numerical value (typically between
0 and 1) that represents how reliable or useful a
particular agent's contributions have been. It is
updated after each round of communication based on
whether the received information actually improved
the recipient's model.

**Context.**
Trust scores are essential for robust distributed
learning. Without them, a single malicious or faulty
agent can corrupt the entire system. Trust scores
adapt over time: new agents start with moderate trust,
and their scores increase or decrease based on track
record.

**Example.**
```python
class TrustTracker:
    def __init__(self, agent_ids,
                 initial_trust=0.5):
        self.trust = {
            aid: initial_trust
            for aid in agent_ids
        }
        self.decay = 0.95  # memory factor
        self.boost = 0.1   # reward increment

    def update(self, agent_id, was_helpful):
        """Update trust based on whether
        agent's contribution helped."""
        if was_helpful:
            self.trust[agent_id] = min(
                1.0,
                self.trust[agent_id] + self.boost
            )
        else:
            self.trust[agent_id] *= self.decay

    def get_trust(self, agent_id):
        return self.trust[agent_id]

# Usage
tracker = TrustTracker(["A", "B", "C"])
tracker.update("A", was_helpful=True)
tracker.update("C", was_helpful=False)
print(tracker.trust)
# {'A': 0.6, 'B': 0.5, 'C': 0.475}
```

---

### `Topology (network)`

**Definition.**
In distributed learning, topology refers to the
pattern of connections between agents. It determines
who can communicate with whom. Common topologies
include ring (each agent connects to two neighbors),
star (all agents connect to one central node), fully
connected (everyone connects to everyone), and random
(connections chosen randomly).

**Context.**
Topology profoundly affects learning speed and
communication cost. A fully connected topology allows
fast information spread but requires many connections.
A ring topology is communication-efficient but slow
to propagate updates across the network. The choice
of topology is a trade-off between convergence speed
and communication budget.

**Example.**
```python
import networkx as nx

def create_topology(agent_ids, topo_type):
    """Create a communication topology."""
    n = len(agent_ids)

    if topo_type == "ring":
        G = nx.cycle_graph(n)
    elif topo_type == "star":
        G = nx.star_graph(n - 1)
    elif topo_type == "fully_connected":
        G = nx.complete_graph(n)
    elif topo_type == "random":
        G = nx.erdos_renyi_graph(n, p=0.3)
    else:
        raise ValueError(f"Unknown: {topo_type}")

    # Map integer nodes to agent IDs
    mapping = dict(enumerate(agent_ids))
    G = nx.relabel_nodes(G, mapping)

    return G

agents = ["A", "B", "C", "D", "E"]
topo = create_topology(agents, "ring")
for a in agents:
    neighbors = list(topo.neighbors(a))
    print(f"  {a} -> {neighbors}")
```

---

### `Communication Budget`

**Definition.**
A communication budget is the limit on how much data
agents can exchange during training. It can be
measured in bytes, number of messages, or rounds of
communication. The budget forces agents to be
selective about what they share and with whom.

**Context.**
Communication is often the bottleneck in distributed
learning, not computation. Sending full model weights
across a network is expensive, especially on mobile
devices or over slow connections. A communication
budget forces efficient strategies: compressing
updates, sharing only with useful peers, or
communicating less frequently.

**Example.**
```python
class CommunicationBudget:
    def __init__(self, max_bytes_per_round):
        self.max_bytes = max_bytes_per_round
        self.used = 0

    def can_send(self, message_size):
        return (self.used + message_size
                <= self.max_bytes)

    def record_send(self, message_size):
        self.used += message_size

    def reset(self):
        """Call at start of each round."""
        self.used = 0

    def utilization(self):
        return self.used / self.max_bytes

# Usage
budget = CommunicationBudget(
    max_bytes_per_round=10_000
)

message_size = 2500  # bytes
if budget.can_send(message_size):
    send_to_peer(weights)
    budget.record_send(message_size)
    print(f"Used: {budget.utilization():.0%}")
```

---

### `Feature Importance Vector`

**Definition.**
A feature importance vector is an array where each
element represents how important a particular input
feature is to the model's predictions. Higher values
mean the feature contributes more to the output.
In distributed learning, agents can share feature
importance vectors instead of raw data.

**Context.**
Sharing feature importance is a lightweight
alternative to sharing full model weights. It tells
other agents which features are most predictive in
your local data without revealing the data itself.
This is especially useful when agents have different
feature distributions due to non-IID data.

**Example.**
```python
from sklearn.ensemble import (
    GradientBoostingClassifier
)
import numpy as np

def get_feature_importance(model, feature_names):
    """Extract feature importance vector."""
    importances = model.feature_importances_
    return dict(zip(feature_names, importances))

# Agent A trains on local data
model_a = GradientBoostingClassifier()
model_a.fit(X_local_a, y_local_a)

features = ["age", "income", "location",
            "tenure", "usage"]

importance_a = get_feature_importance(
    model_a, features
)
print("Agent A's feature importance:")
for feat, imp in sorted(
    importance_a.items(),
    key=lambda x: -x[1]
):
    print(f"  {feat}: {imp:.3f}")

# Share this vector (not the raw data)
# with peers for collaborative learning
```

---

### `Dirichlet Partitioning`

**Definition.**
Dirichlet partitioning uses the Dirichlet distribution
to split a dataset across multiple clients with
controlled non-IID-ness. A single parameter (alpha)
controls how skewed the split is. Low alpha creates
highly non-IID splits; high alpha creates nearly
uniform splits.

**Context.**
Dirichlet partitioning is the standard benchmark
method for simulating non-IID data in federated
learning research. It gives you precise control over
the degree of data heterogeneity, which lets you test
how robust your algorithm is to different levels of
non-IID-ness.

**Example.**
```python
import numpy as np

def dirichlet_partition(
    labels, n_clients, alpha
):
    """Partition data indices using Dirichlet.

    alpha=0.1 : extreme non-IID
    alpha=1.0 : moderate non-IID
    alpha=100 : nearly IID
    """
    n_classes = len(np.unique(labels))
    client_indices = [[] for _ in range(n_clients)]

    for c in range(n_classes):
        class_idx = np.where(labels == c)[0]
        np.random.shuffle(class_idx)

        proportions = np.random.dirichlet(
            [alpha] * n_clients
        )
        proportions = (
            proportions / proportions.sum()
        )
        split_points = (
            np.cumsum(proportions) * len(class_idx)
        ).astype(int)[:-1]

        splits = np.split(class_idx, split_points)
        for i, split in enumerate(splits):
            client_indices[i].extend(split.tolist())

    return client_indices

# Example
labels = np.array([0]*100 + [1]*100 + [2]*100)
partitions = dirichlet_partition(
    labels, n_clients=5, alpha=0.5
)
for i, p in enumerate(partitions):
    class_counts = np.bincount(
        labels[p], minlength=3
    )
    print(f"  Client {i}: {class_counts}")
```

---

### `Epsilon-Greedy`

**Definition.**
Epsilon-greedy is a simple strategy for balancing
exploration (trying new actions) and exploitation
(choosing the best-known action). With probability
epsilon, the agent picks a random action. With
probability (1 - epsilon), it picks the action with
the highest estimated reward.

**Context.**
In distributed learning, epsilon-greedy helps agents
decide whether to communicate with their current best
peer (exploitation) or try a new peer who might be
even better (exploration). Without exploration, agents
get stuck with suboptimal partners. Without
exploitation, they waste budget on random choices.

**Example.**
```python
import numpy as np

class EpsilonGreedy:
    def __init__(self, n_actions, epsilon=0.1):
        self.n_actions = n_actions
        self.epsilon = epsilon
        self.q_values = np.zeros(n_actions)
        self.counts = np.zeros(n_actions)

    def select_action(self):
        if np.random.random() < self.epsilon:
            # Explore: random action
            return np.random.randint(
                self.n_actions
            )
        else:
            # Exploit: best known action
            return np.argmax(self.q_values)

    def update(self, action, reward):
        self.counts[action] += 1
        n = self.counts[action]
        # Incremental mean update
        self.q_values[action] += (
            (reward - self.q_values[action]) / n
        )

# Agent choosing which peer to share with
peers = ["Agent_A", "Agent_B", "Agent_C"]
selector = EpsilonGreedy(
    n_actions=len(peers), epsilon=0.15
)

# Over many rounds, the agent learns which
# peer is most beneficial
for round_num in range(100):
    peer_idx = selector.select_action()
    reward = share_with(peers[peer_idx])
    selector.update(peer_idx, reward)
```

---

### `Peer-to-Peer Exchange`

**Definition.**
Peer-to-peer exchange is direct communication between
two agents without going through a central server.
Each agent sends its model updates, feature importance
vectors, or other information directly to selected
peers, and receives information in return.

**Context.**
Peer-to-peer exchange is the foundation of
decentralized learning. It eliminates the need for a
trusted central server and distributes the
communication load across the network. The challenge
is deciding which peers to exchange with and how to
handle unreliable or slow connections.

**Example.**
```python
class PeerExchange:
    def __init__(self, agent_id, topology):
        self.id = agent_id
        self.topology = topology
        self.inbox = []

    def send_to_peer(self, peer_id, payload):
        """Send data to a specific peer."""
        message = {
            "from": self.id,
            "to": peer_id,
            "payload": payload,
            "size_bytes": len(str(payload))
        }
        # In simulation, append to peer inbox
        return message

    def receive_from_peers(self):
        """Collect all received messages."""
        messages = self.inbox.copy()
        self.inbox.clear()
        return messages

    def get_available_peers(self):
        """Return peers in current topology."""
        return list(
            self.topology.neighbors(self.id)
        )

# Usage
exchange = PeerExchange("Agent_0", topology)
peers = exchange.get_available_peers()
print(f"Available peers: {peers}")

for peer in peers[:2]:  # share with top 2
    exchange.send_to_peer(
        peer, {"weights": local_weights}
    )
```

---

### `Neighbor Selection`

**Definition.**
Neighbor selection is the process by which an agent
chooses which peers to communicate with during a
training round. Instead of communicating with all
available peers (expensive) or random peers
(inefficient), the agent selects the peers most
likely to provide useful information.

**Context.**
Smart neighbor selection dramatically improves the
efficiency of distributed learning. An agent with
data skewed toward class A benefits most from peers
who have data from classes B and C. Reinforcement
learning can be used to learn optimal neighbor
selection policies over time.

**Example.**
```python
import numpy as np

def select_neighbors(
    agent_id, all_agents, trust_scores,
    budget, strategy="top_k"
):
    """Select which peers to share with."""
    candidates = [
        a for a in all_agents
        if a != agent_id
    ]

    if strategy == "top_k":
        # Pick the most trusted peers
        scored = [
            (a, trust_scores.get(a, 0.5))
            for a in candidates
        ]
        scored.sort(key=lambda x: -x[1])
        return [a for a, _ in scored[:budget]]

    elif strategy == "diverse":
        # Pick peers with different data
        # distributions for complementarity
        selected = []
        for _ in range(budget):
            if not candidates:
                break
            choice = candidates.pop(
                np.random.randint(len(candidates))
            )
            selected.append(choice)
        return selected

    elif strategy == "epsilon_greedy":
        # Mostly exploit top peers,
        # sometimes explore new ones
        if np.random.random() < 0.1:
            np.random.shuffle(candidates)
            return candidates[:budget]
        else:
            scored = [
                (a, trust_scores.get(a, 0.5))
                for a in candidates
            ]
            scored.sort(key=lambda x: -x[1])
            return [a for a, _ in scored[:budget]]

# Usage
selected = select_neighbors(
    "Agent_0", agent_list, trust,
    budget=3, strategy="top_k"
)
print(f"Selected peers: {selected}")
```

---

### `Ring Topology`

**Definition.**
A ring topology arranges agents in a circle where
each agent connects only to its two immediate
neighbors (one on each side). Information travels
around the ring, passing from neighbor to neighbor.

**Context.**
Ring topologies are communication-efficient because
each agent only maintains two connections. However,
information takes many hops to travel across the
network (up to N/2 hops for N agents), which slows
convergence. Ring topologies are a good baseline for
evaluating more complex topologies.

**Example.**
```
    A --- B
   /       \
  E         C
   \       /
    D ----
```

```python
import networkx as nx

agents = ["A", "B", "C", "D", "E"]
ring = nx.cycle_graph(len(agents))
ring = nx.relabel_nodes(
    ring, dict(enumerate(agents))
)

for a in agents:
    nbrs = list(ring.neighbors(a))
    print(f"  {a} -> {nbrs}")
# A -> ['E', 'B']
# B -> ['A', 'C']
# C -> ['B', 'D']
# D -> ['C', 'E']
# E -> ['D', 'A']
```

Each agent has exactly degree 2.
Total edges: N (same as number of agents).

---

### `Star Topology`

**Definition.**
A star topology has one central hub node connected
to all other agents. All communication goes through
the hub. Peripheral agents do not connect directly
to each other.

**Context.**
Star topologies are fast for information
dissemination (one hop from hub to any agent) but
create a bottleneck and single point of failure at
the hub. This is essentially the topology used in
standard federated learning, where the central server
is the hub.

**Example.**
```
      B
      |
  E - A - C
      |
      D
```

```python
import networkx as nx

agents = ["A", "B", "C", "D", "E"]
# Star with A as center
star = nx.star_graph(len(agents) - 1)
mapping = dict(enumerate(agents))
star = nx.relabel_nodes(star, mapping)

for a in agents:
    nbrs = list(star.neighbors(a))
    print(f"  {a} -> {nbrs}")
# A -> ['B', 'C', 'D', 'E']
# B -> ['A']
# C -> ['A']
# D -> ['A']
# E -> ['A']
```

Hub has degree (N-1).
Peripheral nodes have degree 1.
Total edges: N-1.

---

### `Fully Connected Topology`

**Definition.**
A fully connected topology connects every agent to
every other agent. Each agent can communicate
directly with any other agent in a single hop. There
are no intermediaries needed.

**Context.**
Fully connected topologies offer the fastest
information spread and best convergence properties.
However, the number of connections grows quadratically
with the number of agents (N * (N-1) / 2 edges),
making them impractical for large networks. They serve
as an upper-bound benchmark for topology performance.

**Example.**
```
    A ------- B
    | \     / |
    |   \ /   |
    |   / \   |
    | /     \ |
    D ------- C
```

```python
import networkx as nx

agents = ["A", "B", "C", "D"]
fc = nx.complete_graph(len(agents))
mapping = dict(enumerate(agents))
fc = nx.relabel_nodes(fc, mapping)

print(f"Edges: {fc.number_of_edges()}")
# Edges: 6  (4 * 3 / 2)

for a in agents:
    nbrs = list(fc.neighbors(a))
    print(f"  {a} -> {nbrs}")
# A -> ['B', 'C', 'D']
# B -> ['A', 'C', 'D']
# C -> ['A', 'B', 'D']
# D -> ['A', 'B', 'C']
```

Every agent has degree (N-1).
Total edges: N * (N-1) / 2.
For 10 agents: 45 edges.
For 100 agents: 4,950 edges.

---

### `Random Topology`

**Definition.**
A random topology connects agents with a specified
probability. Each possible edge is independently
included with probability p. This creates irregular
networks where some agents have many connections and
others have few.

**Context.**
Random topologies model realistic, imperfect networks
where connections form organically. They are used
in research to test algorithm robustness: a good
distributed learning algorithm should work across
different random network structures, not just ideal
topologies.

**Example.**
```python
import networkx as nx

agents = ["A", "B", "C", "D", "E",
          "F", "G", "H"]

# Each edge exists with probability 0.3
random_graph = nx.erdos_renyi_graph(
    len(agents), p=0.3, seed=42
)
mapping = dict(enumerate(agents))
random_graph = nx.relabel_nodes(
    random_graph, mapping
)

print(f"Edges: {random_graph.number_of_edges()}")

for a in agents:
    nbrs = list(random_graph.neighbors(a))
    degree = random_graph.degree(a)
    print(f"  {a} (degree {degree}) -> {nbrs}")

# Check if connected
connected = nx.is_connected(random_graph)
print(f"Connected: {connected}")
# May or may not be connected depending
# on the random seed and probability p
```

Higher p = denser graph, more like fully connected.
Lower p = sparser graph, more likely disconnected.

---

### `Bandwidth Simulation`

**Definition.**
Bandwidth simulation models the speed of network
links between agents. Different pairs of agents may
have different available bandwidth (measured in bytes
or bits per second), affecting how quickly they can
exchange model updates.

**Context.**
In real distributed systems, network bandwidth varies
dramatically. Two machines in the same data center
might share data at 10 Gbps, while a mobile device on
a cellular connection might manage only 1 Mbps.
Simulating bandwidth realistically helps evaluate
whether a distributed learning algorithm is practical
in real-world deployments.

**Example.**
```python
import numpy as np

class BandwidthSimulator:
    def __init__(self, agents, seed=42):
        np.random.seed(seed)
        self.bandwidth = {}
        for i in agents:
            for j in agents:
                if i != j:
                    # Random bandwidth in Mbps
                    bw = np.random.uniform(
                        1, 100
                    )
                    self.bandwidth[(i, j)] = bw

    def transfer_time(self, src, dst,
                      size_bytes):
        """Time to transfer data in seconds."""
        bw_mbps = self.bandwidth[(src, dst)]
        bw_bytes = bw_mbps * 1_000_000 / 8
        return size_bytes / bw_bytes

# Usage
sim = BandwidthSimulator(["A", "B", "C"])
time = sim.transfer_time(
    "A", "B", size_bytes=1_000_000
)
print(f"Transfer time: {time:.3f}s")
```

---

### `Latency Simulation`

**Definition.**
Latency simulation models the delay before data
transmission begins between two agents. Latency is
the time it takes for the first bit to arrive,
separate from the time needed to transfer the full
message (which depends on bandwidth).

**Context.**
Latency matters in distributed learning because agents
must synchronize: if one agent is slow to respond,
everyone waits. High latency can make synchronous
algorithms impractical. Simulating latency helps you
choose between synchronous (wait for all agents) and
asynchronous (proceed without waiting) algorithms.

**Example.**
```python
import numpy as np
import time

class LatencySimulator:
    def __init__(self, base_ms=10,
                 variance_ms=5):
        self.base = base_ms
        self.variance = variance_ms

    def simulate(self, src, dst):
        """Return simulated latency in ms."""
        jitter = np.random.exponential(
            self.variance
        )
        return self.base + jitter

    def wait(self, src, dst):
        """Actually delay execution."""
        delay_ms = self.simulate(src, dst)
        time.sleep(delay_ms / 1000)
        return delay_ms

# Usage
lat = LatencySimulator(
    base_ms=20, variance_ms=10
)
delay = lat.simulate("Agent_0", "Agent_1")
print(f"Latency: {delay:.1f} ms")
```

Typical latency ranges:
- Same machine: < 1 ms
- Same data center: 1-5 ms
- Same region: 10-50 ms
- Cross-continent: 100-300 ms
- Mobile network: 50-500 ms

---

### `Packet Loss`

**Definition.**
Packet loss is the failure of transmitted data to
arrive at its destination. In network simulation, it
is modeled as a probability that any given message
will be lost entirely or arrive corrupted.

**Context.**
Real networks drop packets, especially wireless and
mobile networks. A robust distributed learning
algorithm must handle missing updates gracefully.
Packet loss simulation tests whether your algorithm
degrades gracefully or fails catastrophically when
messages disappear.

**Example.**
```python
import numpy as np

class PacketLossSimulator:
    def __init__(self, loss_rate=0.05):
        """loss_rate: probability of losing
        a message (0.05 = 5%)."""
        self.loss_rate = loss_rate

    def attempt_send(self, message):
        """Return message if successful,
        None if lost."""
        if np.random.random() < self.loss_rate:
            return None  # packet lost
        return message

    def send_with_retry(self, message,
                        max_retries=3):
        """Retry on failure."""
        for attempt in range(max_retries):
            result = self.attempt_send(message)
            if result is not None:
                return result, attempt + 1
        return None, max_retries  # all failed

# Usage
sim = PacketLossSimulator(loss_rate=0.1)

successes = 0
for _ in range(1000):
    result = sim.attempt_send("weights")
    if result is not None:
        successes += 1

print(f"Success rate: {successes/1000:.1%}")
# Approximately 90%
```

---

### `Energy Cost Simulation`

**Definition.**
Energy cost simulation models the power consumed by
agents during computation and communication. Each
training step and each message sent costs a certain
amount of energy. This is particularly relevant for
mobile and IoT devices with limited battery.

**Context.**
In federated learning on mobile devices, energy
efficiency determines whether users will tolerate
the learning process. A phone that drains its battery
for ML training will be rejected by users.
Energy-aware algorithms minimize computation and
communication to preserve battery life.

**Example.**
```python
class EnergySimulator:
    def __init__(self):
        # Joules per operation
        self.compute_per_epoch = 0.5
        self.send_per_mb = 0.3
        self.receive_per_mb = 0.1
        self.idle_per_second = 0.01

    def training_cost(self, epochs):
        return epochs * self.compute_per_epoch

    def communication_cost(self, size_mb,
                           is_sending=True):
        if is_sending:
            return size_mb * self.send_per_mb
        return size_mb * self.receive_per_mb

    def total_round_cost(self, epochs,
                         send_mb, recv_mb):
        return (
            self.training_cost(epochs)
            + self.communication_cost(
                send_mb, True)
            + self.communication_cost(
                recv_mb, False)
        )

# Usage
energy = EnergySimulator()
cost = energy.total_round_cost(
    epochs=5, send_mb=2.0, recv_mb=2.0
)
print(f"Round energy cost: {cost:.2f} J")
```

---

### `F1 Improvement`

**Definition.**
F1 improvement is the change in F1 score (a measure
of classification accuracy that balances precision and
recall) between two time points. In distributed
learning, it measures how much a round of
communication actually helped the model perform
better.

**Context.**
F1 improvement is the key metric for evaluating
whether communication was worthwhile. If an agent
spends energy and bandwidth sharing weights with a
peer but sees no F1 improvement, that communication
was wasted. Tracking F1 improvement per peer helps
agents learn which partners are useful.

**Example.**
```python
from sklearn.metrics import f1_score

def measure_improvement(
    model, X_val, y_val,
    old_weights, new_weights
):
    """Measure F1 improvement from a
    weight update."""
    # Score with old weights
    model.set_weights(old_weights)
    y_pred_old = model.predict(X_val)
    f1_old = f1_score(
        y_val, y_pred_old, average="weighted"
    )

    # Score with new weights
    model.set_weights(new_weights)
    y_pred_new = model.predict(X_val)
    f1_new = f1_score(
        y_val, y_pred_new, average="weighted"
    )

    improvement = f1_new - f1_old
    print(f"F1: {f1_old:.4f} -> {f1_new:.4f} "
          f"(delta: {improvement:+.4f})")

    return improvement
```

Interpretation:
- Positive improvement: communication helped
- Zero improvement: communication was neutral
- Negative improvement: communication hurt
  (possibly due to a bad peer or data mismatch)

---

### `Communication Penalty`

**Definition.**
A communication penalty is a negative term in the
reward function that discourages excessive messaging.
Every message sent costs something (bandwidth, time,
energy), so the penalty ensures agents communicate
only when the expected benefit outweighs the cost.

**Context.**
Without a communication penalty, agents would share
with everyone all the time. The penalty forces them
to be strategic: share only with the peers most
likely to provide useful information, and only when
the model is likely to benefit. This is critical for
resource-constrained environments.

**Example.**
```python
def reward_with_comm_penalty(
    f1_improvement,
    messages_sent,
    bytes_sent,
    penalty_per_message=0.05,
    penalty_per_mb=0.02
):
    """Compute reward with communication
    penalty."""
    quality = f1_improvement * 10.0

    msg_penalty = (
        messages_sent * penalty_per_message
    )
    size_penalty = (
        (bytes_sent / 1_000_000)
        * penalty_per_mb
    )

    total_penalty = msg_penalty + size_penalty

    reward = quality - total_penalty
    return reward

# Good: big improvement, little communication
r1 = reward_with_comm_penalty(
    f1_improvement=0.05,
    messages_sent=1,
    bytes_sent=500_000
)
print(f"Efficient sharing: {r1:.3f}")

# Bad: small improvement, lots of communication
r2 = reward_with_comm_penalty(
    f1_improvement=0.01,
    messages_sent=5,
    bytes_sent=5_000_000
)
print(f"Wasteful sharing: {r2:.3f}")
```

---

### `Energy Penalty`

**Definition.**
An energy penalty is a negative reward component
that accounts for the energy consumed during training
and communication. It discourages agents from
performing unnecessary computation or communication
that drains batteries or increases costs.

**Context.**
Energy penalties are particularly important for
mobile and IoT federated learning. A phone running
federated training overnight should not drain the
battery. The energy penalty in the reward function
teaches the agent to find energy-efficient strategies
that still improve the model.

**Example.**
```python
def compute_energy_penalty(
    training_epochs,
    messages_sent,
    bytes_transmitted,
    compute_cost_per_epoch=0.5,
    comm_cost_per_mb=0.3,
    penalty_weight=0.3
):
    """Compute energy penalty for
    reward function."""
    compute_energy = (
        training_epochs * compute_cost_per_epoch
    )
    comm_energy = (
        (bytes_transmitted / 1_000_000)
        * comm_cost_per_mb
    )
    total_energy = compute_energy + comm_energy

    penalty = total_energy * penalty_weight
    return penalty

# Light round (2 epochs, 1 message)
p1 = compute_energy_penalty(
    training_epochs=2,
    messages_sent=1,
    bytes_transmitted=500_000
)
print(f"Light penalty: {p1:.3f}")

# Heavy round (10 epochs, 5 messages)
p2 = compute_energy_penalty(
    training_epochs=10,
    messages_sent=5,
    bytes_transmitted=5_000_000
)
print(f"Heavy penalty: {p2:.3f}")
```

---

### `Baseline (comparison)`

**Definition.**
A baseline is a simple, well-understood approach that
you compare your new method against. It establishes a
minimum standard of performance. If your sophisticated
algorithm cannot beat the baseline, it is not worth
the added complexity.

**Context.**
In distributed learning research, common baselines
include: local-only training (each agent trains alone,
no communication), centralized training (all data on
one machine, upper bound), and FedAvg (standard
federated averaging). Always report baseline results
alongside your method's results.

**Example.**
```python
def run_baselines(agents, X_test, y_test):
    """Run standard baselines for comparison."""
    results = {}

    # Baseline 1: Local only (no sharing)
    local_scores = []
    for agent in agents:
        agent.train_local_only(epochs=50)
        score = agent.evaluate(X_test, y_test)
        local_scores.append(score)
    results["local_only"] = {
        "mean_f1": np.mean(local_scores),
        "std_f1": np.std(local_scores)
    }

    # Baseline 2: Centralized (upper bound)
    all_X = np.concatenate(
        [a.X_train for a in agents]
    )
    all_y = np.concatenate(
        [a.y_train for a in agents]
    )
    central_model = train_model(all_X, all_y)
    results["centralized"] = {
        "f1": evaluate(central_model,
                       X_test, y_test)
    }

    # Baseline 3: FedAvg
    fedavg_model = run_fedavg(agents, rounds=20)
    results["fedavg"] = {
        "f1": evaluate(fedavg_model,
                       X_test, y_test)
    }

    return results
```

---

### `Local Training`

**Definition.**
Local training is the process of training a model
using only the data available on a single agent or
device. The model sees no data from other participants.
It is the simplest approach but misses the benefit
of learning from the broader population.

**Context.**
Local training serves as both a baseline and a
component of federated learning. In each federated
round, agents first perform local training on their
own data, then share the results. The number of local
training epochs per round is a key hyperparameter:
too few and convergence is slow; too many and models
diverge (client drift).

**Example.**
```python
def local_training(agent, epochs=5,
                   batch_size=32):
    """Train model on local data only."""
    model = agent.model
    X, y = agent.X_train, agent.y_train

    for epoch in range(epochs):
        # Shuffle data
        indices = np.random.permutation(len(X))

        # Mini-batch training
        for start in range(
            0, len(X), batch_size
        ):
            batch_idx = indices[
                start:start + batch_size
            ]
            X_batch = X[batch_idx]
            y_batch = y[batch_idx]
            model.partial_fit(
                X_batch, y_batch
            )

    # Evaluate on local validation set
    score = agent.evaluate_local()
    print(f"  Agent {agent.id}: "
          f"local F1 = {score:.4f}")
    return score
```

---

### `Feature Weight Mapping`

**Definition.**
Feature weight mapping translates the importance or
weight of features from one agent's model to another's
representation. When agents have different feature
sets or different feature orderings, mapping ensures
that shared information aligns correctly.

**Context.**
In heterogeneous distributed learning, agents may
have different features available (one hospital
measures blood pressure, another measures cholesterol).
Feature weight mapping allows partial knowledge
transfer even when feature sets do not perfectly
overlap. This is more flexible than requiring all
agents to use identical features.

**Example.**
```python
def map_feature_weights(
    source_features, source_weights,
    target_features
):
    """Map feature weights from source agent
    to target agent's feature space."""

    # Create mapping dictionary
    source_map = dict(
        zip(source_features, source_weights)
    )

    # Map to target features
    mapped_weights = []
    for feat in target_features:
        if feat in source_map:
            mapped_weights.append(
                source_map[feat]
            )
        else:
            # Feature not in source; use 0
            mapped_weights.append(0.0)

    return mapped_weights

# Agent A has features [age, income, score]
# Agent B has features [age, tenure, score]
source_feats = ["age", "income", "score"]
source_wts = [0.3, 0.5, 0.2]

target_feats = ["age", "tenure", "score"]

mapped = map_feature_weights(
    source_feats, source_wts, target_feats
)
print(f"Mapped weights: {mapped}")
# [0.3, 0.0, 0.2]
# 'tenure' gets 0 because source
# does not have it
```

---

## See Also

- [Graph Analytics and Networks](./09_graph_analytics_and_networks.md)
- [Scaling and Distributed Processing](./11_scaling_and_distributed_processing.md)
- [Anomaly Detection and Operational ML](./12_anomaly_detection_and_operational_ml.md)
- [Reproducibility and Governance](./14_reproducibility_and_governance.md)

---

> **Author** — Simon Parris | Data Science Reference Library
