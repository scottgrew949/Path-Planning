# python/replay_buffer.py
# Experience replay buffer for DQN training.
#
# CONCEPT — What is experience replay?
#   During training the agent generates a stream of (state, action, reward,
#   next_state, done) tuples — one per step. Naively training on each tuple
#   the moment it's generated has two problems:
#     1. Consecutive steps are highly correlated — the net overfits to recent
#        experience and forgets earlier exploration.
#     2. The data distribution shifts as the policy improves — unstable training.
#
#   Experience replay fixes both: store every tuple in a buffer, then sample
#   random BATCHES for training. Random sampling breaks correlation.
#   The buffer holds old AND new experience — diverse, stable distribution.
#
# CONCEPT — Circular buffer
#   The buffer has a fixed max size. When full, new experiences overwrite the
#   oldest ones (circular/ring buffer). This keeps memory bounded while
#   naturally discarding stale experience from an early, worse policy.
#
# Self-driving analog:
#   The replay buffer is the car's dashcam footage archive. Training on random
#   clips from the last 10,000 miles is more stable than only training on
#   what happened in the last 10 seconds.

import random
from collections import deque
import numpy as np
import torch


def _stack_tensors(tensors):
    return torch.cat([t.view(1, -1) if t.dim() == 1 else t for t in tensors])


class ReplayBuffer:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action: int, reward: float, next_state, done: bool):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int):
        # CONCEPT — Why FloatTensor vs LongTensor?
        #   Neural net inputs and rewards are floats — continuous values.
        #   Actions are indices (0-3) — integers used to index into Q-value output.
        #   PyTorch requires the correct dtype for each operation.
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            _stack_tensors(states),
            torch.LongTensor(actions),
            torch.FloatTensor(rewards),
            _stack_tensors(next_states),
            torch.tensor(dones, dtype=torch.float32),
        )

    def __len__(self) -> int:
        return len(self.buffer)

    def is_ready(self, batch_size: int) -> bool:
        # Training should not start until the buffer has enough samples
        # to fill at least one batch — otherwise sample() would fail.
        return len(self) >= batch_size


# ---- SumTree ----------------------------------------------------------------
#
# CONCEPT — Why a sum tree?
#   Prioritized replay needs to: (1) sample experiences proportional to their
#   priority and (2) update individual priorities in O(log n). A flat array
#   supports (2) in O(1) but (1) in O(n) — too slow for 10k-item buffers.
#   A SumTree solves both in O(log n):
#     - Leaves store individual priorities.
#     - Internal nodes store the sum of their subtree.
#     - Root = total priority sum — used to sample a random cumulative value.
#     - To sample: pick a random value in [0, root]. Traverse left if the
#       value fits in the left subtree, else subtract left and go right.
#       This finds the leaf in O(log n).
#
# Memory layout: array of size 2*capacity - 1.
#   Indices 0 to capacity-2: internal nodes.
#   Indices capacity-1 to 2*capacity-2: leaves (one per experience slot).

class SumTree:

    def __init__(self, capacity: int):
        self.capacity    = capacity
        self.tree        = np.zeros(2 * capacity - 1, dtype=np.float64)
        self.data        = [None] * capacity
        self.write_index = 0
        self.count       = 0

    @property
    def total(self) -> float:
        return float(self.tree[0])

    def _leaf_index(self, data_index: int) -> int:
        return data_index + self.capacity - 1

    def _propagate(self, node_index: int, delta: float) -> None:
        parent = (node_index - 1) // 2
        self.tree[parent] += delta
        if parent != 0:
            self._propagate(parent, delta)

    def update(self, data_index: int, priority: float) -> None:
        leaf  = self._leaf_index(data_index)
        delta = priority - self.tree[leaf]
        self.tree[leaf] = priority
        self._propagate(leaf, delta)

    def add(self, priority: float, data) -> None:
        self.data[self.write_index] = data
        self.update(self.write_index, priority)
        self.write_index = (self.write_index + 1) % self.capacity
        self.count       = min(self.count + 1, self.capacity)

    def get(self, cumulative_sum: float):
        # Traverse from root to the leaf whose priority range contains cumulative_sum.
        node_index = 0
        while node_index < self.capacity - 1:
            left  = 2 * node_index + 1
            right = left + 1
            if cumulative_sum <= self.tree[left]:
                node_index = left
            else:
                cumulative_sum -= self.tree[left]
                node_index      = right
        data_index = node_index - (self.capacity - 1)
        return data_index, float(self.tree[node_index]), self.data[data_index]


# ---- PrioritizedReplayBuffer ------------------------------------------------
#
# CONCEPT — Priority = |TD error|
#   TD error = |target Q - predicted Q| measures surprise: how wrong was the
#   network on this transition? High surprise = more to learn. PER samples
#   high-TD-error transitions more often — same intuition as "study what you
#   got wrong, not what you already know."
#
# CONCEPT — Importance sampling correction (beta)
#   Non-uniform sampling introduces bias: if high-priority transitions appear
#   more, gradients are biased toward them. Importance sampling weights
#   w_i = (1 / (N * P(i)))^beta correct this. beta=0 = no correction,
#   beta=1 = full unbiased correction. Anneal beta from 0.4 → 1.0 over
#   training so early training can exploit priorities freely.
#
# CONCEPT — alpha controls prioritization strength
#   P(i) = priority_i^alpha / sum(priority^alpha)
#   alpha=0 → uniform sampling (standard replay). alpha=1 → full priority.
#   alpha=0.6 is the standard default.

class PrioritizedReplayBuffer:

    def __init__(self, capacity: int, alpha: float = 0.6, epsilon: float = 1e-5):
        self.tree        = SumTree(capacity)
        self.alpha       = alpha
        self.epsilon     = epsilon  # floor priority so nothing has P=0
        self.max_priority = 1.0     # new experiences start with max priority

    def push(self, *experience) -> None:
        self.tree.add(self.max_priority ** self.alpha, experience)

    def sample(self, batch_size: int, beta: float = 0.4):
        indices    = []
        priorities = []
        batch      = []
        segment    = self.tree.total / batch_size

        for i in range(batch_size):
            cumulative         = random.uniform(segment * i, segment * (i + 1))
            data_index, priority, experience = self.tree.get(cumulative)
            if experience is None:
                # Buffer not full yet — resample from beginning
                data_index, priority, experience = self.tree.get(random.uniform(0, self.tree.total))
            indices.append(data_index)
            priorities.append(priority)
            batch.append(experience)

        # Importance sampling weights — normalised so max weight = 1.
        total       = self.tree.total
        min_prob    = min(p for p in priorities) / total if total > 0 else 1.0
        max_weight  = (min_prob * len(self)) ** (-beta) if len(self) > 0 else 1.0
        weights     = [((p / total) * len(self)) ** (-beta) / max_weight
                       for p in priorities]

        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            _stack_tensors(states),
            torch.LongTensor(actions),
            torch.FloatTensor(rewards),
            _stack_tensors(next_states),
            torch.tensor(dones, dtype=torch.float32),
            indices,
            torch.FloatTensor(weights),
        )

    def update_priorities(self, indices: list, td_errors) -> None:
        for data_index, td_error in zip(indices, td_errors):
            priority = (abs(float(td_error)) + self.epsilon) ** self.alpha
            self.max_priority = max(self.max_priority, priority)
            self.tree.update(data_index, priority)

    def __len__(self) -> int:
        return self.tree.count

    def is_ready(self, batch_size: int) -> bool:
        return len(self) >= batch_size
